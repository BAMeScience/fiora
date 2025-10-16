import torch
import torch_geometric.nn as geom_nn

# Fiora GNN Modules
from fiora.GNN.FeatureEmbedding import FeatureEmbedding, FeatureEmbeddingPacked
from fiora.GNN.GNN import GNN
from fiora.GNN.GraphPropertyPredictor import GraphPropertyPredictor
from fiora.GNN.EdgePropertyPredictor import EdgePropertyPredictor

# Misc
from typing import Literal, Dict
import dill
import json


class FioraModel(torch.nn.Module):
    def __init__(self, model_params: Dict) -> None:
        ''' Initialize the FioraModel with the given parameters.
            Args:
                model_params (Dict): Dictionary containing model parameters such as node/edge feature layouts, embedding dimensions, hidden dimensions, etc.
        '''
        super().__init__()
        
        self._version_control_model_params(model_params)

        self.edge_dim = model_params["output_dimension"]
        self.node_embedding = FeatureEmbedding(feature_dict=model_params["node_feature_layout"], dim=model_params["embedding_dimension"], aggregation_type=model_params["embedding_aggregation"])
        self.edge_embedding = FeatureEmbedding(feature_dict=model_params["edge_feature_layout"], dim=model_params["embedding_dimension"], aggregation_type=model_params["embedding_aggregation"])
        self.GNN_module = GNN(hidden_features=model_params["hidden_dimension"], depth=model_params["depth"], embedding_dim=self.node_embedding.get_embedding_dimension(), embedding_aggregation_type=model_params["embedding_aggregation"], gnn_type=model_params["gnn_type"], layer_norm=model_params["layer_norm"], residual_connections=model_params["residual_connections"], layer_stacking=model_params["layer_stacking"], input_dropout=model_params["input_dropout"], latent_dropout=model_params["latent_dropout"])
        self.edge_module = EdgePropertyPredictor(edge_feature_dict=model_params["edge_feature_layout"], hidden_features=self.GNN_module.get_embedding_dimension(), static_features=model_params["static_feature_dimension"], out_dimension=model_params["output_dimension"], dense_depth=model_params["dense_layers"], dense_dim=model_params["dense_dim"], embedding_dim=self.edge_embedding.get_embedding_dimension(), embedding_aggregation_type=model_params["embedding_aggregation"], residual_connections=model_params["residual_connections"], subgraph_features=model_params["subgraph_features"], pooling_func=model_params["pooling_func"], input_dropout=model_params["input_dropout"], latent_dropout=model_params["latent_dropout"])
        self.precursor_module = GraphPropertyPredictor(hidden_features=self.GNN_module.get_embedding_dimension(), static_features=model_params["static_feature_dimension"], out_dimension=1, dense_depth=model_params["dense_layers"], dense_dim=model_params["dense_dim"], residual_connections=model_params["residual_connections"], pooling_func=model_params["pooling_func"], input_dropout=model_params["input_dropout"], latent_dropout=model_params["latent_dropout"])
        
        if model_params["prepare_additional_layers"]:
            self.RT_module = GraphPropertyPredictor(hidden_features=self.GNN_module.get_embedding_dimension(), static_features=model_params["static_rt_feature_dimension"], out_dimension=1, dense_depth=model_params["dense_layers"], dense_dim=model_params["dense_dim"], residual_connections=model_params["residual_connections"], pooling_func=model_params["pooling_func"], input_dropout=model_params["input_dropout"], latent_dropout=model_params["latent_dropout"])
            self.CCS_module = GraphPropertyPredictor(hidden_features=self.GNN_module.get_embedding_dimension(), static_features=model_params["static_rt_feature_dimension"], out_dimension=1, dense_depth=model_params["dense_layers"], dense_dim=model_params["dense_dim"], residual_connections=model_params["residual_connections"],  pooling_func=model_params["pooling_func"], input_dropout=model_params["input_dropout"], latent_dropout=model_params["latent_dropout"])
        
        
        self.set_transform("double_softmax")
        self.model_params = model_params
    
    def _version_control_model_params(self, model_params: Dict) -> None:
        ''' Update model parameters to match the latest model version.
            Args:
                model_params (Dict): Dictionary containing model parameters.
        '''

        if "residual_connections" not in model_params:
            model_params["residual_connections"] = False
        if "layer_stacking" not in model_params:
            model_params["layer_stacking"] = False
        if "prepare_additional_layers" not in model_params:
            model_params["prepare_additional_layers"] = True # Defaults to True, since old models have RT/CCS modules
        if "dense_dim" not in model_params:
            model_params["dense_dim"] = None # None defaults to the number of input features (GNN output dimension)
        if "subgraph_features" not in model_params: # No subgraph features in older models
            model_params["subgraph_features"] = False
        if "pooling" not in model_params:
            model_params["pooling_func"] = "avg"
        if "layer_norm" not in model_params:
            model_params["layer_norm"] = False

        return


    def freeze_submodule(self, submodule_name: str):
        module = getattr(self, submodule_name)
        for param in module.parameters():
            param.requires_grad = False
    
    def unfreeze_submodule(self, submodule_name: str):
        module = getattr(self, submodule_name)
        for param in module.parameters():
            param.requires_grad = True
    
    def set_dropout_rate(self, input_dropout: float, latent_dropout: float) -> None:
        
        self.GNN_module.input_dropout.p = input_dropout
        self.GNN_module.latent_dropout.p = latent_dropout
        self.edge_module.input_dropout.p = input_dropout
        self.edge_module.latent_dropout.p = latent_dropout
        self.precursor_module.input_dropout.p = input_dropout
        self.precursor_module.latent_dropout.p = latent_dropout
        self.RT_module.input_dropout.p = input_dropout
        self.RT_module.latent_dropout.p = latent_dropout
        self.CCS_module.input_dropout.p = input_dropout
        self.CCS_module.latent_dropout.p = latent_dropout
    
    def set_transform(self, transformation: Literal["softmax", "double_softmax", "off"]):
        self.softmax = torch.nn.Softmax(dim=0)
        if transformation == "double_softmax":
            self.transform = lambda y: 2. *self.softmax(y) # TODO make torch module
        elif transformation == "softmax":
            self.transform = self.softmax
        elif transformation == "off":
            self.transform = torch.nn.Identity()
        else:
            raise ValueError(f"Unknown transformation type: {transformation}")
            

    '''
        Compile output is the heart of the fragment probability prediction.
        
        It combines edge/fragment prediction with precursor prediction for each individual graph/input and applies softmax. 
        Then, all output values are stacked in a single dimension.
    '''
    def _compile_output(self, edge_values, graph_values, batch) -> torch.tensor:
        output = torch.zeros(edge_values.shape[0] * edge_values.shape[1] + graph_values.shape[0] * 2, device=edge_values.device)
        batch_ptr = 0
        segment_ptr = [0]  # cumulative boundaries per-graph (len=num_graphs+1)
        
        # Map edges to graph index (repeat left nodes according to edge dimension and retrieve graph/batch index)
        edge_graph_map = batch["batch"][torch.repeat_interleave(batch["edge_index"][0,:], self.edge_dim)]
        for i in range(batch.num_graphs):
            edges = edge_values.flatten()[edge_graph_map == i] # Retrieve edge_values for graph i 
            offset = edges.shape[0] + graph_values.shape[1] * 2 # Precursor prediction output is repeated to account for bi-directional edge occurances
            output[batch_ptr:batch_ptr + offset,] = self.transform(torch.cat([edges, graph_values[i], graph_values[i]], axis=-1)) # concat and apply softmax
            batch_ptr += offset
            segment_ptr.append(batch_ptr)

        return output, torch.tensor(segment_ptr, device=edge_values.device, dtype=torch.long)
        
    def get_graph_embedding(self, batch):
        batch["node_embedding"] = self.node_embedding(batch["x"])
        batch["edge_embedding"] = self.edge_embedding(batch["edge_attr"])
        X = self.GNN_module(batch)
        pooling_func = self.precursor_module.pooling_func
        return pooling_func(X, batch["batch"])
        

    def forward(self, batch, with_RT=False, with_CCS=False):

        # Embed node features
        batch["node_embedding"] = self.node_embedding(batch["x"])
        batch["edge_embedding"] = self.edge_embedding(batch["edge_attr"])
        
        X = self.GNN_module(batch)
        
        edge_values = self.edge_module(X, batch)
        graph_values = self.precursor_module(X, batch)
        fragment_probs, segment_ptr = self._compile_output(edge_values, graph_values, batch)        
        
        output = {'fragment_probs': fragment_probs, 'segment_ptr': segment_ptr}
        
        if with_RT:
            rt_values = self.RT_module(X, batch, covariate_tag="static_rt_features")
            output["rt"] = rt_values
        
        if with_CCS:
            ccs_values = self.CCS_module(X, batch, covariate_tag="static_rt_features")
            output["ccs"] = ccs_values

        return output
    
    @classmethod
    def load(cls, PATH: str) -> 'FioraModel':
        
        with open(PATH, 'rb') as f:
            model = dill.load(f)

        if not isinstance(model, cls):
            raise ValueError(f'file {PATH} contains incorrect model class {type(model)}')

        return model
    
    @classmethod
    def load_from_state_dict(cls, PATH: str) -> 'FioraModel':

        PARAMS_PATH = PATH.replace(".pt", "_params.json")
        STATE_PATH = PATH.replace(".pt", "_state.pt")
        
        with open(PARAMS_PATH, 'r') as fp:
            params = json.load(fp)
        model = FioraModel(params)
        model.load_state_dict(torch.load(STATE_PATH, map_location=torch.serialization.default_restore_location, weights_only=True))

        if not isinstance(model, cls):
            raise ValueError(f'file {PATH} contains incorrect model class {type(model)}')

        return model
    
    def save(self, PATH: str, dev: str="cpu") -> None:
        
        prev_device = next(self.parameters()).device
        
        # Set device to cpu for saving
        self.to(dev)
        with open(PATH, 'wb') as f:
            dill.dump(self.to(dev), f)
        
        # Save state_dict and parameters as backup
        PATH = '.'.join(PATH.split('.')[:-1]) + '_params.json'
        with open(PATH, 'w') as fp:
            json.dump(self.model_params, fp)

        PATH = PATH.replace("_params.json", "_state.pt")
        torch.save(self.to(dev).state_dict(), PATH)
        
        #Reset to previous device
        self.to(prev_device)