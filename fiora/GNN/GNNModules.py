import torch
from fiora.GNN.GNNLayers import GCNLayer
from fiora.GNN.FeatureEmbedding import FeatureEmbedding, FeatureEmbeddingPacked
import torch_geometric.nn as geom_nn
from typing import Literal
import dill
import json

'''
Geometric Models
'''


GeometricLayer = {
    "GraphConv": {
        "Layer": geom_nn.GraphConv,
        "divide_output_dim": False,
        "const_args": {'aggr': 'mean'},
        "batch_args": {'edge_index': 'edge_index'}
    },
    "GAT": {
        "Layer": geom_nn.GATConv,
        "divide_output_dim": True,
        "const_args": {'heads': 5},
        "batch_args": {'edge_index': 'edge_index', 'edge_attr': 'edge_embedding'}
    },
    
    "RGCNConv": {
        "Layer": geom_nn.RGCNConv,
        "divide_output_dim": False,
        "const_args": {'aggr': 'mean', 'num_relations': 4},
        "batch_args": {'edge_index': 'edge_index', 'edge_type': 'edge_type'}
    },
    
    "TransformerConv": {
        "Layer": geom_nn.TransformerConv,
        "divide_output_dim": True,
        "const_args": {'heads': 5, 'edge_dim': 300},
        "batch_args": {'edge_index': 'edge_index', 'edge_attr': 'edge_embedding'}
    },
    
    "CGConv": {
        "Layer": geom_nn.CGConv,
        "divide_output_dim": False,
        "const_args": {'aggr': "mean"}, #, 'dim': 300},
        "batch_args": {'edge_index': 'edge_index', 'edge_attr': 'edge_embedding'}
    }
    
    
    
}


class GNN(torch.nn.Module):
    def __init__(self, hidden_features: int, depth: int, dense_depth: int=0, embedding_dim: int=None, embedding_aggregation_type: str='concat', gnn_type: str="GraphConv", input_dropout: float=0, latent_dropout: float=0) -> None:
        super().__init__()

        self.activation = torch.nn.ELU()
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.latent_dropout = torch.nn.Dropout(latent_dropout)
        self.gnn_type = gnn_type
        node_features =  embedding_dim
        
        
        layers = []
        for _ in range(depth):
            layers += [GeometricLayer[gnn_type]["Layer"](node_features, int(hidden_features / GeometricLayer[gnn_type]["const_args"]["heads"]) if GeometricLayer[gnn_type]["divide_output_dim"] else hidden_features, **GeometricLayer[gnn_type]["const_args"])]
            node_features = hidden_features

        self.graph_layers = torch.nn.ModuleList(layers)


    def forward(self, batch):


        #print(batch["edge_features"].shape)
        X = batch["node_embedding"]
        X = self.input_dropout(X)
        
        # Apply graph layers
        batch_args = {key: batch[value] for key, value in GeometricLayer[self.gnn_type]["batch_args"].items()}
        for layer in self.graph_layers:
            X = self.activation(layer(X, **batch_args))
            X = self.latent_dropout(X)
            
        return X



class EdgePredictor(torch.nn.Module):
    def __init__(self, edge_feature_dict, hidden_features: int, static_features: int, out_dimension: int, dense_depth: int=0, embedding_dim: int=200, embedding_aggregation_type: str='concat', input_dropout: float=0, latent_dropout: float=0) -> None:
        super().__init__()

        self.activation = torch.nn.ELU()
        #self.edge_embedding = FeatureEmbedding(edge_feature_dict, embedding_dim, aggregation_type=embedding_aggregation_type)
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.latent_dropout = torch.nn.Dropout(latent_dropout)
        
        dense_layers = []
        num_features = hidden_features*2 + embedding_dim + static_features
        for _ in range(dense_depth):
            dense_layers += [torch.nn.Linear(num_features, num_features)]
        self.dense_layers = torch.nn.ModuleList(dense_layers)
        
        self.output_layer = torch.nn.Linear(num_features, out_dimension)

    def concat_node_pairs(self, X, edge_index):
        src, dst = edge_index
    
        return torch.cat([X[src], X[dst]], dim=1)
        
    def forward(self, X, batch):
        
        # Melt node features into a stack of edges (represented by left and right node)
        X = self.concat_node_pairs(X, batch["edge_index"])
        
        # Add edge features and static features 
        edge_features = batch["edge_embedding"] #self.edge_embedding(batch["edge_attr"])
        edge_features = self.input_dropout(edge_features)
        X = torch.cat([X, edge_features, batch["static_edge_features"]], axis=-1) #self.input_dropout(batch["static_edge_features"])
        
        # Apply fully connected layers
        for layer in self.dense_layers:
            X = self.activation(layer(X))
            X = self.latent_dropout(X)
        
        logits = self.output_layer(X)
        return logits
    

class GraphPredictor(torch.nn.Module):
    def __init__(self, hidden_features: int, static_features: int, out_dimension: int, dense_depth: int=0, input_dropout: float=0, latent_dropout: float=0) -> None:
        super().__init__()

        self.activation = torch.nn.ELU()
        self.pooling_func = geom_nn.global_mean_pool
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.latent_dropout = torch.nn.Dropout(latent_dropout)

        dense_layers = []
        num_features = hidden_features + static_features
        for _ in range(dense_depth):
            dense_layers += [torch.nn.Linear(num_features, num_features)]
        self.dense_layers = torch.nn.ModuleList(dense_layers)
        
        self.output_layer = torch.nn.Linear(num_features, out_dimension)
        
    def forward(self, X, batch, covariate_tag="static_graph_features"):
        X = self.pooling_func(X, batch["batch"])
        X = torch.cat([X, batch[covariate_tag]], axis=-1) # self.input_dropout(batch["static_graph_features"])
    
        for layer in self.dense_layers:
            X = self.activation(layer(X))
            X = self.latent_dropout(X)

               
        logits = self.output_layer(X)
        
        return logits
    



class GNNCompiler(torch.nn.Module):
    def __init__(self, model_params) -> None:
        super().__init__()
        self.edge_dim = model_params["output_dimension"]
        self.node_embedding = FeatureEmbedding(model_params["node_feature_layout"], model_params["embedding_dimension"], aggregation_type=model_params["embedding_aggregation"])
        self.edge_embedding = FeatureEmbedding(model_params["edge_feature_layout"], model_params["embedding_dimension"], aggregation_type=model_params["embedding_aggregation"])
        self.GNN_module = GNN(model_params["hidden_dimension"], model_params["depth"], model_params["dense_layers"], self.node_embedding.get_embedding_dimension(), model_params["embedding_aggregation"], model_params["gnn_type"], model_params["input_dropout"], model_params["latent_dropout"])
        self.edge_module = EdgePredictor(model_params["edge_feature_layout"], model_params["hidden_dimension"], model_params["static_feature_dimension"], model_params["output_dimension"], model_params["dense_layers"], self.edge_embedding.get_embedding_dimension(), model_params["embedding_aggregation"], model_params["input_dropout"], model_params["latent_dropout"])
        self.precursor_module = GraphPredictor(model_params["hidden_dimension"], model_params["static_feature_dimension"], 1, model_params["dense_layers"],  model_params["input_dropout"], model_params["latent_dropout"])
        
        self.RT_module = GraphPredictor(model_params["hidden_dimension"], model_params["static_rt_feature_dimension"], 1, model_params["dense_layers"],  model_params["input_dropout"], model_params["latent_dropout"])
        self.CCS_module = GraphPredictor(model_params["hidden_dimension"], model_params["static_rt_feature_dimension"], 1, model_params["dense_layers"],  model_params["input_dropout"], model_params["latent_dropout"])
        
        self.set_transform("double_softmax")
        self.model_params = model_params
    
    def freeze_submodule(self, submodule_name: str):
        module = getattr(self, submodule_name)
        for param in module.parameters():
            param.requires_grad = False
    
    def unfreeze_submodule(self, submodule_name: str):
        module = getattr(self, submodule_name)
        for param in module.parameters():
            param.requires_grad = True
    
    def set_transform(self, transformation: Literal["softmax", "double_softmax", "off"]):
        self.softmax = torch.nn.Softmax(dim=0)
        if transformation == "double_softmax":
            self.transform = lambda y: 2. *self.softmax(y) # TODO make torch module
        elif transformation == "softmax":
            self.transform = self.softmax
        elif transformation == "off":
            self.transform = torch.nn.Identity()
        else:
            raise ValueError(f"Unknow transformation type: {transformation}")
            

    def compile_output(self, edge_values, graph_values, batch):
        
        output = torch.zeros(edge_values.shape[0] * edge_values.shape[1] + graph_values.shape[0] * 2, device=edge_values.device)
        batch_ptr = 0
        edge_graph_map = batch["batch"][torch.repeat_interleave(batch["edge_index"][0,:], self.edge_dim)] # edge to batch (graph) matching, though edges are repeated as output dimension increases
        for i in range(batch.num_graphs):
            edges = edge_values.flatten()[edge_graph_map == i]
            #edge_values[edge_graph_map == i] = self.transform(edge_values[edge_graph_map == i])
            offset = edges.shape[0] + graph_values.shape[1] * 2 # output size per graph: all edge values + 2*graph_value (multiply by 2 to account for doubled eges)
            output[batch_ptr:batch_ptr + offset,] = self.transform(torch.cat([edges, graph_values[i], graph_values[i]], axis=-1)) # probably: flatten graph_values too
            batch_ptr += offset

        return output
        
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
        fragment_probs = self.compile_output(edge_values, graph_values, batch)        
        
        output = {'fragment_probs': fragment_probs}
        
        if with_RT:
            rt_values = self.RT_module(X, batch, covariate_tag="static_rt_features")
            output["rt"] = rt_values
        
        if with_CCS:
            ccs_values = self.CCS_module(X, batch, covariate_tag="static_rt_features")
            output["ccs"] = ccs_values

        return output
         
    @classmethod
    def load(cls, PATH: str) -> 'GNNCompiler':
        
        with open(PATH, 'rb') as f:
            model = dill.load(f)

        if not isinstance(model, cls):
            raise ValueError(f'file {PATH} contains incorrect model class {type(model)}')

        return model
    
    @classmethod
    def load_from_state_dict(cls, PATH: str) -> 'GNNCompiler':

        PARAMS_PATH = PATH.replace(".pt", "_params.json")
        STATE_PATH = PATH.replace(".pt", "_state.pt")
        
        with open(PARAMS_PATH, 'r') as fp:
            params = json.load(fp)
        model = GNNCompiler(params)
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

    # def save(self, PATH: str, dev: str="cpu") -> None:
        
    #     # Set device to cpu for saving
    #     prev_device = next(self.parameters()).device
    #     self.to(dev)
    #     with open(PATH, 'wb') as f:
    #         dill.dump(self.to(dev), f)

    #     params_path = '.'.join(PATH.split('.')[:-1]) + '_params.json'
    #     with open(params_path, 'w') as fp:
    #         json.dump(self.model_params, fp)
        
    #     state_dict_path = params_path.replace("_params.json", "_state.pt")
    #     self.to(dev)

    #     torch.save(self.state_dict(), state_dict_path, _use_new_zipfile_serialization=False)
        
    #     self.to(prev_device)