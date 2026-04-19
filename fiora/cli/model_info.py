#! /usr/bin/env python
import argparse
import json
import os

from fiora.GNN.FioraModel import FioraModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='fiora-model-info',
        description='Load a FIORA .pt checkpoint and print key model parameters.',
    )
    parser.add_argument(
        '-m',
        '--model',
        required=True,
        help='Path to checkpoint .pt file.',
    )
    parser.add_argument(
        '--as-json',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Print the full model_params dictionary as JSON.',
    )
    return parser.parse_args()


def _load_model(path: str):
    state_path = path.replace('.pt', '_state.pt')
    params_path = path.replace('.pt', '_params.json')
    if os.path.exists(state_path) and os.path.exists(params_path):
        return FioraModel.load_from_state_dict(path)
    return FioraModel.load(path)


def _get(params: dict, key: str, default='-'):
    return params.get(key, default)


def _print_main_params(params: dict) -> None:
    keys = [
        'version',
        'version_number',
        'param_tag',
        'training_label',
        'gnn_type',
        'depth',
        'hidden_dimension',
        'embedding_dimension',
        'embedding_aggregation',
        'layer_stacking',
        'residual_connections',
        'layer_norm',
        'subgraph_features',
        'pooling_func',
        'dense_layers',
        'dense_dim',
        'input_dropout',
        'latent_dropout',
        'output_dimension',
        'static_feature_dimension',
        'static_rt_feature_dimension',
        'prepare_additional_layers',
        'rt_supported',
        'ccs_supported',
    ]
    print('Model parameters:')
    for key in keys:
        print(f'  {key}: {_get(params, key)}')

    # concise feature summary
    atom_features = _get(params, 'atom_features', [])
    setup_features = _get(params, 'setup_features', [])
    rt_features = _get(params, 'rt_features', [])
    print('  atom_features:', atom_features if atom_features else '-')
    print('  setup_features:', setup_features if setup_features else '-')
    print('  rt_features:', rt_features if rt_features else '-')


def main() -> None:
    args = parse_args()
    model = _load_model(args.model)
    params = model.model_params if hasattr(model, 'model_params') else {}
    print(f'Loaded model: {args.model}')
    if args.as_json:
        print(json.dumps(params, indent=2, sort_keys=True))
        return
    _print_main_params(params)


if __name__ == '__main__':
    main()
