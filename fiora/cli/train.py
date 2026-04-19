#! /usr/bin/env python
import argparse
import ast
import json
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from sklearn.model_selection import train_test_split

from fiora.GNN.AtomFeatureEncoder import AtomFeatureEncoder
from fiora.GNN.BondFeatureEncoder import BondFeatureEncoder
from fiora.GNN.CovariateFeatureEncoder import CovariateFeatureEncoder
from fiora.GNN.fabric_training import seed_everything, train_fabric_loop
from fiora.GNN.FioraModel import FioraModel
from fiora.GNN.Losses import (
    GraphwiseKLLoss,
    GraphwiseKLLossMetric,
    WeightedMAELoss,
    WeightedMAEMetric,
    WeightedMSELoss,
    WeightedMSEMetric,
)
from fiora.IO.LibraryLoader import LibraryLoader
from fiora.MOL.constants import DEFAULT_MODES, DEFAULT_PPM
from fiora.MOL.Metabolite import Metabolite
from fiora.MOL.MetaboliteIndex import MetaboliteIndex

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', category=SyntaxWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='fiora-train',
        description='Train a FIORA model from a preprocessed library CSV.',
    )
    parser.add_argument(
        '-i',
        '--input',
        required=True,
        help='Path to preprocessed CSV containing spectra, metadata, and SMILES.',
    )
    parser.add_argument(
        '-o',
        '--output',
        default='checkpoint_fiora.best.pt',
        help='Output path for best checkpoint (.pt).',
    )
    parser.add_argument(
        '--model-params',
        help='Optional path to a JSON file with base model parameters.',
        default=None,
    )
    parser.add_argument(
        '--resume',
        help='Optional path to a checkpoint to resume from (.pt).',
        default=None,
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='Device to run on (e.g. cpu, cuda:0). Default: auto.',
    )
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument(
        '--hidden-dimension',
        type=int,
        default=None,
        help='Override model hidden dimension (default from model params).',
    )
    parser.add_argument(
        '--embedding-dimension',
        type=int,
        default=None,
        help='Override embedding dimension (default from model params).',
    )
    parser.add_argument(
        '--dense-dim',
        type=int,
        default=None,
        help='Override dense layer hidden dimension (None keeps current setting).',
    )
    parser.add_argument(
        '--residual-connections',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Override residual connections setting.',
    )
    parser.add_argument(
        '--layer-stacking',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Override layer stacking setting.',
    )
    parser.add_argument(
        '--loss',
        choices=['graphwise_kl', 'weighted_mse', 'weighted_mae', 'mse'],
        default='graphwise_kl',
    )
    parser.add_argument(
        '--precursor-loss-weight',
        type=float,
        default=1.0,
        help='Multiplier for precursor positions in fragment loss (1.0 keeps original weighting).',
    )
    parser.add_argument(
        '--y-label',
        default='compiled_probsALL',
        help='Label to use as training target.',
    )
    parser.add_argument(
        '--with-rt',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Train RT head if available.',
    )
    parser.add_argument(
        '--with-ccs',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Train CCS head if available.',
    )
    parser.add_argument('--train-val-split', type=float, default=0.8)
    parser.add_argument(
        '--split-by-group',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Split train/val by group_id (prevents leakage).',
    )
    parser.add_argument('--group-id-col', default='group_id')
    parser.add_argument('--datasplit-col', default='datasplit')
    parser.add_argument('--train-label', default='training')
    parser.add_argument('--val-label', default='validation')
    parser.add_argument('--min-peak-matches', type=int, default=2)
    parser.add_argument(
        '--ppm',
        type=float,
        default=None,
        help='Default ppm tolerance if column missing.',
    )
    parser.add_argument('--ppm-col', default='ppm_peak_tolerance')
    parser.add_argument('--summary-col', default='summary')
    parser.add_argument('--peaks-col', default='peaks')
    parser.add_argument('--smiles-col', default='SMILES')
    parser.add_argument('--loss-weight-col', default='loss_weight')
    parser.add_argument('--max-rows', type=int, default=None)
    parser.add_argument('--fragmentation-depth', type=int, default=1)
    parser.add_argument(
        '--use-frag-index',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use MetaboliteIndex to cache fragmentation trees.',
    )
    parser.add_argument(
        '--graph-mismatch-policy',
        choices=['recompute', 'ignore'],
        default='recompute',
    )
    parser.add_argument(
        '--precursor-modes',
        default=None,
        help='Comma-separated precursor modes to encode.',
    )
    parser.add_argument(
        '--instruments',
        default=None,
        help='Comma-separated instrument types to encode.',
    )
    parser.add_argument('--ce-upper-limit', type=float, default=100.0)
    parser.add_argument('--weight-upper-limit', type=float, default=1000.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument(
        '--pin-memory',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Pin host memory for DataLoader (auto: enabled for CUDA).',
    )
    parser.add_argument('--val-every', type=int, default=1)
    parser.add_argument(
        '--use-validation-mask',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Use validation mask during validation.',
    )
    parser.add_argument('--validation-mask-name', default='validation_mask')
    parser.add_argument(
        '--scheduler',
        choices=['plateau', 'none'],
        default='plateau',
    )
    parser.add_argument('--scheduler-patience', type=int, default=8)
    parser.add_argument('--scheduler-factor', type=float, default=0.5)
    parser.add_argument(
        '--rt-metric',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Track RT/CCS metrics instead of fragment metrics.',
    )
    parser.add_argument(
        '--index-col',
        type=int,
        default=0,
        help='CSV index column (default: 0). Use --no-index-col to disable.',
    )
    parser.add_argument(
        '--no-index-col',
        action='store_true',
        help='Disable index_col when reading CSV.',
    )
    parser.add_argument(
        '--history-out',
        default=None,
        help='Optional path to save training history (.json or .csv).',
    )
    return parser.parse_args()


def _parse_dict(val):
    if isinstance(val, dict):
        return val
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    text = str(val).strip()
    if not text:
        return None
    try:
        # Handles canonical JSON and JSON with NaN/Infinity tokens.
        return json.loads(text)
    except Exception:
        pass
    # Fallback for python-literal style dict strings.
    norm = re.sub(r'\b(?:NaN|nan)\b', 'None', text)
    norm = re.sub(r'\b(?:Infinity|inf)\b', '1e309', norm)
    norm = re.sub(r'\b(?:-Infinity|-inf)\b', '-1e309', norm)
    try:
        parsed = ast.literal_eval(norm)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _parse_dict_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(_parse_dict)
    return df


def _safe_metabolite(smiles: str):
    try:
        return Metabolite(smiles)
    except Exception:
        return None


def _is_missing_value(val) -> bool:
    return val is None or (isinstance(val, float) and np.isnan(val))


def _build_summary_from_record(record: dict, metadata_key_map) -> dict:
    summary = {}
    for key, cols in metadata_key_map.items():
        if not isinstance(cols, (list, tuple)):
            cols = [cols]
        for col in cols:
            if col in record:
                value = record[col]
                if not _is_missing_value(value):
                    summary[key] = value
                    break
    return summary


def _parallel_map(func, tasks, num_workers: int):
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            yield from executor.map(func, tasks)
    else:
        for task in tasks:
            yield func(task)


def _progress_iterator(iterable, total: int, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _prepare_metabolite_task(task):
    (
        idx,
        record,
        smiles_col,
        group_id_col,
        summary_col,
        loss_weight_col,
        metadata_key_map,
        node_encoder,
        bond_encoder,
        covariate_encoder,
        rt_encoder,
    ) = task

    smiles = record.get(smiles_col)
    if _is_missing_value(smiles):
        return idx, None

    mol = _safe_metabolite(smiles)
    if mol is None:
        return idx, None

    try:
        mol.create_molecular_structure_graph()
        mol.compute_graph_attributes(node_encoder, bond_encoder)
    except Exception:
        return idx, None

    if group_id_col in record:
        group_id = record.get(group_id_col)
        if not _is_missing_value(group_id):
            try:
                mol.set_id(int(group_id))
            except Exception:
                pass

    summary = record.get(summary_col) if summary_col in record else None
    if summary is None:
        summary = _build_summary_from_record(record, metadata_key_map)

    try:
        mol.add_metadata(summary, covariate_encoder, rt_encoder)
    except Exception:
        return idx, None

    loss_weight = record.get(loss_weight_col) if loss_weight_col in record else None
    if not _is_missing_value(loss_weight):
        try:
            mol.set_loss_weight(float(loss_weight))
        except Exception:
            mol.set_loss_weight(1.0)
    else:
        mol.set_loss_weight(1.0)

    return idx, mol


def _resolve_tolerance(record: dict, ppm_col: str, ppm_default: float) -> float:
    tol = ppm_default
    if ppm_col in record:
        try:
            val = float(record[ppm_col])
            if not np.isnan(val):
                tol = val
        except Exception:
            pass
    return tol


def _match_peaks_task(task):
    idx, metabolite, peaks, tol = task
    if not isinstance(peaks, dict):
        return idx, False
    mz = peaks.get('mz')
    intensity = peaks.get('intensity')
    if mz is None or intensity is None or len(mz) == 0:
        return idx, False
    try:
        metabolite.match_fragments_to_peaks(mz, intensity, tolerance=tol)
        return idx, True
    except Exception:
        return idx, False


def _resolve_device(device: str) -> str:
    if device == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


def _load_model_params(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, 'r') as fp:
        return json.load(fp)


def _choose_loss(loss_name: str):
    if loss_name == 'graphwise_kl':
        return GraphwiseKLLoss(reduction='mean'), {'kl': GraphwiseKLLossMetric}
    if loss_name == 'weighted_mse':
        return WeightedMSELoss(), {'mse': WeightedMSEMetric}
    if loss_name == 'weighted_mae':
        return WeightedMAELoss(), {'mae': WeightedMAEMetric}
    if loss_name == 'mse':
        return torch.nn.MSELoss(), None
    raise ValueError(f'Unknown loss: {loss_name}')


def _save_history(history: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if output_path.lower().endswith('.csv'):
        pd.DataFrame(history).to_csv(output_path, index=False)
    else:
        with open(output_path, 'w') as fp:
            json.dump(history, fp, indent=2)


def _split_geo_data(
    geo_data,
    split_by_group: bool,
    train_val_split: float,
    seed: int,
    train_keys: list[int] | None = None,
    val_keys: list[int] | None = None,
):
    train_keys = train_keys or []
    val_keys = val_keys or []
    if len(geo_data) == 0:
        return [], []

    if split_by_group and hasattr(geo_data[0], 'group_id'):
        group_ids = np.array([int(getattr(x, 'group_id')) for x in geo_data])
        keys = np.unique(group_ids)
        if len(train_keys) > 0 and len(val_keys) > 0:
            train_set = set(int(x) for x in train_keys)
            val_set = set(int(x) for x in val_keys)
            print('Using pre-set train/validation keys')
        else:
            tr, va = train_test_split(
                keys, test_size=1 - train_val_split, random_state=seed
            )
            train_set = set(int(x) for x in tr)
            val_set = set(int(x) for x in va)
        train_data = [x for x in geo_data if int(getattr(x, 'group_id')) in train_set]
        val_data = [x for x in geo_data if int(getattr(x, 'group_id')) in val_set]
        return train_data, val_data

    train_size = int(len(geo_data) * train_val_split)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(geo_data))
    rng.shuffle(indices)
    train_idx = set(indices[:train_size].tolist())
    train_data = [geo_data[i] for i in range(len(geo_data)) if i in train_idx]
    val_data = [geo_data[i] for i in range(len(geo_data)) if i not in train_idx]
    return train_data, val_data


def main() -> None:
    args = parse_args()
    dev = _resolve_device(args.device)
    np.seterr(invalid='ignore')
    seed_everything(args.seed)

    index_col = None if args.no_index_col else args.index_col
    loader = LibraryLoader()
    df = (
        loader.load_from_csv(args.input)
        if index_col == 0
        else pd.read_csv(args.input, index_col=index_col, low_memory=False)
    )

    if args.max_rows:
        df = df.iloc[: args.max_rows].copy()

    df = _parse_dict_columns(df, [args.summary_col, args.peaks_col])

    # Prepare encoders
    overwrite_sets = {}
    if args.instruments:
        overwrite_sets['instrument'] = [
            x.strip() for x in args.instruments.split(',') if x.strip()
        ]
    if args.precursor_modes:
        overwrite_sets['precursor_mode'] = [
            x.strip() for x in args.precursor_modes.split(',') if x.strip()
        ]
    if not overwrite_sets:
        overwrite_sets = None

    node_encoder = AtomFeatureEncoder(
        feature_list=['symbol', 'num_hydrogen', 'ring_type']
    )
    bond_encoder = BondFeatureEncoder(feature_list=['bond_type', 'ring_type'])
    covariate_encoder = CovariateFeatureEncoder(
        feature_list=[
            'collision_energy',
            'molecular_weight',
            'precursor_mode',
            'instrument',
            'element_composition',
        ],
        sets_overwrite=overwrite_sets,
    )
    rt_encoder = CovariateFeatureEncoder(
        feature_list=[
            'molecular_weight',
            'precursor_mode',
            'instrument',
            'element_composition',
        ],
        sets_overwrite=overwrite_sets,
    )
    covariate_encoder.normalize_features['collision_energy']['max'] = (
        args.ce_upper_limit
    )
    covariate_encoder.normalize_features['molecular_weight']['max'] = (
        args.weight_upper_limit
    )
    rt_encoder.normalize_features['molecular_weight']['max'] = args.weight_upper_limit

    metadata_key_map = {
        'name': ['Name', 'NAME', 'Title', 'TITLE'],
        'collision_energy': ['CE', 'COLLISION_ENERGY', 'CollisionEnergy'],
        'instrument': ['Instrument_type', 'instrument', 'INSTRUMENT_TYPE'],
        'precursor_mode': ['Precursor_type', 'ADDUCT', 'PRECURSORTYPE'],
        'precursor_mz': ['PrecursorMZ', 'PEPMASS', 'PRECURSORMZ'],
        'retention_time': ['RETENTIONTIME', 'RTINSECONDS', 'retention_time'],
        'ccs': ['CCS', 'ccs'],
    }

    # Build metabolites
    invalid_rows = []
    metabolite_tasks = (
        (
            idx,
            row.to_dict(),
            args.smiles_col,
            args.group_id_col,
            args.summary_col,
            args.loss_weight_col,
            metadata_key_map,
            node_encoder,
            bond_encoder,
            covariate_encoder,
            rt_encoder,
        )
        for idx, row in df.iterrows()
    )
    metabolite_results = _parallel_map(
        _prepare_metabolite_task, metabolite_tasks, args.num_workers
    )
    for idx, mol in _progress_iterator(
        metabolite_results, total=len(df), desc='Building graphs'
    ):
        if mol is None:
            invalid_rows.append(idx)
            continue
        df.at[idx, 'Metabolite'] = mol

    if invalid_rows:
        df = df.drop(index=invalid_rows)
        print(f'Dropped {len(invalid_rows)} invalid rows.')

    # Fragmentation trees
    if args.use_frag_index:
        mindex = MetaboliteIndex()
        mindex.index_metabolites(df['Metabolite'])
        mindex.create_fragmentation_trees(depth=args.fragmentation_depth)
        mindex.add_fragmentation_trees_to_metabolite_list(
            df['Metabolite'], graph_mismatch_policy=args.graph_mismatch_policy
        )
    else:
        df['Metabolite'].apply(lambda x: x.fragment_MOL(depth=args.fragmentation_depth))

    # Match peaks to fragments
    ppm_default = args.ppm if args.ppm is not None else DEFAULT_PPM
    match_invalid = []
    match_tasks = (
        (
            idx,
            row['Metabolite'],
            row.get(args.peaks_col),
            _resolve_tolerance(row, args.ppm_col, ppm_default),
        )
        for idx, row in df.iterrows()
    )
    for idx, matched in _parallel_map(_match_peaks_task, match_tasks, args.num_workers):
        if not matched:
            match_invalid.append(idx)

    if match_invalid:
        df = df.drop(index=match_invalid)
        print(f'Dropped {len(match_invalid)} rows with invalid peaks.')

    df['num_peak_matches'] = df['Metabolite'].apply(
        lambda x: x.match_stats['num_peak_matches']
    )
    if args.min_peak_matches > 0:
        before = len(df)
        df = df[df['num_peak_matches'] >= args.min_peak_matches]
        print(
            f'Filtered {before - len(df)} rows with < {args.min_peak_matches} peak matches.'
        )

    # Train/val split
    train_keys = []
    val_keys = []
    if args.datasplit_col in df.columns:
        df_train = df[df[args.datasplit_col].isin([args.train_label, args.val_label])]
        if args.group_id_col in df.columns:
            train_keys = (
                df[df[args.datasplit_col] == args.train_label][args.group_id_col]
                .dropna()
                .unique()
                .tolist()
            )
            val_keys = (
                df[df[args.datasplit_col] == args.val_label][args.group_id_col]
                .dropna()
                .unique()
                .tolist()
            )
    else:
        df_train = df

    # Geometric data
    geo_data = []
    for _, row in df_train.iterrows():
        data = row['Metabolite'].as_geometric_data()
        if args.group_id_col in df_train.columns:
            try:
                data.group_id = int(row[args.group_id_col])
            except Exception:
                pass
        geo_data.append(data)
    print(f'Prepared training/validation with {len(geo_data)} data points')

    # Model params
    default_params = {
        'param_tag': 'default',
        'gnn_type': 'RGCNConv',
        'depth': 10,
        'hidden_dimension': 300,
        'residual_connections': False,
        'layer_stacking': True,
        'embedding_aggregation': 'concat',
        'embedding_dimension': 300,
        'subgraph_features': True,
        'pooling_func': 'max',
        'layer_norm': True,
        'dense_layers': 2,
        'dense_dim': 500,
        'input_dropout': 0.25,
        'latent_dropout': 0.25,
        'prepare_additional_layers': False,
        'rt_supported': False,
        'ccs_supported': False,
        'version': 'x.x.x',
    }
    base_params = _load_model_params(args.model_params)
    model_params = dict(default_params)
    model_params.update(base_params)
    model_params.update(
        {
            'node_feature_layout': node_encoder.feature_numbers,
            'edge_feature_layout': bond_encoder.feature_numbers,
            'static_feature_dimension': geo_data[0]['static_edge_features'].shape[1],
            'static_rt_feature_dimension': geo_data[0]['static_rt_features'].shape[1],
            'output_dimension': len(DEFAULT_MODES) * 2,
            'atom_features': node_encoder.feature_list,
            'setup_features': covariate_encoder.feature_list,
            'setup_features_categorical_set': covariate_encoder.categorical_sets,
            'rt_features': rt_encoder.feature_list,
            'prepare_additional_layers': args.with_rt or args.with_ccs,
            'rt_supported': args.with_rt,
            'ccs_supported': args.with_ccs,
        }
    )
    if args.hidden_dimension is not None:
        model_params['hidden_dimension'] = int(args.hidden_dimension)
    if args.embedding_dimension is not None:
        model_params['embedding_dimension'] = int(args.embedding_dimension)
    if args.dense_dim is not None:
        model_params['dense_dim'] = int(args.dense_dim)
    if args.residual_connections is not None:
        model_params['residual_connections'] = bool(args.residual_connections)
    if args.layer_stacking is not None:
        model_params['layer_stacking'] = bool(args.layer_stacking)
    if model_params.get('residual_connections', False):
        if (
            model_params.get('hidden_dimension')
            != model_params.get('embedding_dimension')
            and args.embedding_dimension is None
        ):
            model_params['embedding_dimension'] = model_params['hidden_dimension']
        if args.dense_dim is None and 'dense_dim' not in base_params:
            # Avoid shape-mismatch in dense residual blocks when using default params.
            model_params['dense_dim'] = None

    # Initialize or resume model
    if args.resume:
        state_path = args.resume.replace('.pt', '_state.pt')
        params_path = args.resume.replace('.pt', '_params.json')
        if os.path.exists(state_path) and os.path.exists(params_path):
            model = FioraModel.load_from_state_dict(args.resume)
        else:
            model = FioraModel.load(args.resume)
    else:
        model = FioraModel(model_params)

    if (args.with_rt or args.with_ccs) and not model.model_params.get(
        'prepare_additional_layers', False
    ):
        raise RuntimeError(
            'Model does not include RT/CCS heads but --with-rt/--with-ccs was set.'
        )
    model.model_params['training_label'] = args.y_label

    loss_fn, metric_dict = _choose_loss(args.loss)

    split_by_group = args.split_by_group and args.group_id_col in df_train.columns
    train_data, val_data = _split_geo_data(
        geo_data,
        split_by_group=split_by_group,
        train_val_split=args.train_val_split,
        seed=args.seed,
        train_keys=train_keys,
        val_keys=val_keys,
    )
    has_validation = len(val_data) > 0
    print(f'Train/validation split: {len(train_data)} / {len(val_data)}')

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    checkpoints, history = train_fabric_loop(
        model=model,
        train_data=train_data,
        val_data=val_data,
        loss_fn=loss_fn,
        metric_dict=metric_dict,
        y_label=args.y_label,
        device=dev,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        val_every=args.val_every,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_name=args.scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        with_rt=args.with_rt,
        with_ccs=args.with_ccs,
        rt_metric=args.rt_metric,
        use_validation_mask=args.use_validation_mask,
        validation_mask_name=args.validation_mask_name,
        output_path=output_path,
        logger=print,
        pin_memory=args.pin_memory,
        precursor_loss_weight=args.precursor_loss_weight,
    )
    if args.history_out:
        _save_history(history, args.history_out)
        print(f'Saved training history to {args.history_out}')
    print(f'Finished training. Best checkpoint: {checkpoints["file"]}')


if __name__ == '__main__':
    main()
