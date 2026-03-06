#! /usr/bin/env python
import argparse
import ast
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

from fiora.GNN.AtomFeatureEncoder import AtomFeatureEncoder
from fiora.GNN.BondFeatureEncoder import BondFeatureEncoder
from fiora.GNN.CovariateFeatureEncoder import CovariateFeatureEncoder
from fiora.GNN.FioraModel import FioraModel
from fiora.GNN.Losses import (
    GraphwiseKLLoss,
    GraphwiseKLLossMetric,
    WeightedMAELoss,
    WeightedMAEMetric,
    WeightedMSELoss,
    WeightedMSEMetric,
)
from fiora.GNN.SpectralTrainer import SpectralTrainer
from fiora.IO.LibraryLoader import LibraryLoader
from fiora.MOL.Metabolite import Metabolite
from fiora.MOL.MetaboliteIndex import MetaboliteIndex
from fiora.MOL.constants import DEFAULT_MODES, DEFAULT_PPM

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=SyntaxWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fiora-train",
        description="Train a FIORA model from a preprocessed library CSV.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to preprocessed CSV containing spectra, metadata, and SMILES.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="checkpoint_fiora.best.pt",
        help="Output path for best checkpoint (.pt).",
    )
    parser.add_argument(
        "--model-params",
        help="Optional path to a JSON file with base model parameters.",
        default=None,
    )
    parser.add_argument(
        "--resume",
        help="Optional path to a checkpoint to resume from (.pt).",
        default=None,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on (e.g. cpu, cuda:0). Default: auto.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--loss",
        choices=["graphwise_kl", "weighted_mse", "weighted_mae", "mse"],
        default="graphwise_kl",
    )
    parser.add_argument(
        "--y-label",
        default="compiled_probsALL",
        help="Label to use as training target.",
    )
    parser.add_argument(
        "--with-rt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train RT head if available.",
    )
    parser.add_argument(
        "--with-ccs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train CCS head if available.",
    )
    parser.add_argument("--train-val-split", type=float, default=0.8)
    parser.add_argument(
        "--split-by-group",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split train/val by group_id (prevents leakage).",
    )
    parser.add_argument("--group-id-col", default="group_id")
    parser.add_argument("--datasplit-col", default="datasplit")
    parser.add_argument("--train-label", default="training")
    parser.add_argument("--val-label", default="validation")
    parser.add_argument("--min-peak-matches", type=int, default=2)
    parser.add_argument(
        "--ppm",
        type=float,
        default=None,
        help="Default ppm tolerance if column missing.",
    )
    parser.add_argument("--ppm-col", default="ppm_peak_tolerance")
    parser.add_argument("--summary-col", default="summary")
    parser.add_argument("--peaks-col", default="peaks")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--loss-weight-col", default="loss_weight")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--fragmentation-depth", type=int, default=1)
    parser.add_argument(
        "--use-frag-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MetaboliteIndex to cache fragmentation trees.",
    )
    parser.add_argument(
        "--graph-mismatch-policy",
        choices=["recompute", "ignore"],
        default="recompute",
    )
    parser.add_argument(
        "--precursor-modes",
        default=None,
        help="Comma-separated precursor modes to encode.",
    )
    parser.add_argument(
        "--instruments",
        default=None,
        help="Comma-separated instrument types to encode.",
    )
    parser.add_argument("--ce-upper-limit", type=float, default=100.0)
    parser.add_argument("--weight-upper-limit", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument(
        "--use-validation-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use validation mask during validation.",
    )
    parser.add_argument("--validation-mask-name", default="validation_mask")
    parser.add_argument(
        "--scheduler",
        choices=["plateau", "none"],
        default="plateau",
    )
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument(
        "--rt-metric",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Track RT/CCS metrics instead of fragment metrics.",
    )
    parser.add_argument(
        "--index-col",
        type=int,
        default=0,
        help="CSV index column (default: 0). Use --no-index-col to disable.",
    )
    parser.add_argument(
        "--no-index-col",
        action="store_true",
        help="Disable index_col when reading CSV.",
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
    norm = re.sub(r"\b(?:NaN|nan)\b", "None", text)
    norm = re.sub(r"\b(?:Infinity|inf)\b", "1e309", norm)
    norm = re.sub(r"\b(?:-Infinity|-inf)\b", "-1e309", norm)
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


def _build_summary_from_columns(row, metadata_key_map):
    summary = {}
    for key, cols in metadata_key_map.items():
        if not isinstance(cols, (list, tuple)):
            cols = [cols]
        for col in cols:
            if col in row.index:
                value = row[col]
                if value is not None and not (
                    isinstance(value, float) and np.isnan(value)
                ):
                    summary[key] = value
                    break
    return summary


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def _load_model_params(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, "r") as fp:
        return json.load(fp)


def _choose_loss(loss_name: str):
    if loss_name == "graphwise_kl":
        return GraphwiseKLLoss(reduction="mean"), {"kl": GraphwiseKLLossMetric}
    if loss_name == "weighted_mse":
        return WeightedMSELoss(), {"mse": WeightedMSEMetric}
    if loss_name == "weighted_mae":
        return WeightedMAELoss(), {"mae": WeightedMAEMetric}
    if loss_name == "mse":
        return torch.nn.MSELoss(), None
    raise ValueError(f"Unknown loss: {loss_name}")


def main() -> None:
    args = parse_args()
    dev = _resolve_device(args.device)
    np.seterr(invalid="ignore")

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
        overwrite_sets["instrument"] = [
            x.strip() for x in args.instruments.split(",") if x.strip()
        ]
    if args.precursor_modes:
        overwrite_sets["precursor_mode"] = [
            x.strip() for x in args.precursor_modes.split(",") if x.strip()
        ]
    if not overwrite_sets:
        overwrite_sets = None

    node_encoder = AtomFeatureEncoder(
        feature_list=["symbol", "num_hydrogen", "ring_type"]
    )
    bond_encoder = BondFeatureEncoder(feature_list=["bond_type", "ring_type"])
    covariate_encoder = CovariateFeatureEncoder(
        feature_list=[
            "collision_energy",
            "molecular_weight",
            "precursor_mode",
            "instrument",
            "element_composition",
        ],
        sets_overwrite=overwrite_sets,
    )
    rt_encoder = CovariateFeatureEncoder(
        feature_list=[
            "molecular_weight",
            "precursor_mode",
            "instrument",
            "element_composition",
        ],
        sets_overwrite=overwrite_sets,
    )
    covariate_encoder.normalize_features["collision_energy"]["max"] = (
        args.ce_upper_limit
    )
    covariate_encoder.normalize_features["molecular_weight"]["max"] = (
        args.weight_upper_limit
    )
    rt_encoder.normalize_features["molecular_weight"]["max"] = args.weight_upper_limit

    metadata_key_map = {
        "name": ["Name", "NAME", "Title", "TITLE"],
        "collision_energy": ["CE", "COLLISION_ENERGY", "CollisionEnergy"],
        "instrument": ["Instrument_type", "instrument", "INSTRUMENT_TYPE"],
        "precursor_mode": ["Precursor_type", "ADDUCT", "PRECURSORTYPE"],
        "precursor_mz": ["PrecursorMZ", "PEPMASS", "PRECURSORMZ"],
        "retention_time": ["RETENTIONTIME", "RTINSECONDS", "retention_time"],
        "ccs": ["CCS", "ccs"],
    }

    # Build metabolites
    metabolites = []
    invalid_rows = []
    for idx, row in df.iterrows():
        smiles = row.get(args.smiles_col)
        if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
            invalid_rows.append(idx)
            continue
        mol = _safe_metabolite(smiles)
        if mol is None:
            invalid_rows.append(idx)
            continue
        mol.create_molecular_structure_graph()
        mol.compute_graph_attributes(node_encoder, bond_encoder)

        if args.group_id_col in df.columns:
            try:
                mol.set_id(int(row[args.group_id_col]))
            except Exception:
                pass

        summary = None
        if args.summary_col in df.columns:
            summary = row.get(args.summary_col)
        if summary is None:
            summary = _build_summary_from_columns(row, metadata_key_map)

        try:
            mol.add_metadata(summary, covariate_encoder, rt_encoder)
        except Exception:
            invalid_rows.append(idx)
            continue

        if args.loss_weight_col in df.columns:
            try:
                mol.set_loss_weight(float(row[args.loss_weight_col]))
            except Exception:
                mol.set_loss_weight(1.0)
        else:
            mol.set_loss_weight(1.0)

        metabolites.append(mol)
        df.at[idx, "Metabolite"] = mol

    if invalid_rows:
        df = df.drop(index=invalid_rows)
        print(f"Dropped {len(invalid_rows)} invalid rows.")

    # Fragmentation trees
    if args.use_frag_index:
        mindex = MetaboliteIndex()
        mindex.index_metabolites(df["Metabolite"])
        mindex.create_fragmentation_trees(depth=args.fragmentation_depth)
        mindex.add_fragmentation_trees_to_metabolite_list(
            df["Metabolite"], graph_mismatch_policy=args.graph_mismatch_policy
        )
    else:
        df["Metabolite"].apply(lambda x: x.fragment_MOL(depth=args.fragmentation_depth))

    # Match peaks to fragments
    ppm_default = args.ppm if args.ppm is not None else DEFAULT_PPM
    match_invalid = []
    for idx, row in df.iterrows():
        peaks = row.get(args.peaks_col)
        if not isinstance(peaks, dict):
            match_invalid.append(idx)
            continue
        mz = peaks.get("mz")
        intensity = peaks.get("intensity")
        if mz is None or intensity is None or len(mz) == 0:
            match_invalid.append(idx)
            continue
        tol = ppm_default
        if args.ppm_col in df.columns:
            try:
                val = float(row[args.ppm_col])
                if not np.isnan(val):
                    tol = val
            except Exception:
                pass
        try:
            row["Metabolite"].match_fragments_to_peaks(mz, intensity, tolerance=tol)
        except Exception:
            match_invalid.append(idx)

    if match_invalid:
        df = df.drop(index=match_invalid)
        print(f"Dropped {len(match_invalid)} rows with invalid peaks.")

    df["num_peak_matches"] = df["Metabolite"].apply(
        lambda x: x.match_stats["num_peak_matches"]
    )
    if args.min_peak_matches > 0:
        before = len(df)
        df = df[df["num_peak_matches"] >= args.min_peak_matches]
        print(
            f"Filtered {before - len(df)} rows with < {args.min_peak_matches} peak matches."
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
        data = row["Metabolite"].as_geometric_data().to(dev)
        if args.group_id_col in df_train.columns:
            try:
                data.group_id = int(row[args.group_id_col])
            except Exception:
                pass
        geo_data.append(data)
    print(f"Prepared training/validation with {len(geo_data)} data points")

    # Model params
    default_params = {
        "param_tag": "default",
        "gnn_type": "RGCNConv",
        "depth": 10,
        "hidden_dimension": 300,
        "residual_connections": False,
        "layer_stacking": True,
        "embedding_aggregation": "concat",
        "embedding_dimension": 300,
        "subgraph_features": True,
        "pooling_func": "max",
        "layer_norm": True,
        "dense_layers": 2,
        "dense_dim": 500,
        "input_dropout": 0.25,
        "latent_dropout": 0.25,
        "prepare_additional_layers": False,
        "rt_supported": False,
        "ccs_supported": False,
        "version": "x.x.x",
    }
    base_params = _load_model_params(args.model_params)
    model_params = dict(default_params)
    model_params.update(base_params)
    model_params.update(
        {
            "node_feature_layout": node_encoder.feature_numbers,
            "edge_feature_layout": bond_encoder.feature_numbers,
            "static_feature_dimension": geo_data[0]["static_edge_features"].shape[1],
            "static_rt_feature_dimension": geo_data[0]["static_rt_features"].shape[1],
            "output_dimension": len(DEFAULT_MODES) * 2,
            "atom_features": node_encoder.feature_list,
            "setup_features": covariate_encoder.feature_list,
            "setup_features_categorical_set": covariate_encoder.categorical_sets,
            "rt_features": rt_encoder.feature_list,
            "prepare_additional_layers": args.with_rt or args.with_ccs,
            "rt_supported": args.with_rt,
            "ccs_supported": args.with_ccs,
        }
    )

    # Initialize or resume model
    if args.resume:
        state_path = args.resume.replace(".pt", "_state.pt")
        params_path = args.resume.replace(".pt", "_params.json")
        if os.path.exists(state_path) and os.path.exists(params_path):
            model = FioraModel.load_from_state_dict(args.resume).to(dev)
        else:
            model = FioraModel.load(args.resume).to(dev)
    else:
        model = FioraModel(model_params).to(dev)

    if (args.with_rt or args.with_ccs) and not model.model_params.get(
        "prepare_additional_layers", False
    ):
        raise RuntimeError(
            "Model does not include RT/CCS heads but --with-rt/--with-ccs was set."
        )

    loss_fn, metric_dict = _choose_loss(args.loss)

    split_by_group = args.split_by_group and args.group_id_col in df_train.columns
    only_training = len(val_keys) == 0 and not args.use_validation_mask

    trainer = SpectralTrainer(
        geo_data,
        y_tag=args.y_label,
        problem_type="regression",
        train_val_split=args.train_val_split,
        split_by_group=split_by_group,
        only_training=only_training,
        train_keys=train_keys,
        val_keys=val_keys,
        metric_dict=metric_dict,
        seed=args.seed,
        device=dev,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=args.scheduler_patience,
            factor=args.scheduler_factor,
            mode="min",
        )

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    checkpoints = trainer.train(
        model,
        optimizer,
        loss_fn,
        scheduler=scheduler,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_every_n_epochs=args.val_every,
        use_validation_mask=args.use_validation_mask,
        with_RT=args.with_rt,
        with_CCS=args.with_ccs,
        rt_metric=args.rt_metric,
        mask_name=args.validation_mask_name,
        save_path=output_path,
        tag="train",
    )

    print(f"Finished training. Best checkpoint: {checkpoints['file']}")


if __name__ == "__main__":
    main()
