#! /usr/bin/env python
import argparse
import ast
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger

from fiora.GNN.AtomFeatureEncoder import AtomFeatureEncoder
from fiora.GNN.BondFeatureEncoder import BondFeatureEncoder
from fiora.GNN.CovariateFeatureEncoder import CovariateFeatureEncoder
from fiora.GNN.FioraModel import FioraModel
from fiora.IO.LibraryLoader import LibraryLoader
from fiora.MOL.Metabolite import Metabolite
from fiora.MOL.MetaboliteIndex import MetaboliteIndex
from fiora.MS.SimulationFramework import SimulationFramework

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=SyntaxWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fiora-eval",
        description="Evaluate a trained FIORA model on validation/test splits.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to preprocessed CSV containing spectra/metadata/SMILES.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Path to checkpoint .pt produced by fiora-train.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on (e.g. cpu, cuda:0). Default: auto.",
    )
    parser.add_argument(
        "--datasplit-col",
        default="datasplit",
        help="Column containing split labels (default: datasplit).",
    )
    parser.add_argument(
        "--splits",
        default="validation,test",
        help="Comma-separated splits to evaluate (default: validation,test).",
    )
    parser.add_argument(
        "--score",
        default="spectral_sqrt_cosine",
        help="Score column to summarize after evaluation.",
    )
    parser.add_argument(
        "--y-label",
        default="compiled_probsALL",
        help="Prediction target label used during training.",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.001,
        help="Minimum predicted peak intensity to keep.",
    )
    parser.add_argument(
        "--fragmentation-depth",
        type=int,
        default=1,
        help="Fragmentation depth for metabolite trees.",
    )
    parser.add_argument(
        "--graph-mismatch-policy",
        choices=["recompute", "ignore"],
        default="recompute",
    )
    parser.add_argument("--summary-col", default="summary")
    parser.add_argument("--peaks-col", default="peaks")
    parser.add_argument("--smiles-col", default="SMILES")
    parser.add_argument("--group-id-col", default="group_id")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to write evaluated split CSV files.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show tqdm progress bars (default: true).",
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


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def _parse_dict(val):
    if isinstance(val, dict):
        return val
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    text = str(val).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
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


def _build_summary_from_columns(row):
    metadata_key_map = {
        "name": ["Name", "NAME", "Title", "TITLE"],
        "collision_energy": ["CE", "COLLISION_ENERGY", "CollisionEnergy"],
        "instrument": ["Instrument_type", "instrument", "INSTRUMENT_TYPE"],
        "precursor_mode": ["Precursor_type", "ADDUCT", "PRECURSORTYPE"],
        "precursor_mz": ["PrecursorMZ", "PEPMASS", "PRECURSORMZ"],
        "retention_time": ["RETENTIONTIME", "RTINSECONDS", "retention_time"],
        "ccs": ["CCS", "ccs"],
    }
    summary = {}
    for key, cols in metadata_key_map.items():
        for col in cols:
            if col in row.index:
                value = row[col]
                if value is not None and not (
                    isinstance(value, float) and np.isnan(value)
                ):
                    summary[key] = value
                    break
    return summary


def _prepare_metabolites(
    df: pd.DataFrame, model, progress: bool = True
) -> tuple[pd.DataFrame, int]:
    setup_features = model.model_params.get(
        "setup_features",
        [
            "collision_energy",
            "molecular_weight",
            "precursor_mode",
            "instrument",
            "element_composition",
        ],
    )
    rt_features = model.model_params.get(
        "rt_features",
        ["molecular_weight", "precursor_mode", "instrument", "element_composition"],
    )
    setup_sets = model.model_params.get("setup_features_categorical_set")

    node_encoder = AtomFeatureEncoder(
        feature_list=["symbol", "num_hydrogen", "ring_type"]
    )
    bond_encoder = BondFeatureEncoder(feature_list=["bond_type", "ring_type"])
    setup_encoder = CovariateFeatureEncoder(
        feature_list=setup_features, sets_overwrite=setup_sets
    )
    rt_encoder = CovariateFeatureEncoder(
        feature_list=rt_features, sets_overwrite=setup_sets
    )

    invalid_rows = []
    iterator = df.iterrows()
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=len(df), desc="Prepare metabolites")
        except Exception:
            pass

    for idx, row in iterator:
        smiles = row.get("SMILES")
        if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
            invalid_rows.append(idx)
            continue
        mol = _safe_metabolite(smiles)
        if mol is None:
            invalid_rows.append(idx)
            continue

        mol.create_molecular_structure_graph()
        mol.compute_graph_attributes(node_encoder, bond_encoder)
        if "group_id" in df.columns:
            try:
                mol.set_id(int(row["group_id"]))
            except Exception:
                pass

        summary = row.get("summary")
        if summary is None:
            summary = _build_summary_from_columns(row)

        try:
            mol.add_metadata(summary, setup_encoder, rt_encoder)
        except Exception:
            invalid_rows.append(idx)
            continue
        df.at[idx, "Metabolite"] = mol

    if invalid_rows:
        df = df.drop(index=invalid_rows).copy()
    return df, len(invalid_rows)


def _load_model(path: str, dev: str):
    state_path = path.replace(".pt", "_state.pt")
    params_path = path.replace(".pt", "_params.json")
    if os.path.exists(state_path) and os.path.exists(params_path):
        return FioraModel.load_from_state_dict(path).to(dev)
    return FioraModel.load(path).to(dev)


def _to_csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Metabolite" in out.columns:
        out = out.drop(columns=["Metabolite"])
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].apply(
                lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v
            )
    return out


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
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    if not splits:
        raise SystemExit("No valid --splits provided.")
    if args.datasplit_col not in df.columns:
        raise SystemExit(f"datasplit column '{args.datasplit_col}' not found in input.")

    df = df[df[args.datasplit_col].isin(splits)].copy()
    print(f"Loaded {len(df)} rows for splits: {splits}")
    if len(df) == 0:
        raise SystemExit("No rows left after split filtering.")

    model = _load_model(args.model, dev)
    model.eval()

    # Standardize user-configurable column names for downstream code.
    if args.summary_col != "summary" and args.summary_col in df.columns:
        df["summary"] = df[args.summary_col]
    if args.peaks_col != "peaks" and args.peaks_col in df.columns:
        df["peaks"] = df[args.peaks_col]
    if args.smiles_col != "SMILES" and args.smiles_col in df.columns:
        df["SMILES"] = df[args.smiles_col]
    if args.group_id_col != "group_id" and args.group_id_col in df.columns:
        df["group_id"] = df[args.group_id_col]

    df, dropped = _prepare_metabolites(df, model, progress=args.progress)
    if dropped:
        print(f"Dropped {dropped} invalid rows during metabolite preparation.")

    mindex = MetaboliteIndex()
    mindex.index_metabolites(df["Metabolite"])
    mindex.create_fragmentation_trees(depth=args.fragmentation_depth)
    mindex.add_fragmentation_trees_to_metabolite_list(
        df["Metabolite"], graph_mismatch_policy=args.graph_mismatch_policy
    )

    fiora = SimulationFramework(None, dev=dev)
    use_groundtruth = "peaks" in df.columns
    if not use_groundtruth:
        print(
            "Warning: peaks column not found. Running prediction without score metrics."
        )

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        part = df[df[args.datasplit_col] == split].copy()
        if part.empty:
            print(f"Split '{split}': 0 rows (skipping).")
            continue
        part = fiora.simulate_all(
            part,
            model,
            base_attr_name=args.y_label,
            groundtruth=use_groundtruth,
            min_intensity=args.min_prob,
            progress=args.progress,
            progress_desc=f"{split} split",
        )

        if args.score in part.columns:
            score_vals = pd.to_numeric(part[args.score], errors="coerce")
            print(
                f"Split '{split}': n={len(part)} | "
                f"{args.score}_mean={score_vals.mean():.5f} | "
                f"{args.score}_median={score_vals.median():.5f}"
            )
        else:
            print(f"Split '{split}': n={len(part)} | score '{args.score}' not found.")

        if output_dir is not None:
            out_path = output_dir / f"{split}_eval.csv"
            _to_csv_safe(part).to_csv(out_path, index=False)
            print(f"Wrote {len(part)} rows to {out_path}")


if __name__ == "__main__":
    main()
