#!/usr/bin/env python3
"""Preprocess MSnLib spectra with full parity to lib_loader/msnlib_loader.ipynb."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split

from fiora.GNN.CovariateFeatureEncoder import CovariateFeatureEncoder
from fiora.IO import mgfReader
from fiora.IO.LibraryLoader import LibraryLoader
from fiora.MOL import constants as mol_constants
from fiora.MOL.Metabolite import Metabolite
from fiora.MOL.MetaboliteIndex import MetaboliteIndex

RDLogger.DisableLog('rdApp.*')

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE_DIR / 'library.csv'
DEFAULT_ALLOWED_PRECURSOR_MODES = ['[M+H]+', '[M-H]-', '[M]+', '[M]-']
DEFAULT_SPECTYPES = ['SINGLE_BEST_SCAN', 'SAME_ENERGY', 'SINGLE_SCAN']


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(f'[preprocess_msnlib] {msg}', flush=True)


def _iter_progress(iterable, *, total: int, desc: str, enabled: bool):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm  # Optional dependency
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _load_msnlib_dir(path: Path) -> pd.DataFrame:
    dfs = []
    for filename in sorted(path.iterdir()):
        if not filename.name.endswith('ms2.mgf'):
            continue
        df = pd.DataFrame(mgfReader.read(str(filename)))
        df['file'] = filename.name
        df['lib'] = 'MSnLib'
        parts = filename.name.split('_')
        df['origin'] = parts[1] if len(parts) > 1 else ''
        dfs.append(df)
    if not dfs:
        raise SystemExit(f'No ms2.mgf files found in {path}')
    df = pd.concat(dfs, ignore_index=True)
    df.reset_index(inplace=True)
    return df


def _compute_ce_steps(series: pd.Series, delim: str) -> pd.Series:
    def _parse(val):
        if pd.isna(val):
            return []
        text = str(val).strip()
        if '[' in text and ']' in text:
            text = text.strip('[]')
        parts = [p for p in text.split(delim) if p]
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                continue
        if vals:
            return vals
        # fallback single float
        try:
            return [float(text)]
        except ValueError:
            return []

    return series.apply(_parse)


def _reweight_groups(df: pd.DataFrame) -> pd.DataFrame:
    df['num_per_group'] = df['group_id'].map(df['group_id'].value_counts())
    df['loss_weight'] = 1.0 / df['num_per_group']
    return df


def _apply_hard_soft_filters(df: pd.DataFrame) -> pd.DataFrame:
    hard_filters = {'min_peaks': 2, 'min_coverage': 0.5, 'max_precursor_intensity': 0.9}
    soft_filters = {
        'desired_peaks': 4,
        'desired_coverage': 0.75,
        'desired_peak_percentage': 0.5,
    }
    drop_indices = []
    for i, data in df.iterrows():
        m = data['Metabolite']
        hard_pass = True
        if m.match_stats['num_peak_matches_filtered'] < hard_filters['min_peaks']:
            hard_pass = False
        if m.match_stats['coverage'] < hard_filters['min_coverage']:
            hard_pass = False
        if m.match_stats['precursor_prob'] > hard_filters['max_precursor_intensity']:
            hard_pass = False
        if not hard_pass:
            drop_indices.append(i)
            continue

        soft_pass = False
        if m.match_stats['num_peak_matches_filtered'] >= soft_filters['desired_peaks']:
            soft_pass = True
        if (
            m.match_stats['percent_peak_matches_filtered']
            >= soft_filters['desired_peak_percentage']
        ):
            soft_pass = True
        if m.match_stats['coverage'] >= soft_filters['desired_coverage']:
            soft_pass = True
        if not soft_pass:
            drop_indices.append(i)

    if drop_indices:
        df = df.drop(drop_indices)
    return df


def _assign_reference_splits(
    df: pd.DataFrame,
    reference_path: str,
    casmi16_path: str | None,
    casmi22_path: str | None,
    casmi16t_path: str | None,
    seed: int,
) -> pd.DataFrame:
    L = LibraryLoader()
    df_merged = L.load_from_csv(reference_path)
    other_dfs = {
        'train': df_merged[df_merged['dataset'] == 'training'].drop_duplicates(
            subset=['group_id']
        ),
        'val': df_merged[df_merged['dataset'] == 'validation'].drop_duplicates(
            subset=['group_id']
        ),
        'test': df_merged[df_merged['dataset'] == 'test'].drop_duplicates(
            subset=['group_id']
        ),
    }
    if casmi16_path:
        other_dfs['test'] = pd.concat(
            [other_dfs['test'], pd.read_csv(casmi16_path, index_col=[0])]
        )
    if casmi16t_path:
        other_dfs['test'] = pd.concat(
            [other_dfs['test'], pd.read_csv(casmi16t_path, index_col=[0])]
        )
    if casmi22_path:
        other_dfs['test'] = pd.concat(
            [other_dfs['test'], pd.read_csv(casmi22_path, index_col=[0])]
        )
    other_dfs['test'] = other_dfs['test'].drop_duplicates(subset=['SMILES'])

    lookup_table = {'train': set(), 'val': set(), 'test': set()}
    for key, df_x in other_dfs.items():
        df_x['Metabolite'] = df_x['SMILES'].apply(Metabolite)
        for _, data in df_x.iterrows():
            m = data['Metabolite']
            lookup_table[key].add((m.ExactMolWeight, m.morganFingerCountOnes))

    train, val, test = [], [], []
    for gid in df['group_id'].unique():
        m = df[df['group_id'] == gid].iloc[0]['Metabolite']
        fast_id = (m.ExactMolWeight, m.morganFingerCountOnes)
        found_match = False
        if fast_id in lookup_table['train']:
            for _, data in other_dfs['train'].iterrows():
                if m == data['Metabolite']:
                    train.append(gid)
                    found_match = True
                    break
        if not found_match and fast_id in lookup_table['val']:
            for _, data in other_dfs['val'].iterrows():
                if m == data['Metabolite']:
                    val.append(gid)
                    found_match = True
                    break
        if not found_match and fast_id in lookup_table['test']:
            for _, data in other_dfs['test'].iterrows():
                if m == data['Metabolite']:
                    test.append(gid)
                    break

    keys = np.unique(df['group_id'].astype(int))
    mask = ~np.isin(keys, train + val + test)
    unassigned_keys = keys[mask]
    desired_split_size = int(len(keys) * 0.1)
    test_size_remaining = desired_split_size - len(test)
    val_size_remaining = desired_split_size - len(val)

    test_new_frac = (
        test_size_remaining / len(unassigned_keys) if len(unassigned_keys) else 0
    )
    val_new_frac = (
        val_size_remaining / len(unassigned_keys) if len(unassigned_keys) else 0
    )
    if len(unassigned_keys):
        temp_keys, test_keys = train_test_split(
            unassigned_keys, test_size=test_new_frac, random_state=seed
        )
        adjusted_val_size = (
            val_new_frac / (1 - test_new_frac) if (1 - test_new_frac) else 0
        )
        train_keys, val_keys = train_test_split(
            temp_keys, test_size=adjusted_val_size, random_state=seed
        )
        train = np.concatenate((np.array(train), train_keys))
        val = np.concatenate((np.array(val), val_keys))
        test = np.concatenate((np.array(test), test_keys))

    df['dataset'] = df['group_id'].apply(
        lambda x: 'training' if x in train else 'validation' if x in val else 'test'
    )
    df['datasplit'] = df['dataset']
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Preprocess MSnLib spectra (full parity with msnlib_loader.ipynb).'
    )
    parser.add_argument(
        '--msnlib-dir',
        default=str(BASE_DIR / 'raw'),
        help='Directory with MSnLib ms2.mgf files (default: ./raw).',
    )
    parser.add_argument('--version', default='v7', help='MSnLib version (v5/v7).')
    parser.add_argument(
        '--filter-spectype',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Filter spectra by SPECTYPE (default: true).',
    )
    parser.add_argument(
        '--allowed-spectypes',
        default=','.join(DEFAULT_SPECTYPES),
        help='Comma-separated list of spectypes to keep.',
    )
    parser.add_argument('--ppm-num', type=int, default=10)
    parser.add_argument('--ce-upper-limit', type=float, default=100.0)
    parser.add_argument('--weight-upper-limit', type=float, default=1000.0)
    parser.add_argument(
        '--allowed-precursor-modes',
        default=','.join(DEFAULT_ALLOWED_PRECURSOR_MODES),
        help='Comma-separated precursor modes to keep.',
    )
    parser.add_argument(
        '--reference-splits',
        default=None,
        help='Path to reference datasplits CSV (e.g., datasplits_Jan24.csv).',
    )
    parser.add_argument('--casmi16', default=None, help='Path to CASMI-16 CSV.')
    parser.add_argument('--casmi22', default=None, help='Path to CASMI-22 CSV.')
    parser.add_argument('--casmi16t', default=None, help='Path to CASMI-16T CSV.')
    parser.add_argument(
        '--assign-datasplit',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Assign train/val/test splits if no reference provided (default: true).',
    )
    parser.add_argument('--train-frac', type=float, default=0.8)
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--test-frac', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--verbose',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Print stage-level progress messages (default: true).',
    )
    parser.add_argument(
        '--progress',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Show tqdm progress bars for heavy loops (default: true).',
    )
    parser.add_argument(
        '--output',
        default=str(DEFAULT_OUTPUT),
        help='Output CSV path (default: ./library.csv).',
    )

    args = parser.parse_args()
    msnlib_dir = Path(args.msnlib_dir)

    _log(f'Loading MSnLib files from {msnlib_dir}', args.verbose)
    df = _load_msnlib_dir(msnlib_dir)
    _log(f'Loaded {len(df)} raw spectra rows.', args.verbose)
    delim = ', ' if args.version == 'v5' else ','
    df['CE_steps'] = _compute_ce_steps(df['COLLISION_ENERGY'], delim)
    df['Num_steps'] = df['CE_steps'].apply(len)
    df['CE'] = df['CE_steps'].apply(lambda x: sum(x) / len(x) if x else np.nan)

    if args.filter_spectype:
        allowed_spectypes = [
            s.strip() for s in args.allowed_spectypes.split(',') if s.strip()
        ]
        before = len(df)
        df = df[df['SPECTYPE'].isin(allowed_spectypes)]
        _log(
            f'SPECTYPE filter kept {len(df)}/{before} rows: {allowed_spectypes}',
            args.verbose,
        )

    df['peaks'] = df['peaks'].apply(lambda p: p if isinstance(p, dict) else None)
    before = len(df)
    df = df[df['peaks'].notna()].copy()
    _log(f'Rows with valid peaks: {len(df)}/{before}', args.verbose)

    tolerance = args.ppm_num * mol_constants.PPM
    df['PPM_num'] = args.ppm_num
    df['ppm_peak_tolerance'] = tolerance

    _log('Constructing Metabolite objects...', args.verbose)
    df['Metabolite'] = [
        Metabolite(smiles)
        for smiles in _iter_progress(
            df['SMILES'], total=len(df), desc='Metabolites', enabled=args.progress
        )
    ]
    _log('Building molecular structure graphs...', args.verbose)
    for m in _iter_progress(
        df['Metabolite'], total=len(df), desc='Build graphs', enabled=args.progress
    ):
        m.create_molecular_structure_graph()
    _log('Computing graph attributes...', args.verbose)
    for m in _iter_progress(
        df['Metabolite'], total=len(df), desc='Graph attrs', enabled=args.progress
    ):
        m.compute_graph_attributes(memory_safe=False)

    mindex = MetaboliteIndex()
    _log('Indexing metabolites and creating fragmentation trees...', args.verbose)
    mindex.index_metabolites(df['Metabolite'])
    h_plus = Chem.MolFromSmiles('[H+]')
    mol_constants.ADDUCT_WEIGHTS.update(
        {
            '[M+2H]-': Descriptors.ExactMolWt(h_plus)
            + 1 * Descriptors.ExactMolWt(Chem.MolFromSmiles('[H]')),
            '[M+3H]-': Descriptors.ExactMolWt(h_plus)
            + 2 * Descriptors.ExactMolWt(Chem.MolFromSmiles('[H]')),
        }
    )
    mindex.create_fragmentation_trees()
    mindex.add_fragmentation_trees_to_metabolite_list(
        df['Metabolite'], graph_mismatch_policy='recompute'
    )

    df['group_id'] = df['Metabolite'].apply(lambda x: x.get_id())
    df = _reweight_groups(df)

    _log('Matching fragments to peaks...', args.verbose)
    for metabolite, peaks in _iter_progress(
        zip(df['Metabolite'], df['peaks']),
        total=len(df),
        desc='Match fragments',
        enabled=args.progress,
    ):
        metabolite.match_fragments_to_peaks(
            peaks['mz'],
            peaks['intensity'],
            tolerance=tolerance,
            match_stats_only=True,
        )

    df['PEPMASS'] = pd.to_numeric(df['PEPMASS'], errors='coerce')
    df['RTINSECONDS'] = pd.to_numeric(df['RTINSECONDS'], errors='coerce')
    df['ionization'] = 'ESI'
    df['instrument'] = 'HCD'
    df['Precursor_type'] = df['ADDUCT']

    metadata_key_map = {
        'name': 'NAME',
        'collision_energy': 'CE',
        'instrument': 'instrument',
        'ionization': 'ionization',
        'precursor_mz': 'PEPMASS',
        'precursor_mode': 'Precursor_type',
        'retention_time': 'RTINSECONDS',
        'ce_steps': 'CE_steps',
    }

    setup_encoder = CovariateFeatureEncoder(
        feature_list=[
            'collision_energy',
            'molecular_weight',
            'precursor_mode',
            'instrument',
        ]
    )
    rt_encoder = CovariateFeatureEncoder(
        feature_list=['molecular_weight', 'precursor_mode', 'instrument']
    )
    setup_encoder.normalize_features['collision_energy']['max'] = args.ce_upper_limit
    setup_encoder.normalize_features['molecular_weight']['max'] = (
        args.weight_upper_limit
    )
    rt_encoder.normalize_features['molecular_weight']['max'] = args.weight_upper_limit

    df['summary'] = df.apply(
        lambda x: {key: x[name] for key, name in metadata_key_map.items()}, axis=1
    )
    df.apply(
        lambda x: x['Metabolite'].add_metadata(x['summary'], setup_encoder, rt_encoder),
        axis=1,
    )

    allowed_precursors = [
        x.strip() for x in args.allowed_precursor_modes.split(',') if x.strip()
    ]
    before = len(df)
    df = df[df['ADDUCT'].isin(allowed_precursors)]
    _log(
        f'Precursor mode filter kept {len(df)}/{before} rows: {allowed_precursors}',
        args.verbose,
    )

    correct_energy = df['Metabolite'].apply(
        lambda x: (
            (x.metadata['collision_energy'] <= args.ce_upper_limit)
            and (x.metadata['collision_energy'] > 1)
        )
    )
    before = len(df)
    df = df[correct_energy]
    _log(f'Collision energy filter kept {len(df)}/{before} rows.', args.verbose)

    correct_weight = df['Metabolite'].apply(
        lambda x: x.metadata['molecular_weight'] <= args.weight_upper_limit
    )
    before = len(df)
    df = df[correct_weight]
    _log(f'Molecular weight filter kept {len(df)}/{before} rows.', args.verbose)

    before = len(df)
    df = _apply_hard_soft_filters(df)
    _log(f'Peak-match quality filters kept {len(df)}/{before} rows.', args.verbose)

    if args.reference_splits:
        _log('Assigning datasplits from reference files...', args.verbose)
        df = _assign_reference_splits(
            df,
            args.reference_splits,
            args.casmi16,
            args.casmi22,
            args.casmi16t,
            args.seed,
        )
    elif args.assign_datasplit:
        _log('Assigning random datasplits...', args.verbose)
        group_ids = df['group_id'].unique().tolist()
        rng = np.random.default_rng(args.seed)
        rng.shuffle(group_ids)
        n = len(group_ids)
        n_train = int(n * args.train_frac)
        n_val = int(n * args.val_frac)
        n_test = int(n * args.test_frac)
        if n_train + n_val + n_test > n:
            n_test = max(0, n - (n_train + n_val))
        train_ids = set(group_ids[:n_train])
        val_ids = set(group_ids[n_train : n_train + n_val])
        test_ids = set(group_ids[n_train + n_val : n_train + n_val + n_test])

        def _split_label(gid):
            if gid in train_ids:
                return 'training'
            if gid in val_ids:
                return 'validation'
            if gid in test_ids:
                return 'test'
            return 'training'

        df['datasplit'] = df['group_id'].apply(_split_label)

    if 'datasplit' in df.columns:
        counts = df['datasplit'].value_counts().to_dict()
        _log(f'Split counts: {counts}', args.verbose)

    df = _reweight_groups(df)

    if 'Metabolite' in df.columns:
        df = df.drop(columns=['Metabolite'])

    for col in ['peaks', 'summary']:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: json.dumps(v) if isinstance(v, dict) else v
            )

    _log(f'Writing output to {args.output}', args.verbose)
    df.to_csv(args.output, index=False)
    print(f'Wrote {len(df)} rows to {args.output}')


if __name__ == '__main__':
    main()
