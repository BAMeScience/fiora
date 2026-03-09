import json
from pathlib import Path

import pytest
import torch
from rdkit import RDLogger

from fiora.GNN.AtomFeatureEncoder import AtomFeatureEncoder
from fiora.GNN.BondFeatureEncoder import BondFeatureEncoder
from fiora.GNN.CovariateFeatureEncoder import CovariateFeatureEncoder
from fiora.MOL.Metabolite import Metabolite
from fiora.MOL.MetaboliteIndex import MetaboliteIndex
from fiora.MOL.collision_energy import NCE_to_eV, align_CE, nce_instruments


SAMPLE_SIZE = 100
SPECTRA_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "data" / "mol_core_100_spectra.jsonl"
)

RDLogger.DisableLog("rdApp.*")


@pytest.fixture(scope="module")
def sample_spectra():
    if not SPECTRA_FIXTURE_PATH.exists():
        pytest.skip(f"Missing test fixture: {SPECTRA_FIXTURE_PATH}")

    records = []
    with SPECTRA_FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if len(records) != SAMPLE_SIZE:
        pytest.skip(
            f"Expected {SAMPLE_SIZE} records in fixture, found {len(records)}"
        )
    return records


@pytest.fixture(scope="module")
def prepared_metabolites(sample_spectra):
    node_encoder = AtomFeatureEncoder(
        feature_list=["symbol", "num_hydrogen", "ring_type"]
    )
    bond_encoder = BondFeatureEncoder(feature_list=["bond_type", "ring_type"])
    setup_encoder = CovariateFeatureEncoder(
        feature_list=[
            "collision_energy",
            "molecular_weight",
            "precursor_mode",
            "instrument",
            "element_composition",
        ]
    )
    rt_encoder = CovariateFeatureEncoder(
        feature_list=[
            "molecular_weight",
            "precursor_mode",
            "instrument",
            "element_composition",
        ]
    )

    prepared = []
    for row in sample_spectra:
        metabolite = Metabolite(row["SMILES"])
        metabolite.create_molecular_structure_graph()
        metabolite.compute_graph_attributes(
            node_encoder=node_encoder, bond_encoder=bond_encoder
        )
        metabolite.add_metadata(dict(row["summary"]), setup_encoder, rt_encoder)
        prepared.append({"metabolite": metabolite, "peaks": row["peaks"]})
    return prepared


@pytest.fixture(scope="module")
def indexed_metabolites(prepared_metabolites):
    metabolites = [entry["metabolite"] for entry in prepared_metabolites]
    index = MetaboliteIndex()
    index.index_metabolites(metabolites)
    index.create_fragmentation_trees(depth=1)
    mismatches = index.add_fragmentation_trees_to_metabolite_list(metabolites)
    return {"index": index, "metabolites": metabolites, "mismatches": mismatches}


def test_load_100_spectra_fixture(sample_spectra):
    assert len(sample_spectra) == SAMPLE_SIZE
    first = sample_spectra[0]
    assert {"SMILES", "peaks", "summary", "group_id"}.issubset(first.keys())
    assert len(first["peaks"]["mz"]) == len(first["peaks"]["intensity"])
    assert len(first["peaks"]["mz"]) > 0
    assert "collision_energy" in first["summary"]
    assert "precursor_mode" in first["summary"]


def test_metabolite_graph_building(prepared_metabolites):
    assert len(prepared_metabolites) == SAMPLE_SIZE

    for entry in prepared_metabolites:
        metabolite = entry["metabolite"]
        assert metabolite.edges.shape[1] == 2
        assert len(metabolite.edges_as_tuples) == metabolite.edges.shape[0]
        assert metabolite.node_features.shape[0] == metabolite.Graph.number_of_nodes()
        assert metabolite.bond_features.shape[0] == len(metabolite.edges_as_tuples)
        assert metabolite.setup_features.shape[0] == 1
        assert metabolite.setup_features_per_edge.shape[0] == len(
            metabolite.edges_as_tuples
        )


def test_metabolite_index_and_fragmentation_trees(indexed_metabolites):
    index = indexed_metabolites["index"]
    metabolites = indexed_metabolites["metabolites"]
    mismatches = indexed_metabolites["mismatches"]

    assert len(mismatches) == 0
    assert 0 < index.get_number_of_metabolites() <= SAMPLE_SIZE

    for metabolite in metabolites:
        assert metabolite.fragmentation_tree is not None
        assert metabolite.subgraph_elem_comp.shape[0] == metabolite.edges.shape[0]
        assert metabolite.subgraph_idx_left.shape == metabolite.subgraph_idx_right.shape


def test_peak_matching_and_geometric_export(prepared_metabolites, indexed_metabolites):
    _ = indexed_metabolites  # Ensure fragmentation trees are attached.
    matched = 0

    for entry in prepared_metabolites:
        metabolite = entry["metabolite"]
        peaks = entry["peaks"]
        mz_list = peaks["mz"]
        int_list = peaks["intensity"]

        metabolite.match_fragments_to_peaks(mz_list, int_list)
        geom = metabolite.as_geometric_data()

        assert metabolite.match_stats["num_peaks"] == len(mz_list)
        assert (
            geom.compiled_probsALL.shape[0] == metabolite.edge_count_matrix.numel() + 2
        )
        assert (
            geom.compiled_validation_maskALL.shape[0] == geom.compiled_probsALL.shape[0]
        )
        assert torch.isfinite(geom.compiled_probsSQRT).all()
        matched += 1

    assert matched == SAMPLE_SIZE


def test_edge_count_cols_helper():
    mode_map = {"[M-H]-": 0, "[M+H]+": 1}
    mode_count = len(mode_map)

    left_forward, left_backward = Metabolite._edge_count_cols(
        mode_map, mode_count, "[M+H]+", "left"
    )
    right_forward, right_backward = Metabolite._edge_count_cols(
        mode_map, mode_count, "[M+H]+", "right"
    )

    assert (left_forward, left_backward) == (1, 3)
    assert (right_forward, right_backward) == (3, 1)


def test_collision_energy_helpers():
    assert NCE_to_eV(20.0, 250.0) == pytest.approx(10.0)
    assert align_CE("35eV", 200.0) == pytest.approx(35.0)
    assert align_CE("2keV", 200.0) == pytest.approx(2000.0)
    assert align_CE(20.0, 250.0, instrument=nce_instruments[0]) == pytest.approx(10.0)
    assert align_CE("15% (nominal)", 300.0) == pytest.approx(NCE_to_eV(15.0, 300.0))


def test_edge_count_matrix_accumulates_repeated_matches():
    class _StubFragment:
        def __init__(self, edge, break_side):
            self.edges = [edge]
            self.break_sides = [break_side]

        def num_of_edges(self):
            return 1

    class _StubTree:
        def __init__(self, peak_matches):
            self.peak_matches = peak_matches

        def match_peak_list(self, mz_list, int_list, tolerance=None):
            return self.peak_matches

    mode_map = {"[M+H]+": 0}
    edge = (0, 1)

    peak_matches = {
        100.0: {
            "intensity": 10.0,
            "relative_intensity": 10.0 / 15.0,
            "fragments": [_StubFragment(edge=edge, break_side="left")],
            "ion_modes": [("[M+H]+", 100.0)],
        },
        101.0: {
            "intensity": 5.0,
            "relative_intensity": 5.0 / 15.0,
            "fragments": [_StubFragment(edge=edge, break_side="left")],
            "ion_modes": [("[M+H]+", 101.0)],
        },
    }

    metabolite = Metabolite("CC")
    metabolite.create_molecular_structure_graph()
    metabolite.compute_graph_attributes()
    metabolite.fragmentation_tree = _StubTree(peak_matches)

    metabolite.match_fragments_to_peaks(
        mz_fragments=[100.0, 101.0],
        int_list=[10.0, 5.0],
        mode_map_override=mode_map,
    )

    forward_col, backward_col = Metabolite._edge_count_cols(
        mode_map, len(mode_map), "[M+H]+", "left"
    )
    forward_idx = (
        ((torch.tensor(edge) == metabolite.edges).sum(dim=1) == 2).nonzero().squeeze()
    )
    backward_idx = (
        ((torch.tensor(edge[::-1]) == metabolite.edges).sum(dim=1) == 2)
        .nonzero()
        .squeeze()
    )

    assert metabolite.edge_count_matrix[
        forward_idx, forward_col
    ].item() == pytest.approx(15.0)
    assert metabolite.edge_count_matrix[
        backward_idx, backward_col
    ].item() == pytest.approx(15.0)
