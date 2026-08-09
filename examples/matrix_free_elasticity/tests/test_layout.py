"""Guards for the artifact-naming contract shared by validate/sync.

``validate.py`` writes the raw artifacts and ``sync_results.py`` reads them
back. Before these tests existed, a rename on one side only surfaced at
runtime as a missing-artifact error.
"""

from __future__ import annotations

import utils.contract as contract
import utils.layout as layout


def validation_summaries(dimension: int) -> set:
    return {
        layout.validation_artifact_paths(dimension, name)[0]
        for name, _, _, _ in layout.validation_case_specs(
            contract.REFINEMENTS[dimension]
        )
    }


def test_serial_case_names_are_stable():
    """阶段 1a 的默认范围只有单 rank 算例。"""

    assert {
        layout.case_name(role, operator_level, ranks)
        for role, operator_level, ranks in layout.validation_cases()
    } == {
        "ea-coarse-1rank",
        "ea-medium-1rank",
        "ea-fine-1rank",
        "fa-coarse-1rank",
    }


def test_parallel_case_names_are_stable():
    """阶段 1b 在 1a 之上恰好追加一个 2-rank 算例。"""

    assert {
        layout.case_name(role, operator_level, ranks)
        for role, operator_level, ranks in layout.validation_cases(
            include_parallel=True,
        )
    } == {
        "ea-coarse-1rank",
        "ea-medium-1rank",
        "ea-fine-1rank",
        "ea-fine-2rank",
        "fa-coarse-1rank",
    }


def test_serial_scope_uses_a_single_rank_only():
    assert all(
        ranks == 1
        for _, _, ranks in layout.validation_cases()
    )


def test_every_validation_case_has_a_refinement():
    for dimension in contract.SUPPORTED_DIMENSIONS:
        for include_parallel in (False, True):
            specs = layout.validation_case_specs(
                contract.REFINEMENTS[dimension],
                include_parallel=include_parallel,
            )
            assert len(specs) == len(
                layout.validation_cases(
                    include_parallel=include_parallel,
                )
            )
            assert all(
                refinement > 0 for _, refinement, _, _ in specs
            )
            assert len({name for name, _, _, _ in specs}) == len(specs)


def test_ea_evidence_sources_are_written_by_validate():
    for dimension in contract.SUPPORTED_DIMENSIONS:
        written = validation_summaries(dimension)
        sources = layout.ea_evidence_sources(dimension)
        assert len(sources) == len(layout.EA_EVIDENCE_ROLES)
        for _, source in sources:
            assert source in written


def test_fa_evidence_source_is_written_by_validate():
    for dimension in contract.SUPPORTED_DIMENSIONS:
        assert (
            layout.fa_evidence_source(dimension)
            in validation_summaries(dimension)
        )


def test_validation_artifacts_share_one_stem_per_case():
    summary, solution, vtu = layout.validation_artifact_paths(
        2,
        layout.case_name("coarse", "ea", 1),
    )
    assert summary.stem == solution.stem == vtu.stem
    assert summary.parent == layout.dimension_output_dir(2)
    assert (summary.suffix, solution.suffix, vtu.suffix) == (
        ".json",
        ".npy",
        ".vtu",
    )


def test_run_artifact_path_encodes_the_dimension():
    path = layout.run_artifact_path(
        "json",
        dimension=2,
        operator_level="ea",
        degree=1,
        resolution=(8, 8),
        mpi_size=1,
    )
    assert path.name == "elasticity-2d-ea-p1-8x8-1ranks.json"
    assert path.parent == layout.OUTPUT_DIR


def test_results_analysis_contains_every_generated_block():
    results = layout.RESULTS_PATH.read_text(encoding="utf-8")
    for dimension in contract.SUPPORTED_DIMENSIONS:
        begin, end = layout.results_markers(dimension)
        assert results.count(begin) == 1
        assert results.count(end) == 1
        assert results.index(begin) < results.index(end)


def test_evidence_and_validation_paths_do_not_collide():
    for dimension in contract.SUPPORTED_DIMENSIONS:
        evidence = layout.evidence_path(dimension)
        assert evidence.parent == layout.EVIDENCE_DIR
        assert evidence not in validation_summaries(dimension)
    assert layout.validation_evidence_path("all").parent == (
        layout.OUTPUT_DIR
    )


def test_tolerances_are_ordered_and_positive():
    assert contract.MATVEC_RELATIVE_TOL < (
        contract.EXPLICIT_SOLUTION_RELATIVE_TOL
    )
    assert contract.PARALLEL_L2_DIFFERENCE_TOL < (
        contract.PARALLEL_SOLUTION_RELATIVE_TOL
    )
    assert contract.NORM_FLOOR > 0.0
    assert contract.residual_limit(0.0) == contract.DEFAULT_ATOL
    assert contract.residual_limit(1.0e6) > contract.DEFAULT_ATOL
