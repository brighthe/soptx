import contract
import layout


def test_validation_and_evidence_paths_are_dimension_specific():
    assert layout.validation_summary_path(2) != layout.validation_summary_path(3)
    assert layout.evidence_path(2) != layout.evidence_path(3)
    assert layout.validation_summary_path(2).name.endswith("-2.json")
    assert layout.evidence_path(3).name.endswith("-3d.json")


def test_readme_markers_are_stable():
    readme = layout.README_PATH.read_text(encoding="utf-8")
    for dimension in contract.SUPPORTED_DIMENSIONS:
        begin, end = layout.readme_markers(dimension)
        assert begin.startswith("<!-- BEGIN GENERATED:")
        assert end.startswith("<!-- END GENERATED:")
        assert layout.EVIDENCE_SCOPE in begin
        assert begin in readme
        assert end in readme


def test_contract_and_layout_are_importable_without_numerical_packages():
    assert contract.SCHEMA_VERSION == 3
    assert layout.REPOSITORY_ROOT.name == "soptx"


def test_tolerances_are_positive_and_2d_accuracy_is_stricter():
    assert contract.EXACT_STRAIN_SYMMETRY_MAX_ABS > 0.0
    assert contract.EXACT_GRADIENT_MAX_ABS > 0.0
    assert contract.EXACT_EQUILIBRIUM_MAX_ABS > 0.0
    assert contract.RELATIVE_DISPLACEMENT_L2_MAX_2D < (
        contract.RELATIVE_DISPLACEMENT_L2_MAX_3D
    )
    assert "best_validation_loss_max" in contract.validation_thresholds(2)
    assert "best_validation_loss_max" not in contract.validation_thresholds(3)
