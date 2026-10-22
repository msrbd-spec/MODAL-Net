"""Test T5: Selection guard enforces no test predictions in selection."""

import pytest
from src.evaluation.tables import selection_guard, phase_select


def test_selection_guard_enforcement():
    """selection_guard must raise PermissionError on any test prediction path."""
    # Forbidden paths
    forbidden = [
        "predictions_test.npz",
        "results/phase2a_recipe/p2a_plain/predictions_test.npz",
        "results/phase2a_recipe/p2a_plain/predictions_external_test.npz",
        "results/phase1_zoo/test_metrics.csv",
    ]
    for path in forbidden:
        with pytest.raises(PermissionError, match="SELECTION GUARD VIOLATION"):
            selection_guard(path)

    # Allowed paths (OOF and external DEV)
    allowed = [
        "oof.npz",
        "results/phase2a_recipe/p2a_plain/oof.npz",
        "predictions_external_dev.npz",
        "results/phase2a_recipe/p2a_plain/predictions_external_dev.npz",
        "results/phase0_protocol/splits_summary.csv",
    ]
    for path in allowed:
        # Should not raise
        selection_guard(path)
