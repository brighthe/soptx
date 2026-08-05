"""Smoke test for the PINN elasticity minimal demo."""

from __future__ import annotations

from minimal_demo import run_minimal_demo


def test_minimal_demo_2d_smoke() -> None:
    """Run a quick 2-epoch 2D PINN minimal demo smoke test."""
    run_minimal_demo(dim=2, epochs=2, lr=1e-3)


def test_minimal_demo_3d_smoke() -> None:
    """Run a quick 2-epoch 3D PINN minimal demo smoke test."""
    run_minimal_demo(dim=3, epochs=2, lr=1e-3)
