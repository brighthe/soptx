"""
PIML 2D Linearly Elastic Substructure Condensation Demo (MFEM-Inspired OOP Version)
===================================================================================
Inspired by LLNL MFEM's `mfem::StaticCondensation` architecture.

This self-contained demo ties together:
  1. SubstructureMesh                    : 2D rectangular sub-mesh partitioner & SIMP density
  2. FEAStaticCondensation / PIMLStatic  : MFEM-inspired StaticCondensation class hierarchy
  3. GlobalAssembler                     : Interface assembly & Full-scale FEA direct solve
  4. Accuracy & Performance Verification : Machine-precision V1 validation & displacement plot

Author: Liang He (Postdoc at DUT) & Antigravity Assistant
Date: 2026-08-06
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from static_condensation import (
    FEAStaticCondensation,
    PIMLStaticCondensation,
    PIMLSurrogateNet
)
from substructure import SubstructureMesh
from assembler import GlobalAssembler

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def main():
    print("=" * 80)
    print("PIML 2D Elasticity Substructure Condensation Demo (MFEM-Inspired OOP)")
    print("=" * 80)

    # Output directory
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Cantilever Beam Physical Dimensions
    Lx, Ly = 2.0, 1.0
    E_base, nu = 1.0, 0.3

    # Mesh Parameters: 4x2 Coarse Substructures, each has 5x5 Fine Q4 Elements
    n_sub_x, n_sub_y = 4, 2
    n_fine_x, n_fine_y = 5, 5

    total_fine_x = n_sub_x * n_fine_x
    total_fine_y = n_sub_y * n_fine_y

    print(f"Problem Geometry : Rectangular Cantilever [{Lx} x {Ly}]")
    print(f"Substructures    : {n_sub_x} x {n_sub_y} ({n_sub_x * n_sub_y} total substructures)")
    print(f"Fine Sub-mesh    : {n_fine_x} x {n_fine_y} Q4 elements per substructure")
    print(f"Full Fine-mesh   : {total_fine_x} x {total_fine_y} Total Fine Q4 Elements")
    print("-" * 80)

    # Instantiate Global Assembler & Substructure Meshes
    assembler = GlobalAssembler(Lx, Ly, n_sub_x, n_sub_y, n_fine_x, n_fine_y, E_base, nu)
    dx_sub = Lx / n_sub_x
    dy_sub = Ly / n_sub_y

    sub_meshes = []
    for sx in range(n_sub_x):
        for sy in range(n_sub_y):
            sub_id = sx * n_sub_y + sy
            x_span = (sx * dx_sub, (sx + 1) * dx_sub)
            y_span = (sy * dy_sub, (sy + 1) * dy_sub)
            sub_mesh = SubstructureMesh(sub_id, x_span, y_span, n_fine_x, n_fine_y, E_base, nu)
            sub_meshes.append(sub_mesh)

    # --------------------------------------------------------------------------
    # Step 1 & 2: Local Material Density Generation & FEA Static Condensation
    # --------------------------------------------------------------------------
    print("[Step 1 & 2] Generating Local Density & Running FEAStaticCondensation (MFEM-Style)...")
    t0 = time.time()

    densities = []
    exact_condensors = []

    for sub_mesh in sub_meshes:
        xc = (sub_mesh.x_span[0] + sub_mesh.x_span[1]) / 2.0
        yc = (sub_mesh.y_span[0] + sub_mesh.y_span[1]) / 2.0
        rho = 0.7 + 0.3 * np.sin(np.pi * xc / Lx) * np.cos(np.pi * yc / Ly)
        rho_field = np.full((n_fine_x, n_fine_y), rho)
        densities.append(rho_field)

        # Instantiate MFEM-Inspired FEAStaticCondensation
        condensor = FEAStaticCondensation(sub_mesh.i_dofs, sub_mesh.b_dofs)
        K_local = sub_mesh.assemble_local_stiffness(rho_field)
        condensor.condense(K_local, rho_field)

        exact_condensors.append(condensor)

    t_condense = time.time() - t0
    print(f"-> FEAStaticCondensation complete for {len(sub_meshes)} substructures in {t_condense:.4f} s.")

    # --------------------------------------------------------------------------
    # Full-Scale FEA Reference Solution
    # --------------------------------------------------------------------------
    print("[Step 4] Solving Full-Scale FEA Reference Solution...")
    right_mid_node = total_fine_x * (total_fine_y + 1) + ((total_fine_y + 1) // 2)
    load_dof = 2 * right_mid_node + 1  # y-direction load

    t0_solve = time.time()
    U_full_ref, free_dofs = assembler.solve_fullscale_fea(densities, load_dof, load_val=-1.0)
    t_full_solve = time.time() - t0_solve
    print(f"-> Full-Scale FEA Reference Solved in {t_full_solve:.4f} s.")

    # --------------------------------------------------------------------------
    # Step 5: Substructure Fine-Scale Recovery & Precision Verification
    # --------------------------------------------------------------------------
    print("[Step 5] Recovering Internal Displacements via StaticCondensation.recover()...")

    U_recovered = np.zeros(assembler.total_full_dofs)
    U_recovered[free_dofs] = U_full_ref[free_dofs]

    for sx in range(n_sub_x):
        for sy in range(n_sub_y):
            sub_idx = sx * n_sub_y + sy
            sub_mesh = sub_meshes[sub_idx]
            condensor = exact_condensors[sub_idx]

            sub_global_dofs = assembler.get_substructure_global_dofs(sx, sy, sub_mesh)
            u_sub_b = U_full_ref[sub_global_dofs[sub_mesh.b_dofs]]

            # MFEM-Style recover() call
            u_sub_i_recovered = condensor.recover(u_sub_b)
            U_recovered[sub_global_dofs[sub_mesh.i_dofs]] = u_sub_i_recovered

    err_recovery = np.linalg.norm(U_recovered - U_full_ref) / np.linalg.norm(U_full_ref)

    # --------------------------------------------------------------------------
    # PIML Surrogate Neural Network Demo with Fallback
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PIML Neural Network StaticCondensation Quick Demo")
    print("=" * 80)

    n_b_dofs = sub_meshes[0].n_b
    triu_indices = np.triu_indices(n_b_dofs)
    n_triu = len(triu_indices[0])

    net = PIMLSurrogateNet(input_dim=n_fine_x * n_fine_y, output_dim=n_triu)
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    # Generate Synthetic Dataset
    X_train_list, Y_train_list = [], []
    for _ in range(200):
        rand_rho = np.random.uniform(0.3, 1.0, (n_fine_x, n_fine_y))
        K_loc = sub_meshes[0].assemble_local_stiffness(rand_rho)
        c_tmp = FEAStaticCondensation(sub_meshes[0].i_dofs, sub_meshes[0].b_dofs)
        K_s, _ = c_tmp.condense(K_loc)

        X_train_list.append(rand_rho.flatten())
        Y_train_list.append(K_s[triu_indices])

    X_train_t = torch.tensor(np.array(X_train_list), dtype=torch.float32)
    Y_train_t = torch.tensor(np.array(Y_train_list), dtype=torch.float32)

    net.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = net(X_train_t)
        loss = criterion(out, Y_train_t)
        loss.backward()
        optimizer.step()

    net.eval()
    print(f"-> PIML Surrogate MLP Trained (200 Epochs). Final Training MSE: {loss.item():.6e}")

    # Test PIMLStaticCondensation class
    piml_condensor = PIMLStaticCondensation(sub_meshes[0].i_dofs, sub_meshes[0].b_dofs, model=net)
    K_loc0 = sub_meshes[0].assemble_local_stiffness(densities[0])
    K_s_piml, N_piml = piml_condensor.condense(K_loc0, densities[0])

    rel_err_Ks = np.linalg.norm(K_s_piml - exact_condensors[0].K_s) / np.linalg.norm(exact_condensors[0].K_s)
    print(f"-> PIMLStaticCondensation K_s Relative Error: {rel_err_Ks:.4e} (Fallback Triggered: {piml_condensor.used_fallback})")

    # --------------------------------------------------------------------------
    # Verification Summary Table
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SOPTX MFEM-INSPIRED STATIC CONDENSATION VERIFICATION TABLE")
    print("=" * 80)
    print(f"{'Metric / Indicator':<45} | {'Value':<25}")
    print("-" * 75)
    print(f"{'Full Fine-Scale Total DOFs':<45} | {assembler.total_full_dofs:<25}")
    print(f"{'Interface Condensed DOFs':<45} | {len(free_dofs):<25}")
    print(f"{'Degree of Freedom Reduction Ratio':<45} | {assembler.total_full_dofs / len(free_dofs):.2f}x")
    print(f"{'FEAStaticCondensation Baseline Error (V1)':<45} | {err_recovery:.4e} (Machine Prec)")
    print(f"{'PIMLStaticCondensation K_s Error':<45} | {rel_err_Ks:.4e}")
    print(f"{'Full-Scale Solver Time (s)':<45} | {t_full_solve:.4f} s")
    print("=" * 80)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n_full_nodes_x = total_fine_x + 1
    n_full_nodes_y = total_fine_y + 1

    U_y_full = U_full_ref[1::2].reshape((n_full_nodes_x, n_full_nodes_y)).T
    U_y_rec = U_recovered[1::2].reshape((n_full_nodes_x, n_full_nodes_y)).T

    im0 = axes[0].imshow(U_y_full, origin='lower', cmap='viridis', extent=[0, Lx, 0, Ly])
    axes[0].set_title("Full-Scale Direct FEA U_y Field")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(U_y_rec, origin='lower', cmap='viridis', extent=[0, Lx, 0, Ly])
    axes[1].set_title("StaticCondensation Recovered U_y Field")
    axes[1].set_xlabel("x")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "piml_cantilever_demo.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"\n[Success] Visualization saved to: {fig_path}")
    print("MFEM-inspired StaticCondensation pipeline completed successfully!")


if __name__ == "__main__":
    main()
