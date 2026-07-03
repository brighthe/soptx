"""T4b: TrainedPredictor 管道健全性 (不依赖长训练, torch 缺失自动跳过).

验证 (与 deck ④ 无关的软件正确性, 训练收敛由 benchmark 报告):
- vech <-> K_s 上三角重建互逆, 且重建 K_s 对称;
- TrainedPredictor.predict 返回 (n_b, n_b) 对称 K_s, 且组装进
  InterfaceCondensedSystem 得对称 K^cond;
- save/load 后预测逐比特一致 (归一化统计 + 权重完整往返).
"""

import numpy as np
import pytest
from fealpy.backend import backend_manager as bm

pytest.importorskip("torch")

from soptx.analysis.multiscale import (  # noqa: E402
    CoarseFineMeshPair,
    InterfaceCondensedSystem,
    KsMLP,
    Standardizer,
    SubstructureTemplate,
    TrainedPredictor,
    load_predictor,
    save_predictor,
)
from soptx.analysis.multiscale.trained_predictor import (  # noqa: E402
    ks_to_vech,
    vech_indices,
    vech_to_ks,
)
from soptx.interpolation.linear_elastic_material import (  # noqa: E402
    IsotropicLinearElasticMaterial,
)


def _make_template(L=2):
    bm.set_backend("numpy")
    material = IsotropicLinearElasticMaterial(
        youngs_modulus=1.0, poisson_ratio=0.3, plane_type="plane_stress",
    )
    mesh_pair = CoarseFineMeshPair(domain=(0.0, 2.0, 0.0, 1.0),
                                   ncx=2, ncy=2, L=L)
    template = SubstructureTemplate(mesh_pair, material, q=4)
    return mesh_pair, template


def _random_predictor(template, seed=0):
    """随机初始化 (未训练) 预测器, 仅测管道, 不测精度."""
    import torch

    torch.manual_seed(seed)
    m = template.local_c2d.shape[0]
    iu = vech_indices(template.n_b)
    n_out = iu[0].size
    model = KsMLP(m, n_out, hidden=16, depth=2)
    stats = Standardizer(
        x_mean=np.full(m, 0.6), x_std=np.full(m, 0.2),
        y_mean=np.zeros(n_out), y_std=np.ones(n_out),
    )
    return TrainedPredictor(template, model, stats, device="cpu")


def test_vech_roundtrip_and_symmetry():
    _, template = _make_template()
    iu = vech_indices(template.n_b)
    K = template.condense(np.full(template.local_c2d.shape[0], 0.7)).K_s
    vech = ks_to_vech(K, iu)
    K_back = vech_to_ks(vech, template.n_b, iu)
    assert np.allclose(K_back, K_back.T)
    # 真值 K_s 本就对称, 上三角重建应精确还原
    assert np.allclose(K_back, K)


def test_predict_returns_symmetric_ks_and_assembles():
    mesh_pair, template = _make_template()
    pred = _random_predictor(template)
    rho = np.full(mesh_pair.fine_mesh.number_of_cells(), 0.6)

    K_s = pred.predict_K_s(rho[mesh_pair.sub_cells(0)])
    assert K_s.shape == (template.n_b, template.n_b)
    assert np.allclose(K_s, K_s.T)

    op = pred.predict(rho[mesh_pair.sub_cells(0)])
    assert op.X is None  # T4b 不学 X / N̂ (阶段三)

    system = InterfaceCondensedSystem(mesh_pair, pred, rho)
    A = system.K_cond.toarray()
    assert np.allclose(A, A.T, atol=1e-10)


def test_save_load_roundtrip(tmp_path):
    mesh_pair, template = _make_template()
    pred = _random_predictor(template)
    rho_local = np.full(template.local_c2d.shape[0], 0.55)
    before = pred.predict_K_s(rho_local)

    result = {
        "model": pred.model, "stats": pred.stats,
        "in_dim": template.local_c2d.shape[0],
        "out_dim": vech_indices(template.n_b)[0].size,
        "hidden": pred.model.hidden, "depth": pred.model.depth,
        "n_b": template.n_b,
    }
    path = tmp_path / "pred.pt"
    save_predictor(path, result)
    reloaded = load_predictor(path, template, device="cpu")
    after = reloaded.predict_K_s(rho_local)
    assert np.allclose(before, after, atol=0, rtol=0)
