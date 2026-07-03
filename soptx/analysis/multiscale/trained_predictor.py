"""T4b: TrainedPredictor —— 极小 MLP 预测缩聚刚度 K_s (回填 deck 帧 7 ④).

范围界定 (与 progress-frame7_piml.md「阶段二/三」诚实边界一致):

- 极小 MLP 只预测 **K_s** (组装 K^cond 与计算证据 ④ 误差 ‖K̂_s − K_s‖/‖K_s‖
  都只依赖 K_s), 不学 X / N̂；结构保持参数化 (对称正定 / 能量一致) 与
  训练版前向恢复属阶段三, 故 TrainedPredictor.predict 返回的算子 X=None,
  仅供 InterfaceCondensedSystem 组装 K^cond, 不参与训练版前向求解.
- 预测器只消费 rho_local (逐细单元 coef, 值域约 [0.3, 0.9]), 与
  Exact / Mock 共享同一 predict 接口 (问题无关性: 所有子结构同一模板).

输出参数化: K_s 对称, 网络预测其上三角 vech(K_s) (含对角), 重建时对称化,
保证组装出的 K^cond 对称. 训练与推断均在标准化空间 (输入按训练集逐特征
标准化, 输出 vech 逐分量标准化) 进行, denormalize 后返回物理量.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .multiscale_shape import SubstructureOperator, SubstructureTemplate
from .piml_predictor import MultiscalePredictor


# ----------------------------------------------------------------------
# 网络与归一化统计
# ----------------------------------------------------------------------

class KsMLP(nn.Module):
    """极小 MLP: rho_local (m,) -> vech(K_s) (n_out,), 标准化空间.

    depth 个隐层 (Linear + SiLU) + 线性输出头; 默认 2 层宽 128,
    参数量对 L=5 (m=25, n_out=820) 约 1.2e5, 属"极小网络"量级.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 hidden: int = 128, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Standardizer:
    """逐分量标准化统计 (x-> (x-mean)/std), 以 numpy 数组保存."""

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    @staticmethod
    def fit(X: np.ndarray, Y: np.ndarray, eps: float = 1e-8) -> "Standardizer":
        return Standardizer(
            x_mean=X.mean(0), x_std=np.maximum(X.std(0), eps),
            y_mean=Y.mean(0), y_std=np.maximum(Y.std(0), eps),
        )

    def to_torch(self, device: str = "cpu") -> dict:
        return {k: torch.as_tensor(getattr(self, k), dtype=torch.float32,
                                   device=device)
                for k in ("x_mean", "x_std", "y_mean", "y_std")}


# ----------------------------------------------------------------------
# 训练标注数据 (随机局部密度 + 精确静力缩聚)
# ----------------------------------------------------------------------

def vech_indices(n_b: int) -> tuple[np.ndarray, np.ndarray]:
    """K_s 上三角 (含对角) 索引, 与重建互逆."""
    return np.triu_indices(n_b)


def ks_to_vech(K_s: np.ndarray, iu: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return K_s[iu]


def vech_to_ks(vech: np.ndarray, n_b: int,
               iu: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """由上三角向量重建对称 K_s."""
    K = np.zeros((n_b, n_b), dtype=np.float64)
    K[iu] = vech
    K = K + K.T - np.diag(np.diag(K))
    return K


def make_training_data(template: SubstructureTemplate, n: int,
                       lo: float = 0.3, hi: float = 0.9,
                       seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """采样 n 个 rho_local ~ U[lo, hi]^m, 用精确缩聚标注 vech(K_s).

    返回 (X (n, m), Y (n, n_out))；标注即 ExactPredictor 的真值来源.
    """
    rng = np.random.default_rng(seed)
    m = template.local_c2d.shape[0]
    n_b = template.n_b
    iu = vech_indices(n_b)
    n_out = iu[0].size

    X = rng.uniform(lo, hi, size=(n, m)).astype(np.float64)
    Y = np.empty((n, n_out), dtype=np.float64)
    for i in range(n):
        Y[i] = ks_to_vech(template.condense(X[i]).K_s, iu)
    return X, Y


# ----------------------------------------------------------------------
# 训练与评估
# ----------------------------------------------------------------------

def _rel_frob_from_vech(pred: np.ndarray, true: np.ndarray, n_b: int,
                        iu: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """逐样本 ‖K̂_s − K_s‖_F / ‖K_s‖_F (对称重建后, 与 deck ④ 同口径)."""
    w = np.ones(iu[0].size)
    w[iu[0] != iu[1]] = 2.0  # 上三角误差按对称计双份 (等价整阵 Frobenius)
    diff = pred - true
    num = np.sqrt((w * diff ** 2).sum(1))
    den = np.sqrt((w * true ** 2).sum(1))
    return num / den


def train_ks_mlp(template: SubstructureTemplate, *,
                 n_train: int = 20000, n_val: int = 2000,
                 lo: float = 0.3, hi: float = 0.9,
                 hidden: int = 128, depth: int = 2,
                 epochs: int = 400, batch_size: int = 512,
                 lr: float = 1e-3, weight_decay: float = 0.0,
                 seed: int = 0, device: str = "cpu",
                 verbose: bool = True) -> dict:
    """训练极小 MLP, 返回 {model, stats, history, val_rel_frob, ...}.

    损失 = 标准化 vech(K_s) 的 MSE; 报告的验收指标为验证集上逐样本
    ‖K̂_s − K_s‖_F / ‖K_s‖_F 的均值/中位数/最大 (deck ④ 同口径).
    """
    torch.manual_seed(seed)
    n_b = template.n_b
    iu = vech_indices(n_b)

    Xtr, Ytr = make_training_data(template, n_train, lo, hi, seed=seed)
    Xva, Yva = make_training_data(template, n_val, lo, hi, seed=seed + 1)

    std = Standardizer.fit(Xtr, Ytr)
    t = std.to_torch(device)

    def norm_x(a):
        return (torch.as_tensor(a, dtype=torch.float32, device=device)
                - t["x_mean"]) / t["x_std"]

    def norm_y(a):
        return (torch.as_tensor(a, dtype=torch.float32, device=device)
                - t["y_mean"]) / t["y_std"]

    Xtr_t, Ytr_t = norm_x(Xtr), norm_y(Ytr)
    Xva_t = norm_x(Xva)

    model = KsMLP(template.local_c2d.shape[0], iu[0].size,
                  hidden=hidden, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    n = Xtr_t.shape[0]
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = loss_fn(out, Ytr_t[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item() * idx.numel()
        sched.step()
        ep_loss /= n
        history.append(ep_loss)
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            vr = _eval_val(model, Xva_t, Yva, std, n_b, iu, device)
            print(f"  epoch {ep:4d} | mse(norm)={ep_loss:.3e} | "
                  f"val relF mean={vr['mean']:.3e} max={vr['max']:.3e}",
                  flush=True)

    val = _eval_val(model, Xva_t, Yva, std, n_b, iu, device)
    return {
        "model": model, "stats": std, "history": history,
        "val_rel_frob_mean": val["mean"], "val_rel_frob_median": val["median"],
        "val_rel_frob_max": val["max"], "n_b": n_b,
        "in_dim": template.local_c2d.shape[0], "out_dim": iu[0].size,
        "hidden": hidden, "depth": depth,
    }


@torch.no_grad()
def _eval_val(model, Xva_t, Yva, std, n_b, iu, device) -> dict:
    model.eval()
    pred_n = model(Xva_t).cpu().numpy()
    pred = pred_n * std.y_std + std.y_mean
    rel = _rel_frob_from_vech(pred, Yva, n_b, iu)
    return {"mean": float(rel.mean()), "median": float(np.median(rel)),
            "max": float(rel.max())}


# ----------------------------------------------------------------------
# 预测器 (推断) + 存取
# ----------------------------------------------------------------------

class TrainedPredictor(MultiscalePredictor):
    """极小 MLP 推断缩聚刚度 K_s, 与 Exact/Mock 同一 predict 接口.

    predict(rho_local) 返回 SubstructureOperator(X=None, K_s=K̂_s):
    仅供 InterfaceCondensedSystem 组装 K^cond / 度量 ④ 误差,
    训练版前向恢复 (需 X̂ / 结构保持 N̂) 属阶段三, 本类不提供.
    """

    def __init__(self, template: SubstructureTemplate, model: KsMLP,
                 stats: Standardizer, device: str = "cpu"):
        super().__init__(template)
        self.n_b = template.n_b
        self._iu = vech_indices(self.n_b)
        self.device = device
        self.model = model.to(device).eval()
        self.stats = stats
        self._t = stats.to_torch(device)

    @torch.no_grad()
    def _predict_vech(self, rho: np.ndarray) -> np.ndarray:
        """rho: (m,) 或 (N, m) -> vech(K_s): (n_out,) 或 (N, n_out)."""
        a = np.atleast_2d(np.asarray(rho, dtype=np.float64))
        x = (torch.as_tensor(a, dtype=torch.float32, device=self.device)
             - self._t["x_mean"]) / self._t["x_std"]
        out = self.model(x) * self._t["y_std"] + self._t["y_mean"]
        vech = out.cpu().numpy()
        return vech[0] if np.ndim(rho) == 1 else vech

    def predict_K_s(self, rho_local: np.ndarray) -> np.ndarray:
        return vech_to_ks(self._predict_vech(rho_local), self.n_b, self._iu)

    def predict(self, rho_local: np.ndarray) -> SubstructureOperator:
        return SubstructureOperator(
            X=None,  # 训练版不学 X / N̂ (阶段三); 仅供 K^cond 组装
            K_s=self.predict_K_s(rho_local),
            lu_ii=None,
        )


def save_predictor(path, result: dict) -> None:
    """存 train_ks_mlp 返回结果 (权重 + 归一化统计 + 结构超参)."""
    std: Standardizer = result["stats"]
    torch.save({
        "state_dict": result["model"].state_dict(),
        "in_dim": result["in_dim"], "out_dim": result["out_dim"],
        "hidden": result["hidden"], "depth": result["depth"],
        "n_b": result["n_b"],
        "x_mean": std.x_mean, "x_std": std.x_std,
        "y_mean": std.y_mean, "y_std": std.y_std,
        "val_rel_frob_mean": result.get("val_rel_frob_mean"),
        "val_rel_frob_max": result.get("val_rel_frob_max"),
    }, path)


def load_predictor(path, template: SubstructureTemplate,
                   device: str = "cpu") -> TrainedPredictor:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = KsMLP(ckpt["in_dim"], ckpt["out_dim"],
                  hidden=ckpt["hidden"], depth=ckpt["depth"])
    model.load_state_dict(ckpt["state_dict"])
    std = Standardizer(x_mean=ckpt["x_mean"], x_std=ckpt["x_std"],
                       y_mean=ckpt["y_mean"], y_std=ckpt["y_std"])
    return TrainedPredictor(template, model, std, device=device)
