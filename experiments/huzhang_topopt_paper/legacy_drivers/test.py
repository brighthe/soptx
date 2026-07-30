import matplotlib.font_manager
from matplotlib.font_manager import FontProperties

def list_useful_fonts():
    print("=" * 60)
    print("正在扫描系统中的中文字体和常用英文字体...")
    print("=" * 60)
    
    # 获取所有系统字体
    fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    
    chinese_fonts = []
    serif_fonts = []
    
    for font_path in fonts:
        try:
            prop = matplotlib.font_manager.FontProperties(fname=font_path)
            name = prop.get_name()
            
            # 简单的关键词过滤
            lower_name = name.lower()
            
            # 检查中文常见关键词
            if any(x in lower_name for x in ['hei', 'song', 'kai', 'ming', 'noto sans cjk', 'wenquanyi', 'droid sans fallback']):
                chinese_fonts.append((name, font_path))
            
            # 检查 Serif/Times
            if 'times' in lower_name or 'liberation serif' in lower_name:
                serif_fonts.append((name, font_path))
                
        except:
            continue

    print("【发现的中文字体候选】：")
    if not chinese_fonts:
        print("  (未找到显式的中文字体，请务必手动上传 SimHei.ttf)")
    for name, path in sorted(chinese_fonts):
        print(f"  名称: {name:<25} | 路径: {path}")

    print("\n【发现的 Times/Serif 字体候选】：")
    if not serif_fonts:
        print("  (未找到 Times New Roman，请务必手动上传 times.ttf)")
    for name, path in sorted(serif_fonts):
        print(f"  名称: {name:<25} | 路径: {path}")
        
    print("=" * 60)

"""
有限差分梯度验证
================
验证 AugmentedLagrangianObjective.jac() 的正确性。

用法:
    在优化循环的第一次 jac() 调用前插入如下代码:
        from test_fd_gradient import fd_gradient_check
        fd_gradient_check(al_obj, rho_array, n_test=20, delta=1e-4)

    rho_array: 当前密度数组 (NC,) 或 Function，需与 fun/jac 接口一致
    al_obj:    AugmentedLagrangianObjective 实例
"""

import numpy as np
from fealpy.backend import backend_manager as bm


def fd_gradient_check(
    al_obj,
    density,
    n_test: int = 20,
    delta: float = 1e-4,
    seed: int = 42,
):
    """
    对随机选取的 n_test 个单元做中心差分验证。

    Parameters
    ----------
    al_obj  : AugmentedLagrangianObjective 实例
    density : 当前密度数组，shape (NC,)，numpy array 或 backend tensor
    n_test  : 抽查单元数量
    delta   : 扰动步长
    seed    : 随机种子，保证可复现
    """
    # --- 转为 numpy 操作，避免 backend 差异 ---
    try:
        rho_np = np.array(density, dtype=np.float64).ravel()
    except Exception:
        rho_np = density.numpy().ravel()

    NC = rho_np.shape[0]

    # --- 1. 计算基准点解析梯度 ---
    # 先清除缓存，做一次完整的 fun + jac
    al_obj._cache_g = None
    al_obj._cache_h = None

    state_base = {}
    rho_tensor = bm.tensor(rho_np, dtype=bm.float64)

    J_base = float(al_obj.fun(rho_tensor, state_base))
    dJ_analytical = al_obj.jac(rho_tensor, state_base)

    try:
        dJ_np = np.array(dJ_analytical, dtype=np.float64).ravel()
    except Exception:
        dJ_np = dJ_analytical.numpy().ravel()

    print(f"\n{'='*60}")
    print(f"有限差分梯度验证  (delta={delta:.0e}, n_test={n_test})")
    print(f"基准 J = {J_base:.8f}")
    print(f"解析梯度: mean={dJ_np.mean():.4e}, "
          f"max={dJ_np.max():.4e}, min={dJ_np.min():.4e}")
    print(f"{'='*60}")
    print(f"{'单元e':>8} {'解析梯度':>14} {'FD梯度':>14} {'相对误差':>12} {'绝对误差':>12}")
    print(f"{'-'*62}")

    # --- 2. 随机选取测试单元 ---
    rng = np.random.default_rng(seed)
    # 优先选高应力区（梯度绝对值较大处更容易暴露符号错误）
    abs_grad = np.abs(dJ_np)
    prob = abs_grad / (abs_grad.sum() + 1e-30)
    test_indices = rng.choice(NC, size=min(n_test, NC), replace=False, p=prob)

    errors_rel = []

    for e in test_indices:
        # --- 前向扰动 ---
        rho_plus = rho_np.copy()
        rho_plus[e] = np.clip(rho_np[e] + delta, 1e-3, 1.0)

        al_obj._cache_g = None
        al_obj._cache_h = None
        state_plus = {}
        J_plus = float(al_obj.fun(bm.tensor(rho_plus, dtype=bm.float64), state_plus))

        # --- 后向扰动 ---
        rho_minus = rho_np.copy()
        rho_minus[e] = np.clip(rho_np[e] - delta, 1e-3, 1.0)

        al_obj._cache_g = None
        al_obj._cache_h = None
        state_minus = {}
        J_minus = float(al_obj.fun(bm.tensor(rho_minus, dtype=bm.float64), state_minus))

        # --- 实际步长（因 clip 可能不对称）---
        actual_delta = rho_plus[e] - rho_minus[e]

        dJ_fd = (J_plus - J_minus) / actual_delta
        dJ_an = dJ_np[e]

        abs_err = abs(dJ_fd - dJ_an)
        rel_err = abs_err / (abs(dJ_an) + 1e-30)
        errors_rel.append(rel_err)

        # 标记异常单元
        flag = " !!!" if rel_err > 1e-2 else ""
        print(f"{e:>8d} {dJ_an:>14.6e} {dJ_fd:>14.6e} {rel_err:>12.4e} {abs_err:>12.4e}{flag}")

    print(f"{'-'*62}")
    print(f"相对误差: mean={np.mean(errors_rel):.4e}, "
          f"max={np.max(errors_rel):.4e}, "
          f"通过率(< 1e-3): {np.mean(np.array(errors_rel) < 1e-3)*100:.1f}%")
    print(f"{'='*60}\n")

    # --- 还原缓存状态（让后续优化正常进行）---
    al_obj._cache_g = None
    al_obj._cache_h = None

    return np.array(errors_rel)

if __name__ == "__main__":
    list_useful_fonts()