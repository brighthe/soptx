"""三种稀疏直接法后端的正确性验证入口.

被验证的对象是 ``soptx/solvers/direct.py``: 它把 FEALPy 稀疏算子交给 SuperLU /
MUMPS / MKL PARDISO 求解, 本文件检查这条交接是否正确、是否守规矩. 有限元装配
只是为了造出一个真实的弹性刚度阵, 不是被验证的对象.

三个 case, 一个 case 一个后端, 各自自证:

    scipy    SuperLU
    mumps    MUMPS
    pardiso  MKL PARDISO

四项检查 (checks):

residual   相对残差 ||b - A x|| / ||b|| 落在阈值内. 这是直接法特有的核心断言,
           而且必须对**完整矩阵** A 算: 声明 sym=1/2 时后端只吃下三角 (PARDISO
           吃上三角), 把非对称矩阵按对称声明进去会静默解另一个方程组, 那时后端
           自报的内部残差仍然很小, 只有对完整 A 算残差才抓得住.
solution   制造解的 L2 观测收敛阶不低于门禁. 残差只说明"这个线性系统解对了",
           说明不了"装配与边界条件对了"; 收敛阶补的正是后一半. 这一项复用
           ``examples/lagrange_elasticity/manufactured_convergence_demo.py``
           的加密序列与判定, 不另写一份.
ownership  求解后调用方持有的矩阵数值与索引均未被改写. 分两档输入: FEALPy 稀疏
           算子 (主路径) 与原生 scipy CSR (substructure 接口的路径, 也正是
           ``_scipy_solve`` 里那句 ``tocsr(copy=True)`` 唯一起作用的地方).
guard      非法 solver 名与非法 sym 值都抛 ValueError.

其中 ownership 与 guard 检的都是"什么都没发生", 通过时不打印, 只在失败时把整
组结果连同证据强弱打出来; 它们照样参与判定, 终端安静不等于没跑.

哪个 case 做哪几项见 ``CASE_REGISTRY``. pardiso 只有 residual 与 ownership: 它
尚未接入 ``soptx.solvers.spsolve``, 凡是要走那个入口才成立的检查 (solution 与
guard) 对它都不存在, 接入后应当补齐.

对称性一轴 (``--symmetry``) 是**声明**而不是检测: general/spd/indefinite 分别
对应 MUMPS 的 sym=0/1/2 与 PARDISO 的 mtype=11/2/-2. 本脚本装配的是 SPD 系统,
三档声明都合法, 因此三档都应当给出同一个正确解.

使用方法::

    python examples/linear_solvers/verify_direct_solvers.py --list
    python examples/linear_solvers/verify_direct_solvers.py --case scipy
    python examples/linear_solvers/verify_direct_solvers.py --case mumps --symmetry spd
    python examples/linear_solvers/verify_direct_solvers.py --case scipy mumps --json

迭代法一侧见同目录的 ``verify_cg_solver.py``; 它借用本文件的问题构造
(``build_analyzer`` / ``build_system``), 两边在同一个矩阵上说话.

判定阈值属于本示例, 不属于 ``soptx``; 见本文件的 ``TOLERANCES``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.mesh import (
    HexahedronMesh,
    QuadrangleMesh,
    TetrahedronMesh,
    TriangleMesh,
)

from soptx.fem.analyzers import build_serial_analyzer
from soptx.fem.verification import relative_difference
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems.elasticity import (
    DivergenceFreePolynomialElasticity3D,
    ExponentialSineManufacturedElasticity2D,
    SinusoidalElasticity2D,
)
from soptx.solvers import spsolve


# 三个 case 的注册表: id 即 --case 的取值, 其余字段是裸跑该 case 的缺省组合.
# 与 experiments/huzhang_topopt_paper/run.py 的 --list 同构 (拿到 id 就知道裸跑
# 会跑出什么); 只有三条, 不值得像那边一样引入 cases.toml, 直接写成常量表.
#
# symmetry: 裸跑时对称性一轴会取遍的档位, 是"以什么对称性声明去调用后端", 不是
#           对矩阵做对称性检测 -- 后端不校验, 声明错了会静默解另一个方程组.
# checks:   该 case 会执行的断言项, 词表见 CHECK_DESCRIPTIONS.
CASE_REGISTRY = (
    {
        "id": "scipy",
        "backend": "scipy",
        "library": "SuperLU",
        # spsolve 只把 sym 传给 _mumps_solve, scipy 分支没有这一轴.
        "symmetry": (),
        "checks": ("residual", "solution", "ownership", "guard"),
    },
    {
        "id": "mumps",
        "backend": "mumps",
        "library": "MUMPS",
        "symmetry": ("general", "spd", "indefinite"),
        "checks": ("residual", "solution", "ownership", "guard"),
    },
    {
        "id": "pardiso",
        "backend": "pypardiso",
        "library": "MKL PARDISO",
        # 未接入 soptx.solvers.spsolve: 收敛阶脚本调不到它, direct.py 的参数校验
        # 也管不着它, 故 solution 与 guard 两项对它不成立.
        "symmetry": ("general", "spd", "indefinite"),
        "checks": ("residual", "ownership"),
    },
)

CASES_BY_ID = {case["id"]: case for case in CASE_REGISTRY}

# --system 一轴决定用哪个分析器装配被解的矩阵: spd 走位移元 (lfem), saddle 走
# 胡张混合元的对称不定鞍点系统; saddle 尚未接线, 先只登记 spd 一档.
# 装配级别一律是 FA: 直接法要拿到显式稀疏矩阵才能分解, EA 无矩阵可分解, 因此
# 不作为一根可拨动的轴显示.
SYSTEM_ANALYZERS = {"spd": "lfem"}
DEFAULT_SYSTEM = "spd"

# checks 词表: 断言项 -> 一句话说明.
CHECK_DESCRIPTIONS = {
    "residual": "相对残差 ||b - A x|| / ||b|| 在阈值内 (对完整矩阵算)",
    "solution": "制造解的 L2 观测收敛阶不低于门禁",
    "ownership": "求解后调用方持有的矩阵数值与索引均未被改写",
    "guard": "非法 solver 名与非法 sym 值均抛 ValueError",
}

# 后端中立的对称性档位 -> 各后端的原生标志. 两边取的三角相反: MUMPS 的 sym!=0
# 只吃下三角, MKL PARDISO 的 mtype!=11 只吃上三角. 这类差异正是后端中立的
# Symmetry 枚举将来要盖住的东西.
SYMMETRY_FLAGS = {"general": 0, "spd": 1, "indefinite": 2}
PARDISO_MATRIX_TYPES = {"general": 11, "spd": 2, "indefinite": -2}
SYMMETRY_LABELS = {
    "general": "一般非对称",
    "spd": "对称正定",
    "indefinite": "一般对称",
}
# 没有对称性一轴的 case (scipy) 用它占位: 只跑一遍, 不给后端传任何对称性声明.
NOMINAL_SYMMETRY = "general"

# 判定阈值. 直接法是后向稳定的, 相对残差应落在条件数乘机器精度的量级.
# residual_relative 与 manufactured_convergence_demo.RESIDUAL_TOLERANCE 同口径,
# 两处判残差用同一把尺子.
TOLERANCES = {
    "residual_relative": 1.0e-10,
    "matrix_perturbation_relative": 0.0,
}


# --------------------------------------------------------------------------
# 问题构造 (verify_cg_solver.py 按路径加载本文件后借用)
# --------------------------------------------------------------------------
# 模型与网格的可选项. 与 examples/matrix_free_elasticity/verify_ea_correctness.py
# 保持同一套字典表约定: 维数决定问题、网格与材料假设, 其余参数一律向问题对象要.
#
# 对求解器而言, 换模型换网格改变的是**稀疏模式与条件数**而非物理: 维数决定直接法
# 的填充增长 (2D 约 O(n^1.5), 3D 约 O(n^2)), 单元类型与阶数决定每行非零数与稠密
# 块大小, 模型只影响条件数从而影响 CG 迭代数. 因此这几条轴都要能独立拨动.
PROBLEM_FACTORIES = {
    2: {
        "sinusoidal": SinusoidalElasticity2D,
        "exponential": ExponentialSineManufacturedElasticity2D,
    },
    3: {"polynomial": DivergenceFreePolynomialElasticity3D},
}
MESH_FACTORIES = {
    2: {"tri": TriangleMesh, "quad": QuadrangleMesh},
    3: {"tet": TetrahedronMesh, "hex": HexahedronMesh},
}
MATERIAL_HYPOTHESES = {2: "plane_strain", 3: "3D"}
DEFAULT_MODELS = {2: "sinusoidal", 3: "polynomial"}
DEFAULT_MESH_TYPES = {2: "tri", 3: "tet"}
MESH_DIMENSIONS = {
    mesh_type: dimension
    for dimension, factories in MESH_FACTORIES.items()
    for mesh_type in factories
}


def build_problem(dimension: int, model: str):
    """按空间维度与模型名称构造受支持的制造解问题."""
    factory = PROBLEM_FACTORIES[dimension].get(model)
    if factory is None:
        supported = ", ".join(sorted(PROBLEM_FACTORIES[dimension]))
        raise ValueError(f"{dimension}D 不支持模型 {model!r}; 可选模型: {supported}.")
    return factory()


def build_mesh(dimension: int, mesh_type: str, domain, resolution: int):
    """按空间维度与网格类型构造各向同性的均匀网格."""
    factory = MESH_FACTORIES[dimension].get(mesh_type)
    if factory is None:
        supported = ", ".join(sorted(MESH_FACTORIES[dimension]))
        raise ValueError(f"{dimension}D 不支持网格 {mesh_type!r}; 可选网格: {supported}.")
    counts = dict(zip(("nx", "ny", "nz"), (resolution,) * dimension))
    return factory.from_box(list(domain), **counts)


def build_analyzer(
    dimension: int,
    resolution: int,
    degree: int,
    model: str,
    mesh_type: str,
    operator_level: str = "fa",
):
    """构造串行拉格朗日位移元分析器, 不装配.

    ``operator_level`` 取 ``"fa"`` 得到可分解的全局稀疏矩阵, 取 ``"ea"`` 得到
    只支持 ``@`` 的 matrix-free 算子; 两者对应同一个离散算子. 返回分析器本身,
    供 CG 一侧取 Dirichlet 基准向量 (``prescribed_solution``) 与算子对角
    (``assemble_operator_diagonal``).

    Returns
    -------
    (analyzer, mesh, space, problem)
    """
    problem = build_problem(dimension, model)
    mesh = build_mesh(dimension, mesh_type, problem.domain, resolution)
    scalar_space = LagrangeFESpace(mesh, p=degree, ctype="C")
    space = TensorFunctionSpace(scalar_space, shape=(-1, dimension))
    material = IsotropicLinearElasticMaterial(
        hypothesis=MATERIAL_HYPOTHESES[dimension],
        lame_lambda=problem.lam,
        shear_modulus=problem.mu,
        device=bm.get_device(mesh),
    )
    analyzer = build_serial_analyzer(space, problem, material, degree, operator_level)
    return analyzer, mesh, space, problem


def build_system(
    dimension: int,
    resolution: int,
    degree: int,
    model: str,
    mesh_type: str,
    operator_level: str = "fa",
):
    """装配一个带 Dirichlet 边界的线弹性系统.

    返回施加过本质边界条件的全局算子 (对称正定) 与右端项. ``"fa"`` 下算子是
    稀疏矩阵, ``"ea"`` 下是 ``DirichletBCOperator``; 后者的右端项在 Dirichlet 自由
    度上取边界值, 迭代法的初值必须携带同样的分量.

    Returns
    -------
    (operator, load, mesh, space, problem)
    """
    analyzer, mesh, space, problem = build_analyzer(
        dimension, resolution, degree, model, mesh_type, operator_level
    )
    matrix = analyzer.assemble_stiff_matrix()
    operator, load = analyzer.apply_bc(matrix, analyzer.assemble_body_force_vector())
    return operator, load, mesh, space, problem


# --------------------------------------------------------------------------
# 矩阵探针
# --------------------------------------------------------------------------
def as_scipy(operator):
    """算子转 scipy 稀疏矩阵; 已是 scipy 时原样返回 (可能是共享视图)."""
    return operator.to_scipy() if hasattr(operator, "to_scipy") else operator


def describe_operator(operator) -> dict[str, Any]:
    """报告算子的实际存储格式.

    ownership 一项的可解释性依赖这个: 若算子导出为 COO, 则 ``to_scipy().tocsr()``
    本身就产生新数组, 后端无论如何都碰不到调用方的内存, 此时"未被改写"并不构成
    后端不改写输入的证据. 因此这里把格式显式打出来, 并在报告里标注证据强弱.
    """
    scipy_matrix = as_scipy(operator)
    return {
        "operator_type": type(operator).__name__,
        "scipy_format": getattr(scipy_matrix, "format", "unknown"),
        "shape": tuple(int(value) for value in scipy_matrix.shape),
        "nnz": int(scipy_matrix.nnz),
    }


def matrix_fingerprint(operator) -> tuple[np.ndarray, np.ndarray]:
    """算子非零值与列索引的副本, 用于求解前后比对.

    索引也要比: PARDISO 的 ``_check_A`` 会在索引未排序时原地 ``sort_indices()``,
    那是一次真实的调用方矩阵改写, 只比数值会漏掉.
    """
    csr = as_scipy(operator).tocsr()
    return np.array(csr.data, copy=True), np.array(csr.indices, copy=True)


# --------------------------------------------------------------------------
# 三个后端的统一调用面
# --------------------------------------------------------------------------
def backend_available(name: str) -> tuple[bool, str]:
    """后端在当前环境是否可用, 及不可用的原因."""
    if name == "scipy":
        return True, ""
    if name == "mumps":
        try:
            import mumps  # noqa: F401
        except ImportError:
            return False, "未安装 PyMUMPS"
        return True, ""
    if name == "pypardiso":
        try:
            import pypardiso  # noqa: F401
        except ImportError:
            return False, "未安装 pypardiso 或找不到 mkl_rt"
        return True, ""
    raise ValueError(f"未知后端: {name}")


def resolve_cases(case_ids: list[str]) -> list[dict[str, Any]] | None:
    """把 --case 给的 id 解析成注册表条目; 任一后端不可用即判本次验证失败.

    --case 是显式必选的: 用户写了哪个 case, 就是要求验证哪个后端. 环境里装不上
    就是这次验证没做成, 不能静默跳过后再以退出码 0 报"全部通过". 想只测某一个
    后端, 写 ``--case scipy`` 即可, 不必依赖跳过.

    Returns
    -------
    list[dict] or None
        全部可用时返回注册表条目 (顺序照 --case); 有不可用者返回 None.
    """
    cases: list[dict[str, Any]] = []
    blocked: list[str] = []
    for case_id in case_ids:
        case = CASES_BY_ID[case_id]
        usable, reason = backend_available(case["backend"])
        if usable:
            if case not in cases:
                cases.append(case)
        else:
            blocked.append(f"  case {case_id} 不可用: {reason}")

    if blocked:
        for line in blocked:
            print(line)
        return None
    return cases


def resolve_symmetries(case: dict[str, Any], requested: list[str] | None) -> list[str]:
    """定出该 case 这一次要跑哪几档对称性声明.

    没有对称性一轴的 case (scipy) 恒返回单档占位, 显式给了 ``--symmetry`` 也不
    报错 -- 同时选中多个 case 时, 该选项只应作用在有这根轴的 case 上.
    """
    axis = case["symmetry"]
    if not axis:
        return [NOMINAL_SYMMETRY]
    if requested is None:
        return list(axis)
    return [name for name in requested if name in axis]


def _pardiso_solve(operator, load, symmetry: str):
    """本地 PARDISO 分支; pypardiso 接入 soptx.solvers 后本函数应删除.

    每次都新建 ``PyPardisoSolver``: 它会把上一次的因子缓存在 ``factorized_A``
    上并在同一个矩阵上静默复用, 不新建会把"完整求解"测成"仅回代".

    mtype!=11 时 MKL PARDISO 只接受上三角部分, 与 MUMPS 的 sym!=0 只接受下三角
    正好相反 -- 这类后端差异正是后端中立的 Symmetry 枚举要盖住的.
    """
    from pypardiso import PyPardisoSolver
    from scipy.sparse import triu

    matrix = as_scipy(operator).tocsr()
    matrix_type = PARDISO_MATRIX_TYPES[symmetry]
    if matrix_type != 11:
        matrix = triu(matrix, format="csr")

    right_hand_side = bm.to_numpy(load)
    solver = PyPardisoSolver(mtype=matrix_type)
    start = time.perf_counter()
    solution = solver.solve(matrix, right_hand_side)
    elapsed = time.perf_counter() - start
    solver.free_memory(everything=True)
    return np.asarray(solution).ravel(), elapsed


def solve_once(operator, load, backend: str, symmetry: str = NOMINAL_SYMMETRY):
    """用指定后端按指定对称性声明求解一次, 返回 (解, 墙钟秒数).

    Parameters
    ----------
    symmetry : str
        对称性**声明**, 不是检测结果. 后端不校验它, 声明错了会静默解另一个方程
        组, 因此 residual 一项必须对完整矩阵算.
    """
    if backend == "pypardiso":
        return _pardiso_solve(operator, load, symmetry)

    if backend == "mumps":
        from soptx.core.mpi_runtime import ensure_mpi_initialized

        ensure_mpi_initialized()
        flag = SYMMETRY_FLAGS[symmetry]
    else:
        flag = 0

    start = time.perf_counter()
    solution = spsolve(operator, load, solver=backend, sym=flag)
    elapsed = time.perf_counter() - start
    return np.asarray(bm.to_numpy(solution)), elapsed


# --------------------------------------------------------------------------
# 报告排版
# --------------------------------------------------------------------------
def display_width(text: str) -> int:
    """终端里占的列数: 东亚宽字符 (W) 与全角字符 (F) 占两格, 其余按一格算."""
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def pad(text: str, width: int) -> str:
    """按显示宽度右补空格; str.ljust 数的是字符数, 含中文的列会对不齐."""
    return text + " " * max(width - display_width(text), 0)


def symmetry_label(case: dict[str, Any], symmetry: str) -> str:
    """对称性档位在报告里的显示名, 带上该后端的原生标志值."""
    if not case["symmetry"]:
        return "单档"
    if case["backend"] == "pypardiso":
        native = f"mtype={PARDISO_MATRIX_TYPES[symmetry]}"
    else:
        native = f"sym={SYMMETRY_FLAGS[symmetry]}"
    return f"{symmetry} ({native}, {SYMMETRY_LABELS[symmetry]})"


def verdict(ok: bool) -> str:
    """把布尔判定翻成报告里的两个字面量."""
    return "OK" if ok else "FAIL"


# --------------------------------------------------------------------------
# residual: 相对残差
# --------------------------------------------------------------------------
def check_residual(context: dict[str, Any], case: dict[str, Any], symmetries: list[str]):
    """对完整矩阵算相对残差 ||b - A x|| / ||b||.

    对完整矩阵而不是对后端实际吃进去的那半个三角算, 是这一项的要害: sym!=0 时
    后端只读一侧三角, 若声明与矩阵实际对称性不符, 它会解出另一个方程组的解而
    自身毫无察觉; 只有拿完整 A 回代才暴露得出来.
    """
    matrix = as_scipy(context["operator"]).tocsr()
    right_hand_side = np.asarray(bm.to_numpy(context["load"])).ravel()
    reference_norm = float(np.linalg.norm(right_hand_side))

    labels = {name: symmetry_label(case, name) for name in symmetries}
    width = max(display_width(text) for text in labels.values())
    threshold = TOLERANCES["residual_relative"]
    entries: dict[str, Any] = {}
    lines: list[str] = []
    passed = True
    for name in symmetries:
        solution, elapsed = solve_once(
            context["operator"], context["load"], case["backend"], name
        )
        absolute = float(np.linalg.norm(right_hand_side - matrix @ solution))
        relative = absolute / reference_norm if reference_norm > 0.0 else absolute
        ok = relative <= threshold
        passed = passed and ok
        entries[name] = {"relative_residual": relative, "seconds": elapsed}
        lines.append(
            f"  {pad(labels[name], width)}  {relative:.3e}"
            f"  ({elapsed:.4f} s)  [{verdict(ok)}]"
        )
    # 通过时不打印: 残差数字已进 JSON, 逐层残差也在 solution 的误差表里.
    if not passed:
        print("[residual] 相对残差 ||b - A x|| / ||b|| 超出阈值, 对完整矩阵算")
        for line in lines:
            print(line)
        print(f"  阈值 {threshold:.0e} -> {verdict(passed)}")
        print("")
    return {"passed": passed, "entries": entries}


# --------------------------------------------------------------------------
# solution: 制造解收敛阶
# --------------------------------------------------------------------------
_CONVERGENCE_MODULE: Any = None


def convergence_module():
    """按路径加载制造解收敛脚本并缓存.

    ``examples/`` 不是包, 只能按文件路径加载. 收敛阶的加密序列、观测阶公式与
    门禁阈值全在那边定义, 这里只借用, 不复制一份.
    """
    global _CONVERGENCE_MODULE
    if _CONVERGENCE_MODULE is None:
        path = (
            _REPOSITORY_ROOT
            / "examples"
            / "lagrange_elasticity"
            / "manufactured_convergence_demo.py"
        )
        spec = importlib.util.spec_from_file_location(
            "manufactured_convergence_demo", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CONVERGENCE_MODULE = module
    return _CONVERGENCE_MODULE


def check_solution(context: dict[str, Any], case: dict[str, Any], symmetries: list[str]):
    """在一串加密网格上跑制造解, 判观测收敛阶是否达标.

    这一项自带加密序列 (``--levels`` 与 ``--base``), 不使用 ``--n``: 单张网格上
    的一个 L2 误差数说明不了收敛性, 必须逐层比较才出得来观测阶.

    只跑 ``NOMINAL_SYMMETRY`` 一档. 观测收敛阶由离散化决定, 与后端内部走 LU 还
    是 LDL^T 无关; 对称性声明有没有被后端按约定解读, 由 residual 一项对完整矩阵
    求残差来判. 在这里重复扫 sym 只是把同一组误差算若干遍.
    """
    module = convergence_module()
    levels = context["levels"]
    name = NOMINAL_SYMMETRY if NOMINAL_SYMMETRY in symmetries else symmetries[0]
    entries: dict[str, Any] = {}
    passed = True
    # 被复用的 demo 按独立脚本写成, 自带整张加密表格、门禁段与结论段. 通过时这
    # 些细节没有价值, 吞掉只留逐层误差表; 失败时再把缓冲区原样吐出来, 细节不丢.
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            summary = module.run_manufactured_convergence_benchmark(
                dim=context["dimension"],
                model=context["model"],
                mesh_type=context["mesh_type"],
                degree=context["order"],
                levels=levels,
                base=context["base"],
                solver=case["backend"],
                mumps_sym=SYMMETRY_FLAGS[name],
            )
    except (AssertionError, ValueError) as error:
        passed = False
        entries[name] = {
            "symmetry": name,
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }
        print("[solution] 制造解收敛阶未达门禁")
        for line in captured.getvalue().splitlines():
            print(f"  | {line}")
        print(f"  [FAIL] {error}")
        print("")
    else:
        order = summary["final_l2_order"]
        theoretical = summary["theoretical_order"]
        gate = summary["minimum_l2_order_gate"]
        rows = summary["levels"]
        orders = level_orders(rows)
        entries[name] = {
            "symmetry": name,
            "levels": [
                {
                    "subdivisions": row["subdivisions"],
                    "mesh_size": row["mesh_size"],
                    "dofs": row["dofs"],
                    "l2_error": row["l2_error"],
                    "l2_relative": row["l2_relative"],
                    "l2_order": level_order,
                    "residual": row["residual"],
                    "seconds": row["seconds"],
                }
                for row, level_order in zip(rows, orders)
            ],
            "final_l2_order": order,
            "max_residual": summary["max_residual"],
            "gate": gate,
            "theoretical_order": theoretical,
            "passed": True,
        }
        # 通过时的全部输出就是这张表; 门禁值与理论阶进 JSON.
        print_error_table(rows, orders)
    return {"passed": passed, "entries": entries}


def level_orders(rows: list[dict[str, Any]]) -> list[float | None]:
    """逐层观测阶: 每层 h 减半, 阶即相邻两层 L2 误差比值的以 2 为底对数.

    第一层没有上一层可比, 记 ``None``; 误差为 0 (或负) 时同样记 ``None``.
    """
    orders: list[float | None] = [None]
    for coarse, fine in zip(rows[:-1], rows[1:]):
        if coarse["l2_error"] > 0.0 and fine["l2_error"] > 0.0:
            orders.append(math.log2(coarse["l2_error"] / fine["l2_error"]))
        else:
            orders.append(None)
    return orders


def print_error_table(rows: list[dict[str, Any]], orders: list[float | None]) -> None:
    """按 n | gdof | h | ||u-u_h||_0 | order | residual 打印逐层误差表."""
    header = (
        f"{'n':>5} {'gdof':>9} {'h':>9} {'||u-u_h||_0':>13} {'order':>7} {'residual':>10}"
    )
    print(header)
    print("-" * len(header))
    for row, order in zip(rows, orders):
        order_text = "—" if order is None else f"{order:.3f}"
        print(
            f"{row['subdivisions']:>5} {row['dofs']:>9} {row['mesh_size']:>9.5f}"
            f" {row['l2_error']:>13.4e} {order_text:>7} {row['residual']:>10.2e}"
        )


# --------------------------------------------------------------------------
# ownership: 调用方矩阵的所有权
# --------------------------------------------------------------------------
# 两档输入: FEALPy 稀疏算子是主路径, 原生 scipy CSR 是 substructure 接口传进来
# 的路径, 也正是 _scipy_solve 里那句 tocsr(copy=True) 唯一起作用的地方.
INPUT_ARMS = ("tensor", "csr")
INPUT_ARM_LABELS = {"tensor": "FEALPy 算子", "csr": "scipy CSR"}


def ownership_evidence(
    arm: str, case: dict[str, Any], symmetry: str, scipy_format: str
) -> str:
    """判定这一格的"未被改写"算强证据还是弱证据.

    只要输入在到达后端之前必然被复制过一次, 后端就根本碰不到调用方的内存, 这时
    候通过是结构上必然的, 不构成后端不改写输入的证据. 老实标出来, 免得把"测不
    到"读成"测过了".
    """
    if arm == "tensor" and scipy_format != "csr":
        return "弱: 导出为 " + scipy_format + ", tocsr() 即复制"
    if case["backend"] == "mumps" and symmetry != "general":
        return "弱: sym!=0 时 tril() 另建矩阵"
    if case["backend"] == "pypardiso" and symmetry != "general":
        return "弱: mtype!=11 时 triu() 另建矩阵"
    return "强: 输入直接交给后端"


def check_ownership(context: dict[str, Any], case: dict[str, Any], symmetries: list[str]):
    """求解前后比对调用方矩阵的非零值与列索引.

    这一项通过时不打印: 它检的是"什么都没发生", 逐格罗列一串 0.000e+00 只会把
    真正有内容的 residual 与 solution 挤下去. 一旦有格子失败, 整张表连同证据强
    弱一起打出来 -- 那时候每一格都是判读线索.
    """
    scipy_format = context["layout"]["scipy_format"]
    threshold = TOLERANCES["matrix_perturbation_relative"]
    entries: dict[str, Any] = {}
    lines: list[str] = []
    passed = True
    for name in symmetries:
        for arm in INPUT_ARMS:
            if arm == "tensor":
                candidate = context["operator"]
            else:
                candidate = as_scipy(context["operator"]).tocsr(copy=True)

            values_before, indices_before = matrix_fingerprint(candidate)
            solve_once(candidate, context["load"], case["backend"], name)
            values_after, indices_after = matrix_fingerprint(candidate)

            _, value_change = relative_difference(values_after, values_before)
            index_change = int(np.count_nonzero(indices_after != indices_before))
            ok = value_change <= threshold and index_change == 0
            passed = passed and ok
            evidence = ownership_evidence(arm, case, name, scipy_format)
            entries[f"{name}/{arm}"] = {
                "value_change": value_change,
                "index_change": index_change,
                "evidence": evidence,
            }
            label = f"{symmetry_label(case, name)} + {INPUT_ARM_LABELS[arm]}"
            lines.append(
                f"  {label}: 数值相对变化 {value_change:.3e}, 索引变动 "
                f"{index_change} 处  [{verdict(ok)}]  {evidence}"
            )

    if not passed:
        print("[ownership] 求解后调用方持有的矩阵被改写")
        for line in lines:
            print(line)
        print("")
    return {"passed": passed, "entries": entries}


# --------------------------------------------------------------------------
# guard: 非法参数
# --------------------------------------------------------------------------
def check_guard(context: dict[str, Any], case: dict[str, Any]):
    """非法 solver 名与非法 sym 值都应当抛 ValueError 而不是静默降级.

    与 ownership 同理, 通过时不打印: "该抛的都抛了"没有可读的内容, 只在有探针
    没抛、或抛错了类型时才把整组结果打出来.
    """
    operator = context["operator"]
    load = context["load"]

    probes: list[tuple[str, Any]] = [
        ("非法 solver 名 umfpack", lambda: spsolve(operator, load, solver="umfpack")),
    ]
    if case["backend"] == "mumps":
        probes.append(
            ("非法 sym=3", lambda: spsolve(operator, load, solver="mumps", sym=3))
        )
        probes.append(
            ("非法 sym=-1", lambda: spsolve(operator, load, solver="mumps", sym=-1))
        )

    entries: dict[str, Any] = {}
    lines: list[str] = []
    passed = True
    for description, probe in probes:
        try:
            probe()
        except ValueError as error:
            entries[description] = {"raised": "ValueError", "message": str(error)}
            lines.append(f"  {description}: 抛出 ValueError  [OK]")
            continue
        except Exception as error:  # noqa: BLE001 - 抛错类型不对也是失败, 要记下来
            passed = False
            entries[description] = {
                "raised": type(error).__name__,
                "message": str(error),
            }
            lines.append(
                f"  {description}: 抛出 {type(error).__name__} 而非 ValueError  [FAIL]"
            )
            continue
        passed = False
        entries[description] = {"raised": None, "message": "未抛异常"}
        lines.append(f"  {description}: 未抛异常  [FAIL]")

    if not passed:
        print("[guard] 非法参数未按约定抛 ValueError")
        for line in lines:
            print(line)
        print("")
    return {"passed": passed, "entries": entries}


CHECK_FUNCTIONS = {
    "residual": check_residual,
    "solution": check_solution,
    "ownership": check_ownership,
}


def run_case(context: dict[str, Any], case: dict[str, Any], requested_symmetries):
    """跑完一个 case 的全部 checks, 返回该 case 的结论."""
    symmetries = resolve_symmetries(case, requested_symmetries)
    case_id = case["id"]

    # --symmetry 把这个 case 支持的档位筛空了: 一项都没跑过, 不能算通过.
    if not symmetries:
        supported = " ".join(case["symmetry"])
        print(f"--symmetry 与 case {case_id} 无交集; 该 case 支持: {supported}")
        return {"passed": False, "symmetries": [], "checks": {}}

    # 通过时的输出只有 solution 的误差表. 单个 case 时表无需署名; 多个 case
    # 时用一行 case=... 区分各自的表. 其余检查通过时不打印, 失败时自行发声.
    if context["multiple_cases"]:
        print("")
        print(f"case={case_id}")
    results: dict[str, Any] = {}
    passed = True
    for check in case["checks"]:
        if check == "guard":
            outcome = check_guard(context, case)
        else:
            outcome = CHECK_FUNCTIONS[check](context, case, symmetries)
        results[check] = outcome
        passed = passed and outcome["passed"]

    return {"passed": passed, "symmetries": symmetries, "checks": results}


# --------------------------------------------------------------------------
def list_cases(
    dimension: int, model: str, mesh_type: str, resolution: int, degree: int
) -> int:
    """打印三个 case 裸跑会跑出什么组合, 让调用方拿到 id 就知道该选哪个.

    problem / mesh / analyzer 三列对三个 case 同值 -- 它们是全局轴, 不随后端变.
    列出来是为了回答"这三个 case 是在什么矩阵上受检的": 直接法的填充与性能只由
    稀疏模式决定, 而稀疏模式正是由问题/网格/阶数/分析器定下来的. 本函数接受已
    解析的轴而非常量, 因此 --list 与 --dim/--model/--mesh/--n/--order 同时给出
    时, 这几列显示的是本次真正会用的组合.
    """
    problem_text = f"{model} {dimension}D"
    mesh_text = f"{mesh_type} " + "x".join([str(resolution)] * dimension)
    analyzer_text = f"{SYSTEM_ANALYZERS[DEFAULT_SYSTEM]} P{degree}"
    header = ("case-id", "problem", "mesh", "analyzer", "library")
    rows = [
        (
            case["id"],
            problem_text,
            mesh_text,
            analyzer_text,
            case["library"],
        )
        for case in CASE_REGISTRY
    ]

    widths = [
        max(display_width(row[index]) for row in (header, *rows))
        for index in range(len(header))
    ]
    for row in (header, *rows):
        print(
            "  ".join(
                pad(value, widths[index]) for index, value in enumerate(row)
            ).rstrip()
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="三种稀疏直接法后端的正确性验证")
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出三个 case 及其缺省组合, 不做任何求解",
    )
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3), help="空间维数, 默认 2")
    parser.add_argument("--n", type=int, default=40, help="每方向单元数, 默认 40")
    parser.add_argument("--order", type=int, default=1, help="有限元阶数, 默认 1")
    parser.add_argument(
        "--model",
        default=None,
        help="制造解模型, 默认按维数取 " + str(DEFAULT_MODELS),
    )
    parser.add_argument(
        "--mesh",
        default=None,
        choices=sorted(MESH_DIMENSIONS),
        help="网格类型, 默认按维数取 " + str(DEFAULT_MESH_TYPES),
    )
    parser.add_argument(
        "--case",
        nargs="+",
        choices=[case["id"] for case in CASE_REGISTRY],
        help="要验证的 case id, 可多选; 除 --list 外必选. 显式要求的后端不可用即失败",
    )
    parser.add_argument(
        "--symmetry",
        nargs="+",
        default=None,
        choices=sorted(SYMMETRY_FLAGS),
        help="对称性声明档位, 默认取遍该 case 支持的全部档位; 对 scipy 不适用",
    )
    parser.add_argument(
        "--levels", type=int, default=5, help="solution 一项的加密层数, 默认 5"
    )
    parser.add_argument(
        "--base",
        type=int,
        default=None,
        help="solution 一项最粗一档的每方向单元数, 默认取收敛脚本的缺省值",
    )
    parser.add_argument("--json", action="store_true", help="把结果写入 outputs/")
    arguments = parser.parse_args()

    if not arguments.list and not arguments.case:
        parser.error("需要给出 --case, 或用 --list 查看有哪些 case")

    dimension = arguments.dim
    model = arguments.model or DEFAULT_MODELS[dimension]
    mesh_type = arguments.mesh or DEFAULT_MESH_TYPES[dimension]
    if MESH_DIMENSIONS[mesh_type] != dimension:
        print(
            f"网格 {mesh_type!r} 是 {MESH_DIMENSIONS[mesh_type]}D, 与 --dim {dimension} 不符"
        )
        return 1

    if arguments.list:
        return list_cases(dimension, model, mesh_type, arguments.n, arguments.order)

    cases = resolve_cases(arguments.case)
    if cases is None:
        return 1

    # 横幅按 key=value 报问题、网格分辨率、网格类型与空间次数: 这里验证的是
    # 代数方程组求解, 后两项不是被验证的对象, 但它们决定矩阵的稀疏模式与条件数.
    grid = ", ".join([str(arguments.n)] * dimension)
    print(
        f"直接法验证: problem={model}, grid={grid}, mesh={mesh_type},"
        f" order={arguments.order}"
    )
    print("cases: " + " ".join(case["id"] for case in cases))
    print("")

    operator, load, mesh, space, problem = build_system(
        dimension, arguments.n, arguments.order, model, mesh_type
    )
    layout = describe_operator(operator)
    number_of_dofs = int(space.number_of_global_dofs())

    context = {
        "operator": operator,
        "load": load,
        "mesh": mesh,
        "space": space,
        "problem": problem,
        "layout": layout,
        "dimension": dimension,
        "model": model,
        "mesh_type": mesh_type,
        "order": arguments.order,
        "levels": arguments.levels,
        "base": arguments.base,
        "multiple_cases": len(cases) > 1,
    }

    case_results: dict[str, Any] = {}
    for case in cases:
        case_results[case["id"]] = run_case(context, case, arguments.symmetry)

    report = {
        "dimension": dimension,
        "model": model,
        "mesh_type": mesh_type,
        "resolution": arguments.n,
        "order": arguments.order,
        "number_of_dofs": number_of_dofs,
        "operator_layout": layout,
        "tolerances": TOLERANCES,
        "cases": case_results,
    }

    if arguments.json:
        output_directory = Path(__file__).resolve().parent / "outputs"
        output_directory.mkdir(exist_ok=True)
        target = output_directory / (
            f"verify_direct_solvers_{dimension}d_{mesh_type}_{model}"
            f"_n{arguments.n}_p{arguments.order}.json"
        )
        target.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # 通过时不打结论: 退出码 0 就是结论. 失败时点名哪些 case 没过.
    failed = [name for name, entry in case_results.items() if not entry["passed"]]
    if failed:
        print("")
        print("判定: FAIL  (" + " ".join(failed) + ")")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
