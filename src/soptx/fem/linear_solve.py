"""Lagrange 位移有限元分析器的线性求解策略.

按名字构造求解器 (``build_solver``), 在给定算子上运行并整理诊断 (``run_solver``).
与分析器状态有关的部分 -- 预条件子绑定哪个算子, 几何多重网格的层次 -- 留在分析器,
以构造好的预条件子传入. 分布式与子结构分析器各有自己的求解路径, 不经过本模块.

选项字典由调用方合并好再传入, 约定优先级为: 求解调用的 kwargs > 构造时的
``solver_options`` > 本模块的默认值.
"""

from typing import Any, Dict, Optional, Tuple

from soptx.backend import backend_manager as bm
from soptx.solvers import LinearSolver, create
from soptx.sparse import COOTensor, CSRTensor

DEFAULT_CG_MAXITER = 5000
DEFAULT_CG_ATOL = 1e-12
DEFAULT_CG_RTOL = 1e-12
# 带预条件子时周期性用 b - A x 校正递推残差的间隔
DEFAULT_RESIDUAL_REFRESH = 50


def build_solver(solver_type: str,
                 options: Dict[str, Any],
                 *,
                 preconditioner: Optional[LinearSolver] = None,
            ) -> Tuple[LinearSolver, Dict[str, Any], Optional[Tuple[float, float]]]:
    """按名字构造尚未 setup 的求解器.

    Parameters
    ----------
    solver_type : 求解器名字, 须已在 ``soptx.solvers.registry`` 注册.
    options : 合并后的求解选项. 'cg' 读 maxiter, atol, rtol, residual_refresh 与 precond
        (只记入诊断); 'mumps' 读 sym.
    preconditioner : 已 setup 的预条件子, 只用于 'cg'; 为 None 时不带预条件.

    Returns
    -------
    solver : 尚未 setup 的求解器.
    extra : 并入诊断的后端专属键.
    tol : 迭代解法的 (atol, rtol), 直接法为 None.

    Notes
    -----
    带预条件子时 CG 改在 ||r||_2 下停机 (norm_type='unpreconditioned'): 默认的 natural
    范数 sqrt(r^T M^-1 r) 与分析器返回的 2-范数 relres 口径不同, Jacobi 的 diag^-1 可达
    1e6 量级, 两者能差几个数量级; 同时开启递推残差刷新 (未显式给出时取 50), 抵消长迭代
    下的漂移. 不带预条件子时三种范数数值等价, 取默认的 natural.

    'mumps' 的 sym=0 按一般非对称矩阵分解; 对称消元后的刚度阵仍对称正定, 传 1 (正定)
    或 2 (一般对称) 只读下三角, 因子存储与运算量大致减半, 由调用方显式开启.
    """
    if solver_type == 'cg':
        maxiter = options.get('maxiter', DEFAULT_CG_MAXITER)
        atol = options.get('atol', DEFAULT_CG_ATOL)
        rtol = options.get('rtol', DEFAULT_CG_RTOL)
        residual_refresh = int(options.get('residual_refresh', 0))
        norm_type = 'natural'
        if preconditioner is not None:
            norm_type = 'unpreconditioned'
            if residual_refresh <= 0:
                residual_refresh = DEFAULT_RESIDUAL_REFRESH

        # batch_first=False: 批量右端项的第一维是自由度维
        solver = create('cg', M=preconditioner, atol=atol, rtol=rtol, maxit=maxiter,
                        batch_first=False,
                        norm_type=norm_type,
                        residual_refresh=residual_refresh)

        return solver, {'maxit': maxiter, 'precond': options.get('precond', None)}, (atol, rtol)

    if solver_type == 'mumps':
        mumps_sym = int(options.get('sym', 0))

        return create('mumps', sym=mumps_sym), {'sym': mumps_sym}, None

    return create(solver_type), {}, None


def run_solver(solver: LinearSolver,
               operator: Any,
               F: Any,
               out: Any,
               *,
               name: str,
               extra: Dict[str, Any],
               tol: Optional[Tuple[float, float]],
               x0: Optional[Any] = None,
            ) -> Dict[str, Any]:
    """在算子上 setup 并求解, 解就地写入 ``out``, 返回整理后的诊断.

    Parameters
    ----------
    solver : ``build_solver`` 构造的求解器.
    operator : 要绑定的算子.
    F : 右端项, (gdof, ) 或批量的 (gdof, nrhs).
    out : 就地写入的解向量.
    name : 求解器名字, 记入诊断.
    extra : ``build_solver`` 给出的后端专属诊断键.
    tol : ``build_solver`` 给出的 (atol, rtol); 不为 None 时附加迭代解法的诊断键.
    x0 : 迭代解法的初值; 为 None 时从零开始.

    Returns
    -------
    info : 含 'name', 'niter', 'relres', 'converged' 与 ``extra``; 迭代解法另含 'reason',
        'reference_norm', 'recursive_residual' 与 'true_residual'.

    Raises
    ------
    OperatorCapabilityError
        算子给不出求解器所需的能力 (如直接法遇到矩阵自由算子); 由调用方补上下文后报错.

    Notes
    -----
    分解不跨调用复用: 直接法持有的 SuperLU 分解或 MUMPS 上下文用完即释放, 预条件子位上的
    直接法持有另一份, 一并释放, 否则 MUMPS 侧的内存不回收.
    """
    try:
        solver.setup(operator)
        out[:], raw = solver.solve(F[:], x0)
    finally:
        for owner in (solver, getattr(solver, 'M', None)):
            close = getattr(owner, 'close', None)
            if close is not None:
                close()

    info = {'name': name, **extra,
            'niter': int(raw['niter']),
            'relres': float(raw['relres']),
            'converged': bool(raw['converged'])}

    if tol is not None:
        # 收敛与否以求解器自己的退出原因为准, 不在这里重判: 热启动时 rtol 的参照量是
        # ||r0|| 而非 ||F||, 判据口径已在求解器内部对齐 (见 build_solver 的 norm_type)
        reason = raw.get('reason', None)
        if reason is not None:
            info['reason'] = reason.name
        info['reference_norm'] = float(raw.get('reference_norm', 0.0))
        info['recursive_residual'] = float(raw['residual'])
        true_residual = raw.get('true_residual', None)
        info['true_residual'] = None if true_residual is None else float(true_residual)

    return info


def as_iterative_operator(K: Any) -> Any:
    """把算子转成迭代解法可以直接作用的形式.

    Parameters
    ----------
    K : 已施加边界条件的系统算子; 'fa' 下为稀疏矩阵, 其余层级下为支持 @ 的算子对象.

    Returns
    -------
    operator : PyTorch 后端下的稀疏矩阵转为 torch 原生 CSR, 其余原样返回.

    Notes
    -----
    numpy 后端的 csr_spmm 就是 scipy 的 csr_matvec, 直接用 CSR; 转 COO 只会多出 (2, nnz)
    的 int64 索引 (首档约 6 GiB). PyTorch 后端下 soptx 的稀疏张量 matmul 在 GPU 上有问题,
    改用 torch 原生稀疏矩阵.
    """
    if bm.backend_name != 'pytorch' or not isinstance(K, (CSRTensor, COOTensor)):
        return K

    import torch
    K_coo_torch = torch.sparse_coo_tensor(
                                    indices=bm.stack([K.row, K.col]),
                                    values=bm.tensor(K.data),
                                    size=K.shape,
                                    device=K.data.device
                                )
    #? matmul 函数下 K 必须是 COO 格式, 不能是 CSR 格式, 否则 GPU 下 device_put 函数会出错
    K._values = bm.copy(K._values)

    return K_coo_torch.to_sparse_csr()
