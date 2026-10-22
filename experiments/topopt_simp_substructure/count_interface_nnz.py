"""按网格结构统计全局边界总刚 K_Gamma 与细网格总刚 K 的非零元个数.

只依赖结构化六面体网格与子结构划分的拓扑, 不装配矩阵, 也不依赖密度: 三线性元下 K 的两个节点
同属一个单元即耦合; K_Gamma 的两个边界节点同属一个子结构即耦合 (子结构缩聚刚度 K_s^j 为满阵).
节点层计数乘 9 即自由度层非零元个数. 结果用于 substructure_vs_fem.md §3.2.
"""

import argparse

import numpy as np
import scipy.sparse as sp

GRID = (210, 35, 35)
N_FINE = 5


def parse_args(argv=None):
    """解析细网格与子结构细分数.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数, 缺省时读取 ``sys.argv``.

    Returns
    -------
    args : argparse.Namespace
        含细网格 ``grid`` 与每个子结构各方向细单元数 ``n_fine``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, nargs=3, default=GRID, metavar=('NX', 'NY', 'NZ'),
                        help=f'细网格各方向单元数, 默认 {" ".join(map(str, GRID))}')
    parser.add_argument('--n-fine', type=int, default=N_FINE,
                        help=f'每个子结构各方向细单元数 m, 默认 {N_FINE}')
    args = parser.parse_args(argv)
    if args.n_fine < 2:
        parser.error('--n-fine 至少为 2, 以保留内部节点')
    if any(n < 1 or n % args.n_fine for n in args.grid):
        parser.error(f'--grid 各方向须为 --n-fine={args.n_fine} 的正整数倍')
    return args


def count_nnz(grid, m):
    """统计 K 与 K_Gamma 的维数与非零元个数.

    Parameters
    ----------
    grid : tuple of int
        细网格各方向单元数.
    m : int
        每个子结构各方向细单元数.

    Returns
    -------
    counts : dict
        ``n_dofs`` 与 ``nnz`` 为 K 的维数与非零元个数, ``n_dofs_gamma`` 与 ``nnz_gamma`` 为 K_Gamma 的.
    """
    nx, ny, nz = grid
    n_nodes = (nx + 1) * (ny + 1) * (nz + 1)
    # K: 各方向上相邻 (含自身) 的节点对数为 3 (n + 1) - 2, 三个方向相乘即耦合节点对数
    pairs = 1
    for n in grid:
        pairs *= 3 * (n + 1) - 2

    # K_Gamma: 节点-子结构关联矩阵 B 只含子结构边界节点, B B^T 的非零模式即耦合节点对
    r = np.arange(m + 1)
    i, j, k = np.meshgrid(r, r, r, indexing='ij')
    on_boundary = (i == 0) | (i == m) | (j == 0) | (j == m) | (k == 0) | (k == m)
    li, lj, lk = i[on_boundary], j[on_boundary], k[on_boundary]
    a, b, c = (s.ravel() for s in np.meshgrid(*(np.arange(n // m) for n in grid), indexing='ij'))
    gi = a[:, None] * m + li
    gj = b[:, None] * m + lj
    gk = c[:, None] * m + lk
    rows = ((gi * (ny + 1) + gj) * (nz + 1) + gk).ravel()
    cols = np.repeat(np.arange(a.size), li.size)
    incidence = sp.csr_matrix((np.ones(rows.size, dtype=np.int32), (rows, cols)), shape=(n_nodes, a.size))
    incidence = incidence[np.unique(rows)]
    pairs_gamma = (incidence @ incidence.T).nnz
    return dict(n_dofs=3 * n_nodes, nnz=9 * pairs,
                n_dofs_gamma=3 * incidence.shape[0], nnz_gamma=9 * pairs_gamma)


def main(argv=None):
    """打印 K 与 K_Gamma 的维数、平均每行非零元与非零元总数.

    Parameters
    ----------
    argv : list of str, optional
        命令行参数, 缺省时读取 ``sys.argv``.

    Returns
    -------
    status : int
        正常结束时为 0.
    """
    args = parse_args(argv)
    counts = count_nnz(tuple(args.grid), args.n_fine)
    print(f'网格 {tuple(args.grid)}, m={args.n_fine}')
    for name, n, nnz in (('K', counts['n_dofs'], counts['nnz']),
                         ('K_Gamma', counts['n_dofs_gamma'], counts['nnz_gamma'])):
        print(f'{name:8s} 维数 {n:>10d}, 平均每行 {nnz / n:7.1f}, 非零元 {nnz:.4e}')
    print(f'维数比 {counts["n_dofs_gamma"] / counts["n_dofs"]:.3f}, '
          f'非零元比 {counts["nnz_gamma"] / counts["nnz"]:.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
