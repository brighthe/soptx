"""预条件子筛选的合成输入密度: 三维 MBB 梁 (``FullMBBBeamPoint3d``) 上的均匀、0/1 桁架与过滤桁架.

不执行优化, 在给定基准网格上直接构造三种单元物理密度, 写入本目录供
``run_fem_preconditioners.py --density`` 读取:

- ``uniform.npy``: 处处取 VOLFRAC, 对应优化第 1 轮的常系数问题.
- ``binary.npy``: 0/1 桁架, 代表成形后的高反差细杆件. 杆件在 x-y 平面内以线段给出, 宽
  MEMBER_WIDTH_CELLS 个基准单元; 上弦贯穿全宽以承接顶面中心的集中力, 其余杆件只在 z 两侧面各一层
  (中间空心, 与 48x8x8 真实设计的截面一致), 下弦与拱腿经两端底角支座. 构造后检查全部实体单元
  面连通且支座与加载点单元均为实体.
- ``gray.npy``: 以优化所用的同一密度过滤器 (半径 FILTER_RADIUS_CELLS 个单元) 过滤 binary.npy,
  代表结构已成雏形但密度仍灰的中期.

坐标以基准单元尺寸为 1, 物理域为 [0, NX] x [0, NY] x [0, NZ]. 三个文件与生成参数整份重写
sources.json; 换基准网格即覆盖上一套.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from soptx.backend import backend_manager as bm
from soptx.mesh import HexahedronMesh
from soptx.topology.filters import Filter


# 均匀密度取值, 与 run_full_scale_fem.py 的 VOLFRAC 相同
VOLFRAC = 0.12
# 桁架杆宽 (基准单元数); 须大于 sqrt(2) / 2 * 2, 否则 45 度斜杆的单元只共棱不共面, 面连通断开
MEMBER_WIDTH_CELLS = 1.5
# 灰度设计的过滤半径 (基准单元数), 与 run_full_scale_fem.py 的 FILTER_RADIUS_CELLS 相同
FILTER_RADIUS_CELLS = 3.0

GRID = (48, 8, 8)
OUTPUT_DIR = Path(__file__).resolve().parent


def parse_args(argv=None):
    """解析基准网格, 其余设置均为模块顶部常量.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, nargs=3, default=GRID, metavar=('NX', 'NY', 'NZ'),
                        help=f'基准网格各方向单元数, 须满足 NX = 6 NY, NY = NZ (默认 {GRID})')
    args = parser.parse_args(argv)
    nx, ny, nz = args.grid
    if nx != 6 * ny or ny != nz or ny < 4:
        parser.error(f'--grid 须满足 NX = 6 NY, NY = NZ 且 NY >= 4, 得到 {tuple(args.grid)}')
    args.grid = tuple(args.grid)
    return args


def main(argv=None):
    """构造三种合成密度, 写入 NPY 并更新 sources.json.
    """
    args = parse_args(argv)
    bm.set_backend('numpy')
    nx, ny, nz = args.grid
    length, height, width = float(nx), float(ny), float(nz)
    half = MEMBER_WIDTH_CELLS / 2

    # 1. 桁架杆件: x-y 平面内的线段 (x0, y0, x1, y1) 与 z 向分布 ('full' 贯穿全宽, 'walls' 只在两侧面);
    #    a 为上弦端点到梁端的水平距离, 拱腿自底角支座斜向上接上弦端点
    a = length / 6
    top, bottom = height - half, half
    members = [
        ('top_chord', (a, top, length - a, top), 'full'),
        ('bottom_chord', (0.0, bottom, length, bottom), 'walls'),
        ('left_leg', (half, bottom, a, top), 'walls'),
        ('right_leg', (length - half, bottom, length - a, top), 'walls'),
        ('vertical_1', (length / 3, bottom, length / 3, top), 'walls'),
        ('vertical_2', (length / 2, bottom, length / 2, top), 'walls'),
        ('vertical_3', (2 * length / 3, bottom, 2 * length / 3, top), 'walls'),
        ('diagonal_left', (length / 2, top, length / 3, bottom), 'walls'),
        ('diagonal_right', (length / 2, top, 2 * length / 3, bottom), 'walls'),
    ]

    # 单元中心到线段的距离严格小于半杆宽即为实体; 两侧面各取同样宽度的一层
    xc, yc = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing='ij')
    zc = np.arange(nz) + 0.5
    z_masks = {'full': np.ones(nz, dtype=bool),
               'walls': (np.abs(zc - half) < half) | (np.abs(zc - (width - half)) < half)}
    binary_mask = np.zeros((nx, ny, nz), dtype=bool)
    for _, (x0, y0, x1, y1), z_kind in members:
        dx, dy = x1 - x0, y1 - y0
        t = np.clip(((xc - x0) * dx + (yc - y0) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        distance = np.hypot(xc - (x0 + t * dx), yc - (y0 + t * dy))
        binary_mask |= (distance < half)[:, :, None] & z_masks[z_kind][None, None, :]
    binary = binary_mask.astype(np.float64)

    # 面连通 (6 邻域) 检查: 全部实体单元连成一个分量, 且两端底角支座与顶面中心加载点所在单元均为实体
    labels, n_components = ndimage.label(binary_mask)
    key_cells = [(0, 0, 0), (0, 0, nz - 1), (nx - 1, 0, 0), (nx - 1, 0, nz - 1)]
    key_cells += [(i, ny - 1, k) for i in (nx // 2 - 1, nx // 2) for k in (nz // 2 - 1, nz // 2)]
    if n_components != 1 or not all(binary_mask[cell] for cell in key_cells):
        raise RuntimeError(f'桁架不连通或支座、加载点单元非实体: {n_components} 个面连通分量, '
                           f'关键单元 {[bool(binary_mask[cell]) for cell in key_cells]}')

    # 2. 灰度设计: 与优化相同的密度过滤 (锥形权重, 半径按基准单元计)
    mesh = HexahedronMesh.from_box([0.0, length, 0.0, height, 0.0, width], nx, ny, nz)
    density_filter = Filter(design_mesh=mesh, filter_type='density', rmin=FILTER_RADIUS_CELLS,
                            density_location='element')
    n_cells = nx * ny * nz
    gray = bm.to_numpy(density_filter.filter_design_variable(
        design_variable=bm.from_numpy(binary.reshape(-1)),
        physical_density=bm.zeros((n_cells, ), dtype=bm.float64))[:]).reshape(nx, ny, nz)

    uniform = np.full((nx, ny, nz), VOLFRAC, dtype=np.float64)

    # 3. 写文件并整份重写 sources.json
    sources = dict(description='预条件子筛选 (run_fem_preconditioners.py --density) 的输入单元物理密度, '
                               '形状 (NX, NY, NZ) 即基准网格, 单元按 x 最慢、z 最快排列, 由 make_densities.py 合成',
                   files={})
    generator = dict(script='experiments/topopt_simp_fem/fem_preconditioners/make_densities.py',
                     command=f'python experiments/topopt_simp_fem/fem_preconditioners/make_densities.py '
                             f'--grid {nx} {ny} {nz}')
    truss = dict(member_width_cells=MEMBER_WIDTH_CELLS,
                 members=[dict(name=name, segment_xy=list(segment), z=z_kind)
                          for name, segment, z_kind in members])

    def statistics(field):
        return dict(shape=list(field.shape), min=float(field.min()), max=float(field.max()),
                    mean=float(field.mean()), fraction_solid_gt_0_99=float((field > 0.99).mean()),
                    fraction_intermediate_0_1_to_0_9=float(((field > 0.1) & (field < 0.9)).mean()))

    entries = {
        'uniform.npy': (uniform, dict(kind='均匀物理密度, 处处取 VOLFRAC, 对应优化第 1 轮 (初始设计)',
                                      value=VOLFRAC)),
        'binary.npy': (binary, dict(kind='0/1 桁架, 代表成形后的高反差细杆件', truss=truss,
                                    connectivity='scipy.ndimage.label 默认 6 邻域 (共面相连)',
                                    n_components=int(n_components), supports_and_load_connected=True)),
        'gray.npy': (gray, dict(kind='binary.npy 经密度过滤的灰度设计, 代表结构已成雏形但密度仍灰的中期',
                                derived_from='binary.npy', filter_type='density',
                                filter_radius_cells=FILTER_RADIUS_CELLS)),
    }
    for name, (field, info) in entries.items():
        np.save(OUTPUT_DIR / name, field)
        digest = hashlib.sha256((OUTPUT_DIR / name).read_bytes()).hexdigest()
        sources['files'][name] = dict(info, source=dict(kind='synthetic', **generator),
                                      **statistics(field), sha256=digest)
        print(f'[{name}] 网格 {nx}x{ny}x{nz}, 均值 {field.mean():.4f}, 最大 {field.max():.4f}, '
              f'实体 (>0.99) {(field > 0.99).mean():.3f}, 中间 (0.1-0.9) '
              f'{((field > 0.1) & (field < 0.9)).mean():.3f}', flush=True)
    (OUTPUT_DIR / 'sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[完成] 桁架面连通分量 {n_components}, 结果写入 {OUTPUT_DIR}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
