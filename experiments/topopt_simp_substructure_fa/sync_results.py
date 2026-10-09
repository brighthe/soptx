"""把选定的子结构与传统有限元运行从 outputs/ 固化到入库的 results/.

``outputs/`` 是运行脚本的临时产物, 不入库; ``results/`` 是 ``results_analysis.md`` 的数据来源,
入库. 对照按"对"组织, 每对是两个运行目录 A 与 B, A 是被考察的方法, B 是参照, 侧名由各自
``config.json`` 的 ``method`` 决定 (``fa``, ``full_trace``, ``linear_corner``):

- full_trace 对 fa: 精确子结构与传统有限元代数等价, 两侧差应为舍入级 (对应 2 节);
- linear_corner 对 full_trace 或 fa: 角点线性插值是方法近似, 两侧差是方法误差 (对应 3 节).

传统有限元参照运行的来源约定: 只为本目录服务的运行 (210x35x35 与小档) 由
``../topopt_simp_fem/run_fa.py`` 加 ``--output-dir`` 直接写到本目录 ``outputs/``; 首档 390x65x65
的 FA 是 ``topopt_simp_fem`` 自己的运行, 不复制, 引用其 ``outputs/``.

本脚本对每一对做两件事:

1. 把两侧运行目录下的 ``config.json``, ``summary.json``, ``history.json`` 原样拷入
   ``results/<运行目录名>/``;
2. 读两侧末轮的 ``*_final.npy``, 写出 ``results/final_fields.json`` 中该对的条目: 各场的统计指纹
   与 sha256, 及两侧的逐场比对. npy 本身不入库.

各对可以分别固化: 某一对的 A 侧目录尚不存在时跳过该对, 保留 ``final_fields.json`` 中已有的其他对.
只依赖 numpy 与标准库, 不导入 soptx.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Tuple

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUTS = EXPERIMENT_DIR / 'outputs'
FEM_OUTPUTS = EXPERIMENT_DIR.parent / 'topopt_simp_fem' / 'outputs'
RUN_FILES = ('config.json', 'summary.json', 'history.json')
FIELD_FILES = ('density_final.npy', 'design_density_final.npy', 'displacement_final.npy')
# 两侧 config.json 中允许不同的键: 方法名与子结构剖分, 只有一侧才有的装配方式; 求解器设置按 1 节
# 表格由方法决定, 两侧不同是设计, 差异只提示不报错
METHOD_KEYS = {'method', 'trace', 'n_sub', 'n_fine', 'chunk_size', 'assembly_method'}
SOLVER_KEYS = {'solver', 'cg_options', 'cg_tolerance'}
# config.json 的 method 到侧名
METHOD_LABELS = {'fem_fa': 'fa', 'substructure_full_trace': 'full_trace',
                 'substructure_linear_corner': 'linear_corner'}
# 二值化阈值
SOLID_THRESHOLD = 0.5

# 默认固化的对: 名称, A 侧目录, B 侧目录
DEFAULT_PAIRS: List[Tuple[str, Path, Path]] = [
    ('full_trace_vs_fa_210x35x35',
     OUTPUTS / 'full_trace_hex_210x35x35_m5_cg', OUTPUTS / 'fa_hex_210x35x35_cg'),
    ('full_trace_vs_fa_60x10x10_mgcg',
     OUTPUTS / 'full_trace_hex_60x10x10_m5_cg', OUTPUTS / 'fa_hex_60x10x10_cg'),
    ('full_trace_vs_fa_60x10x10_direct',
     OUTPUTS / 'full_trace_hex_60x10x10_m5_scipy', OUTPUTS / 'fa_hex_60x10x10_scipy'),
    ('linear_corner_vs_full_trace_60x10x10',
     OUTPUTS / 'linear_corner_hex_60x10x10_m5_mumps', OUTPUTS / 'full_trace_hex_60x10x10_m5_cg'),
    ('linear_corner_vs_fa_390x65x65',
     OUTPUTS / 'linear_corner_hex_390x65x65_m5_mumps', FEM_OUTPUTS / 'fa_hex_390x65x65_cg'),
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='把选定的子结构与传统有限元运行从 outputs/ 固化到 results/')
    parser.add_argument('--pair', nargs=3, action='append', metavar=('NAME', 'DIR_A', 'DIR_B'),
                        help='要固化的一对运行: 名称与 A, B 两侧运行目录, 可重复; 不给时固化脚本内置的默认各对')
    parser.add_argument('--results-dir', type=Path, default=EXPERIMENT_DIR / 'results',
                        help='固化目标目录 (默认: results)')
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    """分块计算文件的 sha256.

    Parameters
    ----------
    path : 文件路径.

    Returns
    -------
    digest : 十六进制摘要.
    """
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def field_stats(a: np.ndarray) -> Dict[str, Any]:
    """场的统计指纹, 用于按容差比较重跑结果 (sha256 只能确认逐字节相同的文件).

    Parameters
    ----------
    a : 场数组.

    Returns
    -------
    stats : shape, dtype, min, max, mean 与 L2 范数.
    """
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'min': float(a.min()), 'max': float(a.max()),
            'mean': float(a.mean()), 'l2_norm': float(np.linalg.norm(a.ravel()))}


def display_path(path: Path) -> str:
    """相对实验目录的路径, 不在其下时为相对 experiments/ 的路径; 写入 json 以标明数据来自哪次运行."""
    for base in (EXPERIMENT_DIR, EXPERIMENT_DIR.parent):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def side_label(config: Dict[str, Any], run: Path) -> str:
    """由 config.json 的 method 得侧名.

    Parameters
    ----------
    config : 运行的 config.json 内容.
    run : 运行目录, 只用于报错.

    Returns
    -------
    label : ``METHOD_LABELS`` 中的侧名.

    Raises
    ------
    ValueError
        method 不在 ``METHOD_LABELS`` 中.
    """
    method = config.get('method')
    if method not in METHOD_LABELS:
        raise ValueError(f'{run} 的 method 为 {method!r}, 不在 {sorted(METHOD_LABELS)} 中')
    return METHOD_LABELS[method]


def pair_description(label_a: str, label_b: str) -> str:
    """按两侧方法给出这一对比对的性质."""
    if {label_a, label_b} == {'full_trace', 'fa'}:
        return 'full_trace 与 FA 代数等价, 两侧差应为舍入级'
    if 'linear_corner' in (label_a, label_b):
        return 'linear_corner 的角点线性插值是方法近似, 两侧差是方法误差而非舍入'
    return f'{label_a} 对 {label_b}'


def check_pair(name: str, runs: Dict[str, Path]) -> Tuple[str, str]:
    """检查一对运行齐全且可比, 返回两侧侧名.

    Parameters
    ----------
    name : 对名.
    runs : 'a', 'b' 到运行目录的映射.

    Returns
    -------
    label_a, label_b : 两侧侧名.

    Raises
    ------
    FileNotFoundError
        缺少所需文件; 缺 summary.json 说明运行未正常结束.
    ValueError
        两侧方法相同, 或 config.json 除方法与求解器外还有其他差异.
    """
    for side, run in runs.items():
        missing = [f for f in RUN_FILES + FIELD_FILES if not (run / f).is_file()]
        if missing:
            raise FileNotFoundError(f'{name} 的 {side} 侧运行目录 {run} 缺少 {missing}')
    configs = {side: json.loads((run / 'config.json').read_text(encoding='utf-8')) for side, run in runs.items()}
    label_a, label_b = side_label(configs['a'], runs['a']), side_label(configs['b'], runs['b'])
    if label_a == label_b:
        raise ValueError(f'{name} 两侧都是 {label_a}')
    keys = set(configs['a']) | set(configs['b'])
    differing = sorted(k for k in keys - METHOD_KEYS - SOLVER_KEYS if configs['a'].get(k) != configs['b'].get(k))
    if differing:
        raise ValueError(f'{name} 两侧 config.json 除方法与求解器外还有差异: {differing}')
    solver_differing = sorted(k for k in SOLVER_KEYS if configs['a'].get(k) != configs['b'].get(k))
    if solver_differing:
        print(f'[提示] {name} 两侧求解器设置不同: {solver_differing}', flush=True)
    return label_a, label_b


def compare_pair(runs: Dict[str, Path], labels: Tuple[str, str]) -> Dict[str, Any]:
    """一对运行末轮场的统计指纹, sha256 与逐场比对.

    Parameters
    ----------
    runs : 'a', 'b' 到运行目录的映射.
    labels : 两侧侧名.

    Returns
    -------
    record : 写入 final_fields.json 中该对的内容.
    """
    label_a, label_b = labels
    data = {side: {f: np.load(run / f) for f in FIELD_FILES} for side, run in runs.items()}
    fields = {}
    for side, label in (('a', label_a), ('b', label_b)):
        fields[label] = {'directory': display_path(runs[side])}
        for f in FIELD_FILES:
            entry = {'sha256': sha256(runs[side] / f), **field_stats(data[side][f])}
            if f == 'density_final.npy':
                entry['n_solid'] = int(np.count_nonzero(data[side][f] > SOLID_THRESHOLD))
            fields[label][f] = entry

    rho_a, rho_b = data['a']['density_final.npy'], data['b']['density_final.npy']
    x_a, x_b = data['a']['design_density_final.npy'], data['b']['design_density_final.npy']
    u_a, u_b = data['a']['displacement_final.npy'], data['b']['displacement_final.npy']
    solid_a, solid_b = rho_a > SOLID_THRESHOLD, rho_b > SOLID_THRESHOLD
    # 离阈值最近的单元与阈值之距; 远大于物理密度差时, 二值化结果相同不是巧合
    distance = min(np.min(np.abs(rho_a - SOLID_THRESHOLD)), np.min(np.abs(rho_b - SOLID_THRESHOLD)))
    ta, tb = label_a.upper(), label_b.upper()

    return {
        'description': pair_description(label_a, label_b),
        'sides': {'a': label_a, 'b': label_b},
        'fields': fields,
        'comparison': {
            'physical_density_max_abs_diff': {
                'definition': f'max |rho_{ta} - rho_{tb}|, density_final.npy',
                'value': float(np.max(np.abs(rho_a - rho_b)))},
            'design_density_max_abs_diff': {
                'definition': f'max |x_{ta} - x_{tb}|, design_density_final.npy',
                'value': float(np.max(np.abs(x_a - x_b)))},
            'displacement_relative_l2_diff': {
                'definition': f'||u_{ta} - u_{tb}||_2 / ||u_{tb}||_2, displacement_final.npy',
                'value': float(np.linalg.norm(u_a - u_b) / np.linalg.norm(u_b))},
            'binary_topology': {
                'definition': 'density_final.npy 取 rho > 0.5 后逐单元比较',
                'identical': bool(np.array_equal(solid_a, solid_b)),
                'n_mismatch': int(np.count_nonzero(solid_a != solid_b)),
                'min_distance_to_threshold': float(distance)},
        },
    }


def main(argv=None):
    args = parse_args(argv)
    results = args.results_dir.resolve()
    if args.pair:
        pairs = [(name, Path(a).resolve(), Path(b).resolve()) for name, a, b in args.pair]
    else:
        pairs = [(name, a.resolve(), b.resolve()) for name, a, b in DEFAULT_PAIRS]

    # A 侧目录不存在视为该对尚未运行, 跳过 (B 侧参照可能早已存在); 存在但缺文件则报错
    present = {name: {'a': a, 'b': b} for name, a, b in pairs if a.is_dir()}
    skipped = [name for name, a, _ in pairs if not a.is_dir()]
    if not present:
        raise FileNotFoundError('各对的 A 侧运行目录都不存在, 没有可固化的内容')
    labels = {name: check_pair(name, runs) for name, runs in present.items()}

    # 先算完比对再写盘, 中途出错不留下半新半旧的 results/
    records = {name: compare_pair(runs, labels[name]) for name, runs in present.items()}

    fields_path = results / 'final_fields.json'
    existing: Dict[str, Any] = {}
    if fields_path.is_file():
        existing = json.loads(fields_path.read_text(encoding='utf-8')).get('pairs', {})
    merged = {**existing, **records}

    results.mkdir(parents=True, exist_ok=True)
    for runs in present.values():
        for run in runs.values():
            target = results / run.name
            target.mkdir(exist_ok=True)
            for f in RUN_FILES:
                # copy2 保留修改时间, 它是运行时间的唯一记录
                shutil.copy2(run / f, target / f)
    fields_path.write_text(json.dumps({
        'description': ('两侧末轮场: 各场的统计指纹与 sha256, 及两侧比对, 按对照对分组. 原始 npy 未存档; '
                        'sha256 只能确认逐字节相同的文件, 重跑结果按容差与统计指纹比较'),
        'pairs': merged,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # results/ 下不属于任一对已固化运行的条目可能是旧运行的残留, 只提示, 不删除
    expected = {Path(side['directory']).name for record in merged.values() for side in record['fields'].values()}
    expected.add('final_fields.json')
    stale = sorted(p.name for p in results.iterdir() if p.name not in expected)
    if stale:
        print(f'[提示] {results} 下另有 {stale}, 不属于已固化的运行, 请确认是否删除', flush=True)

    for name, runs in present.items():
        comparison = records[name]['comparison']
        print(f'[完成] {name}: 已固化 {runs["a"].name} 与 {runs["b"].name} 到 {results}; '
              f'物理密度最大差 {comparison["physical_density_max_abs_diff"]["value"]:.3e}, '
              f'二值化{"相同" if comparison["binary_topology"]["identical"] else "不同"}'
              f' (不一致单元 {comparison["binary_topology"]["n_mismatch"]})', flush=True)
    for name in skipped:
        print(f'[跳过] {name}: A 侧运行目录不存在', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
