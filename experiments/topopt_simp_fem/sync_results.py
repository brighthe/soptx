"""把选定的 EA 与 FA 运行从 outputs/ 固化到入库的 results/.

``outputs/`` 是运行脚本的临时产物, 不入库; ``results/`` 是 ``fem_ea_vs_fa_topopt_results.md`` 的数据来源,
入库. 本脚本做两件事:

1. 把两侧运行目录下的 ``config.json``, ``summary.json``, ``history.json`` 原样拷入
   ``results/<运行目录名>/``;
2. 读两侧末轮的 ``*_final.npy``, 写出 ``results/final_fields.json``: 各场的统计指纹与 sha256,
   及两侧的逐场比对. npy 本身不入库.

只依赖 numpy 与标准库, 不导入 soptx.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_FILES = ('config.json', 'summary.json', 'history.json')
FIELD_FILES = ('density_final.npy', 'design_density_final.npy', 'displacement_final.npy')
# 两侧 config.json 中允许不同的键: 只差算子层级及由它决定的方法名与 FA 的边界条件施加方式
CONFIG_KEYS_ALLOWED_TO_DIFFER = {'method', 'operator_level', 'fa_dirichlet'}
# 二值化阈值
SOLID_THRESHOLD = 0.5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='把选定的 EA 与 FA 运行从 outputs/ 固化到 results/')
    parser.add_argument('--ea', type=Path, default=EXPERIMENT_DIR / 'outputs' / 'ea_hex_210x35x35_cg_jacobi',
                        help='EA 运行目录 (默认: outputs/ea_hex_210x35x35_cg_jacobi)')
    parser.add_argument('--fa', type=Path, default=EXPERIMENT_DIR / 'outputs' / 'fa_hex_210x35x35_cg_jacobi',
                        help='FA 运行目录 (默认: outputs/fa_hex_210x35x35_cg_jacobi)')
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
    """相对实验目录的路径, 不在其下时为绝对路径; 写入 json 以标明数据来自哪次运行."""
    try:
        return path.relative_to(EXPERIMENT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def check_runs(runs: Dict[str, Path]) -> None:
    """检查两侧运行齐全且可比.

    Parameters
    ----------
    runs : 侧名 ('ea', 'fa') 到运行目录的映射.

    Raises
    ------
    FileNotFoundError
        缺少所需文件; 缺 summary.json 说明运行未正常结束.
    ValueError
        两侧 config.json 除算子层级外还有其他差异, 或算子层级与侧名不符.
    """
    for side, run in runs.items():
        missing = [name for name in RUN_FILES + FIELD_FILES if not (run / name).is_file()]
        if missing:
            raise FileNotFoundError(f'{side} 运行目录 {run} 缺少 {missing}')
    configs = {side: json.loads((run / 'config.json').read_text(encoding='utf-8')) for side, run in runs.items()}
    for side, config in configs.items():
        if config.get('operator_level') != side:
            raise ValueError(f'{runs[side]} 的 operator_level 为 {config.get("operator_level")!r}, 应为 {side!r}')
    keys = set(configs['ea']) | set(configs['fa'])
    differing = sorted(key for key in keys - CONFIG_KEYS_ALLOWED_TO_DIFFER
                       if configs['ea'].get(key) != configs['fa'].get(key))
    if differing:
        raise ValueError(f'两侧 config.json 除算子层级外还有差异: {differing}')


def final_fields(runs: Dict[str, Path]) -> Dict[str, Any]:
    """两侧末轮场的统计指纹, sha256 与逐场比对.

    Parameters
    ----------
    runs : 侧名 ('ea', 'fa') 到运行目录的映射.

    Returns
    -------
    record : 写入 final_fields.json 的内容.
    """
    data = {side: {name: np.load(run / name) for name in FIELD_FILES} for side, run in runs.items()}
    fields = {}
    for side, run in runs.items():
        fields[side] = {'directory': display_path(run)}
        for name in FIELD_FILES:
            entry = {'sha256': sha256(run / name), **field_stats(data[side][name])}
            if name == 'density_final.npy':
                entry['n_solid'] = int(np.count_nonzero(data[side][name] > SOLID_THRESHOLD))
            fields[side][name] = entry

    rho_ea, rho_fa = data['ea']['density_final.npy'], data['fa']['density_final.npy']
    x_ea, x_fa = data['ea']['design_density_final.npy'], data['fa']['design_density_final.npy']
    u_ea, u_fa = data['ea']['displacement_final.npy'], data['fa']['displacement_final.npy']
    solid_ea, solid_fa = rho_ea > SOLID_THRESHOLD, rho_fa > SOLID_THRESHOLD
    # 离阈值最近的单元与阈值之距; 远大于物理密度差时, 二值化结果相同不是巧合
    distance = min(np.min(np.abs(rho_ea - SOLID_THRESHOLD)), np.min(np.abs(rho_fa - SOLID_THRESHOLD)))

    return {
        'description': ('EA 与 FA 末轮场: 各场的统计指纹与 sha256, 及两侧比对. 原始 npy 未存档; '
                        'sha256 只能确认逐字节相同的文件, 重跑结果按容差与统计指纹比较'),
        'fields': fields,
        'comparison': {
            'physical_density_max_abs_diff': {
                'definition': 'max |rho_EA - rho_FA|, density_final.npy',
                'value': float(np.max(np.abs(rho_ea - rho_fa)))},
            'design_density_max_abs_diff': {
                'definition': 'max |x_EA - x_FA|, design_density_final.npy',
                'value': float(np.max(np.abs(x_ea - x_fa)))},
            'displacement_relative_l2_diff': {
                'definition': '||u_EA - u_FA||_2 / ||u_EA||_2, displacement_final.npy',
                'value': float(np.linalg.norm(u_ea - u_fa) / np.linalg.norm(u_ea))},
            'binary_topology': {
                'definition': 'density_final.npy 取 rho > 0.5 后逐单元比较',
                'identical': bool(np.array_equal(solid_ea, solid_fa)),
                'n_mismatch': int(np.count_nonzero(solid_ea != solid_fa)),
                'min_distance_to_threshold': float(distance)},
        },
    }


def main(argv=None):
    args = parse_args(argv)
    runs = {'ea': args.ea.resolve(), 'fa': args.fa.resolve()}
    results = args.results_dir.resolve()
    check_runs(runs)

    # 先算完比对再写盘, 中途出错不留下半新半旧的 results/
    record = final_fields(runs)

    results.mkdir(parents=True, exist_ok=True)
    for run in runs.values():
        target = results / run.name
        target.mkdir(exist_ok=True)
        for name in RUN_FILES:
            # copy2 保留修改时间, 它是运行时间的唯一记录
            shutil.copy2(run / name, target / name)
    (results / 'final_fields.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n',
                                               encoding='utf-8')

    # results/ 下不属于本次两侧运行的条目可能是旧运行的残留, 只提示, 不删除
    expected = {run.name for run in runs.values()} | {'final_fields.json'}
    stale = sorted(path.name for path in results.iterdir() if path.name not in expected)
    if stale:
        print(f'[提示] {results} 下另有 {stale}, 不属于本次固化的运行, 请确认是否删除', flush=True)

    comparison = record['comparison']
    print(f'[完成] 已固化 {runs["ea"].name} 与 {runs["fa"].name} 到 {results}; '
          f'物理密度最大差 {comparison["physical_density_max_abs_diff"]["value"]:.3e}, '
          f'二值化{"相同" if comparison["binary_topology"]["identical"] else "不同"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
