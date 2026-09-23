"""论文固定稳定化系数入口, 复用现有优化与后处理流程.

分析器原生默认固定系数; 本入口只管理论文结果目录与后处理.
历史默认 outputs 不覆盖; --output-root 显式选择固定系数结果集.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import config


def activate(root: Path) -> None:
    """在导入驱动前设置论文结果目录, 系数采用分析器原生默认值."""
    config.bootstrap_source_path()
    config.OUTPUT_DIR = root
    config.FIGURE_DIR = root / 'figures'


def refresh(root: Path, baseline: Path) -> None:
    """复用未受影响的运行, 重算交叉表与低阶应力场, 更新五幅论文图."""
    beam = 'compliance-fixed-fixed-half'
    stress = 'cantilever-middle-2d-stress'
    bearings = ('bearing-compressible', 'bearing-incompressible')
    stress_tag = 'analyzer-huzhang__lfem_constraint-apparent__load_pad_radius-1.5__order-2__solid_thr-0.5'
    required = [(beam, 'analyzer-huzhang__order-2'),
                *((c, 'analyzer-huzhang__order-2') for c in bearings),
                (stress, stress_tag)]
    sources = []
    for case, tag in required:
        run_dir = root / case / tag
        summary = json.loads((run_dir / 'summary.json').read_text())
        if not summary.get('converged'):
            raise RuntimeError(f'运行未收敛: {run_dir}')
        sources.append(str(run_dir))

    # 未改变的优化结果只建立只读用途链接, 后处理文件写入新的独立目录.
    unchanged = [(beam, f'analyzer-{m}__order-{k}')
                 for m in ('lfem', 'huzhang') for k in (2, 3, 4)
                 if (m, k) != ('huzhang', 2)]
    unchanged += [(c, f'analyzer-lfem__order-{k}') for c in bearings for k in (1, 2)]
    for case, tag in unchanged:
        src, dst = baseline / case / tag, root / case / tag
        if not (src / 'summary.json').is_file():
            raise FileNotFoundError(src)
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)
        sources.append(str(src))
    probe_rel = Path(stress) / 'postprocess/discretization_probe'
    (root / probe_rel).mkdir(parents=True, exist_ok=True)
    for method, order in [('lfem', 2), ('lfem', 3), ('lfem', 4), ('huzhang', 3), ('huzhang', 4)]:
        tag = f'{method}-{order}-pad-solid'
        for suffix in ('__fields.npz', '__probe.json', f'__{method}-{order}.vtu'):
            src, dst = baseline / probe_rel / (tag+suffix), root / probe_rel / (tag+suffix)
            if not src.is_file():
                raise FileNotFoundError(src)
            if not dst.exists():
                dst.symlink_to(src)
            sources.append(str(src))

    import provenance
    import compliance_reanalysis as cr
    import bearing_reanalysis as br
    mode = {'stabilization_coefficient_mode': 'fixed',
            'density_shear_ratio': None, 'algorithm_changes': 'none'}
    stamp = provenance.run_stamp()
    block = cr.cross_table()
    if not all(v['passed'] for v in block['self_check'].values()):
        raise RuntimeError('固支梁再分析对角线自检失败')
    cr.write_json({'case_id': beam, 'cross': block, 'provenance': stamp, **mode})
    cr.print_cross_markdown(block)
    for case in bearings:
        block = br.cross_table(case)
        br.write_json(case, {'case_id': case, 'cross': block, 'provenance': stamp, **mode})
        br.print_cross_markdown(case, block)

    import discretization_probe as dp
    rc = dp.run_discretization_probe(['--design', stress_tag])
    if rc:
        raise RuntimeError('应力再分析自检失败, 不发布图件')
    probe_file = root / probe_rel / 'huzhang-2-pad-solid__probe.json'
    data = json.loads(probe_file.read_text())
    data.update(mode)
    probe_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

    for name in ('compliance_topology', 'compliance_convergence', 'bearing_topologies',
                 'stress_hz_orders_topologies', 'stress_traction_jump'):
        importlib.import_module('plots.'+name).main()

    files = list(config.EXPERIMENT_DIR.glob('*.py')) + list((config.EXPERIMENT_DIR/'plots').glob('*.py'))
    files += [config.CASES_FILE,
              config.SOURCE_DIR/'soptx/fem/analyzers/huzhang_mfem_analyzer.py',
              config.SOURCE_DIR/'soptx/fem/integrators/jump_penalty_integrator.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    report = {**mode, 'provenance': stamp, 'inputs': sources, 'source_sha256': hashes,
              'note': 'Historical inputs retained. No analytical sensitivity validation implied.'}
    (root/'paper_refresh_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('[done] fixed-coefficient paper refresh', flush=True)


def main() -> int:
    """转发优化/绘图参数, 或执行论文同步后处理."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('command', choices=('run', 'compare', 'refresh'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.output_root.resolve()
    baseline = config.OUTPUT_DIR
    if root == baseline.resolve():
        parser.error('固定结果必须放在独立目录, 不覆盖历史 outputs')
    activate(root)
    if args.command == 'refresh':
        if args.arguments:
            parser.error('refresh 不接受额外参数')
        refresh(root, baseline)
        return 0
    forwarded = args.arguments
    if args.command == 'run':
        if '--output' in forwarded:
            parser.error('请用 --output-root 指定输出目录')
        forwarded = [*forwarded, '--output', str(root)]
    sys.argv = [args.command+'.py', *forwarded]
    return importlib.import_module(args.command).main() or 0


if __name__ == '__main__':
    raise SystemExit(main())