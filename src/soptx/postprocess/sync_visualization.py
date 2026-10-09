"""把优化过程的 ParaView 帧增量同步 (或迁移) 到查看目录, 不改动求解流程.

计算在 WSL 中进行, 结果目录 (``config.json``, ``history.json``, ``*.npy`` 等) 是计算与核查的
依据; 每轮帧 ``iterations/*.vtu`` 与 ``evolution.pvd`` 体积大, 跨文件系统读取慢, 因此复制到
Windows 本地目录 (WSL 下为 ``/mnt/c/...``) 供 ParaView 查看. 只处理 ``evolution.pvd`` 已发布的帧;
源端须以临时文件替换的方式发布完整的 VTU 与 PVD.

原为 ``experiments/topopt_simp_substructure_fa`` 目录内的脚本, 去掉写死的默认路径并增加迁移模式后
并入本模块, 供各实验共用.

用法::

    python -m soptx.postprocess.sync_visualization --source-dir <结果目录> --destination-dir <查看目录>
    python -m soptx.postprocess.sync_visualization ... --move     # 运行结束后迁移并删除源帧
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

_MANIFEST = '.visualization_sync.json'
_LOCK = '.visualization_sync.lock'
_LOCATION = 'visualization_location.json'


def _signature(path: Path) -> List[int]:
    """读取大小与纳秒修改时间, 用于增量判断及复制期间的变化检查."""
    value = path.stat()
    return [value.st_size, value.st_mtime_ns]


def _digest(path: Path) -> str:
    """分块计算文件 SHA-256, 不把完整 VTU 读入内存."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _publish(path: Path, data: bytes) -> None:
    """在目标目录写临时文件后替换发布, 失败时只清理自己的临时文件."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.sync-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _windows_path(path: Path) -> str:
    """WSL 的 ``/mnt/<盘符>/...`` 转成 Windows 路径, 其余原样返回."""
    parts = path.parts
    if len(parts) >= 3 and parts[:2] == ('/', 'mnt') and len(parts[2]) == 1:
        return f'{parts[2].upper()}:\\' + '\\'.join(parts[3:])
    return str(path)


def _published_frames(source: Path, destination: Path) -> tuple[bytes, List[str]]:
    """读取源 PVD, 返回其字节与已发布帧的相对路径; 只接受 ``iterations/*.vtu``."""
    pvd_bytes = (source / 'evolution.pvd').read_bytes()
    root = ET.fromstring(pvd_bytes)
    if root.tag != 'VTKFile' or root.get('type') != 'Collection':
        raise ValueError('evolution.pvd 不是 PVD Collection')
    datasets = root.findall('./Collection/DataSet')
    if not datasets:
        raise ValueError('PVD 尚未发布任何帧, 请稍后重试')
    paths: List[str] = []
    for dataset in datasets:
        relative = PurePosixPath(dataset.get('file', ''))
        if len(relative.parts) != 2 or relative.parts[0] != 'iterations' or relative.suffix != '.vtu':
            raise ValueError(f'仅支持 iterations/*.vtu 相对路径: {relative}')
        if ':' in str(relative) or '\\' in str(relative):
            raise ValueError('PVD 路径含不支持的字符')
        src, dst = source / str(relative), destination / str(relative)
        if not src.resolve().is_relative_to(source) or not dst.resolve().is_relative_to(destination):
            raise ValueError('帧路径越出源或目标目录')
        if str(relative) not in paths:
            paths.append(str(relative))
    return pvd_bytes, paths


def sync_visualization(source_dir: Path, destination_dir: Path, *, dry_run: bool = False,
                       move: bool = False) -> Dict[str, Any]:
    """先同步完整帧, 最后发布 PVD; 单次执行, 不持续监控.

    Parameters
    ----------
    source_dir : 计算结果目录, 含 ``config.json``, ``evolution.pvd`` 与 ``iterations/``.
    destination_dir : 查看目录; 不能与源目录相同或互相包含.
    dry_run : 只统计待处理的帧与数据量, 不写目标.
    move : 同步后逐帧重新核对源与目标的 SHA-256, 一致才删除源帧与源 PVD, 并在源目录写
        ``visualization_location.json`` 记录迁移位置. 要求源目录已有 ``summary.json``
        (运行已结束), 否则拒绝迁移.

    Returns
    -------
    report : ``dict(frames, copied, unchanged, bytes, moved)``.

    Raises
    ------
    ValueError
        目录关系、PVD 或帧格式不符, 目标绑定了其他来源或配置, 或迁移时运行尚未结束.
    FileExistsError
        目标中同名帧内容不同, 或已有同步进程持有锁.
    RuntimeError
        复制或核对期间源帧、源配置发生变化, 或迁移前的核对不一致.

    Notes
    -----
    增量清单 ``.visualization_sync.json`` 记录来源、配置摘要、各帧大小、修改时间与 SHA-256;
    未变化的帧按大小与修改时间跳过. 同名帧内容不同或来源配置变化时拒绝覆盖, 应为不同运行
    指定不同的目标目录. 异常退出遗留 ``.visualization_sync.lock`` 时, 先确认没有同步进程再手动
    删除.
    """
    source = Path(source_dir).resolve()
    destination = Path(destination_dir).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('源与目标目录不能相同或互相包含')
    if move and not (source / 'summary.json').is_file():
        raise ValueError('源目录没有 summary.json, 运行可能尚未结束, 拒绝迁移')
    pvd_bytes, paths = _published_frames(source, destination)

    # 配置摘要用于区分工况; 同工况重跑的帧变化另由文件摘要检查
    config_hash = _digest(source / 'config.json')
    manifest_path = destination / _MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest and (manifest.get('source') != str(source) or manifest.get('config_sha256') != config_hash):
        raise ValueError('目标绑定了其他来源或配置, 请为本次运行指定新目录')
    previous = manifest.get('frames', {})
    plan = []
    records: Dict[str, Dict[str, Any]] = {}
    for relative in paths:
        src, dst = source / relative, destination / relative
        source_stat = _signature(src)
        # 轻量完整性检查, 不解析含原始二进制的整个 XML
        with src.open('rb') as stream:
            header = stream.read(256)
            stream.seek(max(0, source_stat[0] - 128))
            tail = stream.read()
        if b'UnstructuredGrid' not in header or b'</VTKFile>' not in tail:
            raise ValueError(f'帧尚未完整发布或格式不符: {src}')
        old = previous.get(relative, {})
        if dst.is_file() and old.get('source_stat') == source_stat and old.get('destination_stat') == _signature(dst):
            records[relative] = old
            continue
        plan.append((relative, source_stat))
    total_bytes = sum(stat[0] for _, stat in plan)
    print(f'[同步] {len(paths)} 帧, 已确认未变 {len(records)}, 待复制或核对 {len(plan)}, '
          f'数据量 {total_bytes / 1e9:.2f} GB', flush=True)
    print(f'[源] {source}\n[目标] {destination}', flush=True)
    report = dict(frames=len(paths), copied=len(plan), unchanged=len(records), bytes=total_bytes, moved=False)
    if dry_run:
        return report

    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / _LOCK
    # 防止两个同步进程同时发布目标索引
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        for number, (relative, before) in enumerate(plan, 1):
            src, dst = source / relative, destination / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                sha = _digest(src)
                if _digest(dst) != sha:
                    raise FileExistsError(f'同名帧内容不同, 未覆盖: {dst}. 请指定新目标目录')
                if _signature(src) != before:
                    raise RuntimeError('核对期间源帧变化, 请重试')
            else:
                with tempfile.NamedTemporaryFile(dir=dst.parent, prefix='.sync-', delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    shutil.copyfile(src, temporary)
                    sha = _digest(temporary)
                    if _signature(src) != before or _digest(src) != sha:
                        raise RuntimeError('复制期间源帧变化或内容校验失败, 未发布 PVD')
                    if dst.exists():
                        raise FileExistsError(f'目标帧在复制期间出现: {dst}')
                    temporary.replace(dst)
                finally:
                    temporary.unlink(missing_ok=True)
            records[relative] = dict(source_stat=before, destination_stat=_signature(dst), sha256=sha)
            print(f'[帧 {number}/{len(plan)}] {relative}', flush=True)
        # 源目录发生重跑或覆盖时不发布新索引; 新追加的帧留待下次同步
        if _digest(source / 'config.json') != config_hash:
            raise RuntimeError('同步期间源配置变化, 未发布 PVD')
        for relative in paths:
            if _signature(source / relative) != records[relative]['source_stat']:
                raise RuntimeError('同步期间已发布帧发生变化, 未发布 PVD')
            if _signature(destination / relative) != records[relative]['destination_stat']:
                raise RuntimeError('同步期间目标帧发生变化, 未发布 PVD')
        result = dict(source=str(source), config_sha256=config_hash, frames=records)
        _publish(manifest_path, json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8'))
        _publish(destination / 'evolution.pvd', pvd_bytes)
        print(f'[完成] ParaView 打开 {_windows_path(destination / "evolution.pvd")}', flush=True)

        if move:
            _move_source(source, destination, paths, records, pvd_bytes)
            report['moved'] = True
    finally:
        lock.unlink(missing_ok=True)
    return report


def _move_source(source: Path, destination: Path, paths: List[str], records: Dict[str, Dict[str, Any]],
                 pvd_bytes: bytes) -> None:
    """逐帧重新核对源与目标的 SHA-256, 全部一致后删除源帧与源 PVD, 并记录迁移位置."""
    total = 0
    for number, relative in enumerate(paths, 1):
        sha = records[relative]['sha256']
        if _digest(source / relative) != sha or _digest(destination / relative) != sha:
            raise RuntimeError(f'迁移前核对不一致, 未删除任何源帧: {relative}')
        total += (source / relative).stat().st_size
        print(f'[核对 {number}/{len(paths)}] {relative}', flush=True)
    if (destination / 'evolution.pvd').read_bytes() != pvd_bytes:
        raise RuntimeError('目标 PVD 与源不一致, 未删除任何源帧')
    location = dict(destination=_windows_path(destination), pvd='evolution.pvd', frames=len(paths), bytes=total,
                    verification='每帧源与目标 SHA-256 一致, PVD 逐字节一致',
                    moved_at_utc=datetime.now(timezone.utc).isoformat())
    _publish(source / _LOCATION, json.dumps(location, ensure_ascii=False, indent=2).encode('utf-8'))
    for relative in paths:
        (source / relative).unlink()
    (source / 'evolution.pvd').unlink()
    iteration_dir = source / 'iterations'
    if iteration_dir.is_dir() and not any(iteration_dir.iterdir()):
        iteration_dir.rmdir()
    print(f'[迁移] 已删除源帧 {len(paths)} 个 ({total / 1e9:.2f} GB), 位置记录于 {source / _LOCATION}', flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口, 见模块说明."""
    parser = argparse.ArgumentParser(description='把 ParaView 帧增量同步或迁移到查看目录')
    parser.add_argument('--source-dir', type=Path, required=True, help='计算结果目录 (含 evolution.pvd)')
    parser.add_argument('--destination-dir', type=Path, required=True,
                        help='查看目录; WSL 下 Windows 本地盘写作 /mnt/c/...')
    parser.add_argument('--dry-run', action='store_true', help='只列出待处理帧与数据量, 不写目标')
    parser.add_argument('--move', action='store_true',
                        help='同步并核对后删除源帧与源 PVD, 在源目录记录迁移位置; 要求运行已结束')
    args = parser.parse_args(argv)
    sync_visualization(args.source_dir, args.destination_dir, dry_run=args.dry_run, move=args.move)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
