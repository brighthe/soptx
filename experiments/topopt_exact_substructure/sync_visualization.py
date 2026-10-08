"""手动增量同步 ParaView 可视化副本, 不删除已有结果或改动求解流程.

仅处理 evolution.pvd 已发布的 iterations/*.vtu. 源端应采用临时文件替换发布
完整 VTU 和 PVD, 不应原地覆盖正在读取的文件. 同名帧内容不同时拒绝覆盖.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import xml.etree.ElementTree as ET


def parse_args(argv=None):
    """解析源目录, Windows 查看目录和只读预览选项."""
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    default_target = (Path("C:/workspace/soptx-results/topopt_exact_substructure/linear_corner_mumps")
                      if os.name == "nt" else
                      Path("/mnt/c/workspace/soptx-results/topopt_exact_substructure/linear_corner_mumps"))
    parser.add_argument("--source-dir", type=Path, default=base / "outputs" / "linear_corner_mumps",
                        help="计算结果目录, 默认取脚本旁的 outputs/linear_corner_mumps")
    parser.add_argument("--destination-dir", type=Path, default=default_target,
                        help="Windows 本地可视化目录; WSL 下使用 /mnt/c/... 路径")
    parser.add_argument("--dry-run", action="store_true", help="只列出待处理帧与大小, 不写入目标")
    return parser.parse_args(argv)


def signature(path):
    """读取大小与纳秒修改时间, 用于增量判断及复制期间变化检查."""
    value = path.stat()
    return [value.st_size, value.st_mtime_ns]


def digest(path):
    """分块计算文件 SHA-256, 不将完整 VTU 载入内存."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def publish(path, data):
    """在目标目录写临时文件后替换发布, 失败时只清理自己的临时文件."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".sync-", delete=False) as stream:
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


def main(argv=None):
    """先同步完整帧, 最后发布 PVD; 默认单次执行, 不持续监控."""
    args = parse_args(argv)
    source = args.source_dir.resolve()
    destination = args.destination_dir.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("源与目标目录不能相同或互相包含")
    if os.name != "nt" and args.destination_dir == parse_args([]).destination_dir:
        if not Path("/mnt/c/Windows").is_dir():
            raise ValueError("未发现 WSL 的 C 盘挂载, 请显式指定 --destination-dir")
    pvd_bytes = (source / "evolution.pvd").read_bytes()
    root = ET.fromstring(pvd_bytes)
    if root.tag != "VTKFile" or root.get("type") != "Collection":
        raise ValueError("输入不是 PVD Collection")
    datasets = root.findall("./Collection/DataSet")
    if not datasets:
        raise ValueError("PVD 尚未发布任何帧, 请稍后重试")
    paths = []
    for dataset in datasets:
        relative = PurePosixPath(dataset.get("file", ""))
        if len(relative.parts) != 2 or relative.parts[0] != "iterations" or relative.suffix != ".vtu":
            raise ValueError(f"仅支持 iterations/*.vtu 相对路径: {relative}")
        if ":" in str(relative) or "\\" in str(relative):
            raise ValueError("PVD 路径含不支持的字符")
        src, dst = source / str(relative), destination / str(relative)
        if not src.resolve().is_relative_to(source) or not dst.resolve().is_relative_to(destination):
            raise ValueError("帧路径越出源或目标目录")
        if str(relative) not in paths:
            paths.append(str(relative))
    # 配置摘要用于区分工况; 同工况重跑的帧变化另由文件摘要检查.
    config_hash = digest(source / "config.json")
    manifest_path = destination / ".visualization_sync.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest and (manifest.get("source") != str(source) or manifest.get("config_sha256") != config_hash):
        raise ValueError("目标绑定了其他来源或配置, 请为本次运行指定新目录")
    previous = manifest.get("frames", {})
    plan = []
    records = {}
    for relative in paths:
        src, dst = source / relative, destination / relative
        source_stat = signature(src)
        # 轻量完整性检查, 不解析含原始二进制的整个 XML.
        with src.open("rb") as stream:
            header = stream.read(256)
            stream.seek(max(0, source_stat[0]-128))
            tail = stream.read()
        if b'UnstructuredGrid' not in header or b'</VTKFile>' not in tail:
            raise ValueError(f"帧尚未完整发布或格式不符: {src}")
        old = previous.get(relative, {})
        if (dst.is_file() and old.get("source_stat") == source_stat
                and old.get("destination_stat") == signature(dst)):
            records[relative] = old
            continue
        plan.append((relative, source_stat))
    total_bytes = sum(stat[0] for _,stat in plan)
    print(f"[同步] {len(paths)} 帧, 已确认未变 {len(records)}, 待复制或核对 {len(plan)}, 数据量 {total_bytes/1e9:.2f} GB", flush=True)
    print(f"[源] {source}\n[目标] {destination}", flush=True)
    if args.dry_run:
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / ".visualization_sync.lock"
    # 防止两个同步进程同时发布目标索引; 异常退出遗留锁需人工确认后清理.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        for number,(relative,before) in enumerate(plan,1):
            src, dst = source / relative, destination / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                sha = digest(src)
                if digest(dst) != sha:
                    raise FileExistsError(f"同名帧内容不同, 未覆盖: {dst}. 请指定新目标目录")
                if signature(src) != before:
                    raise RuntimeError("核对期间源帧变化, 请重试")
            else:
                with tempfile.NamedTemporaryFile(dir=dst.parent,prefix=".sync-",delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    shutil.copyfile(src, temporary)
                    sha = digest(temporary)
                    if signature(src) != before or digest(src) != sha or signature(src) != before:
                        raise RuntimeError("复制期间源帧变化或内容校验失败, 未发布 PVD")
                    if dst.exists():
                        raise FileExistsError(f"目标帧在复制期间出现: {dst}")
                    temporary.replace(dst)
                finally:
                    temporary.unlink(missing_ok=True)
            records[relative] = dict(source_stat=before,destination_stat=signature(dst),sha256=sha)
            print(f"[帧 {number}/{len(plan)}] {relative}", flush=True)
        # 源目录发生重跑/覆盖时不发布新索引. 新追加帧留待下次同步.
        if digest(source / "config.json") != config_hash:
            raise RuntimeError("同步期间源配置变化, 未发布 PVD")
        for relative in paths:
            if signature(source / relative) != records[relative]["source_stat"]:
                raise RuntimeError("同步期间已发布帧发生变化, 未发布 PVD")
            if signature(destination / relative) != records[relative]["destination_stat"]:
                raise RuntimeError("同步期间目标帧发生变化, 未发布 PVD")
        result = dict(source=str(source),config_sha256=config_hash,frames=records)
        publish(manifest_path,json.dumps(result,ensure_ascii=False,indent=2).encode("utf-8"))
        publish(destination / "evolution.pvd",pvd_bytes)
        print(f"[完成] ParaView 打开 {destination / 'evolution.pvd'}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
