"""``soptx.postprocess.sync_visualization`` 增量同步与迁移的测试 (小型伪造帧)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import pytest

from soptx.postprocess.sync_visualization import main, sync_visualization


def _frame(tag: str) -> bytes:
    """足以通过头尾完整性检查的伪 VTU."""
    return (b'<?xml version="1.0"?><VTKFile type="UnstructuredGrid">' + tag.encode() * 64
            + b'</VTKFile>\n')


def _run(tmp_path, n_frames=3, finished=True):
    source = tmp_path / 'wsl' / 'run'
    (source / 'iterations').mkdir(parents=True)
    (source / 'config.json').write_text('{"grid": [6, 1, 1]}')
    root = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
    collection = ET.SubElement(root, 'Collection')
    for k in range(1, n_frames + 1):
        (source / 'iterations' / f'iter_{k:04d}.vtu').write_bytes(_frame(f'frame{k}'))
        ET.SubElement(collection, 'DataSet', timestep=str(k), group='', part='0',
                      file=f'iterations/iter_{k:04d}.vtu')
    ET.ElementTree(root).write(source / 'evolution.pvd', encoding='utf-8', xml_declaration=True)
    if finished:
        (source / 'summary.json').write_text('{}')
    return source, tmp_path / 'win' / 'run'


def test_sync_copies_frames_and_publishes_pvd_then_is_incremental(tmp_path) -> None:
    source, destination = _run(tmp_path)
    report = sync_visualization(source, destination)
    assert report['copied'] == 3 and not report['moved']
    for k in range(1, 4):
        name = f'iterations/iter_{k:04d}.vtu'
        assert (destination / name).read_bytes() == (source / name).read_bytes()
    assert (destination / 'evolution.pvd').read_bytes() == (source / 'evolution.pvd').read_bytes()
    assert not (destination / '.visualization_sync.lock').exists()
    # 源保持不变; 再同步一次时全部按清单跳过
    assert (source / 'iterations' / 'iter_0001.vtu').exists()
    again = sync_visualization(source, destination)
    assert again['copied'] == 0 and again['unchanged'] == 3


def test_dry_run_writes_nothing(tmp_path) -> None:
    source, destination = _run(tmp_path)
    report = sync_visualization(source, destination, dry_run=True)
    assert report['copied'] == 3 and not destination.exists()


def test_conflicting_frame_is_not_overwritten(tmp_path) -> None:
    source, destination = _run(tmp_path)
    (destination / 'iterations').mkdir(parents=True)
    (destination / 'iterations' / 'iter_0002.vtu').write_bytes(_frame('other'))
    with pytest.raises(FileExistsError, match='同名帧内容不同'):
        sync_visualization(source, destination)
    assert not (destination / 'evolution.pvd').exists()


def test_destination_bound_to_other_config_is_rejected(tmp_path) -> None:
    source, destination = _run(tmp_path)
    sync_visualization(source, destination)
    (source / 'config.json').write_text('{"grid": [12, 2, 2]}')
    with pytest.raises(ValueError, match='其他来源或配置'):
        sync_visualization(source, destination)


def test_move_verifies_then_removes_source_frames(tmp_path) -> None:
    source, destination = _run(tmp_path)
    report = sync_visualization(source, destination, move=True)
    assert report['moved']
    assert not (source / 'iterations').exists() and not (source / 'evolution.pvd').exists()
    assert (source / 'config.json').exists() and (source / 'summary.json').exists()
    location = json.loads((source / 'visualization_location.json').read_text())
    assert location['frames'] == 3 and location['pvd'] == 'evolution.pvd'
    assert sorted(p.name for p in (destination / 'iterations').iterdir()) == [
        'iter_0001.vtu', 'iter_0002.vtu', 'iter_0003.vtu']


def test_move_refuses_unfinished_run(tmp_path) -> None:
    source, destination = _run(tmp_path, finished=False)
    with pytest.raises(ValueError, match='summary.json'):
        sync_visualization(source, destination, move=True)
    assert (source / 'iterations' / 'iter_0001.vtu').exists() and not destination.exists()


def test_nested_directories_are_rejected(tmp_path) -> None:
    source, _ = _run(tmp_path)
    with pytest.raises(ValueError, match='互相包含'):
        sync_visualization(source, source / 'copy')


def test_command_line_entry(tmp_path) -> None:
    source, destination = _run(tmp_path)
    assert main(['--source-dir', str(source), '--destination-dir', str(destination), '--move']) == 0
    assert (destination / 'evolution.pvd').exists() and not (source / 'evolution.pvd').exists()


def test_windows_path_conversion() -> None:
    from pathlib import Path
    from soptx.postprocess.sync_visualization import _windows_path
    assert _windows_path(Path('/mnt/c/workspace/soptx-results/run')) == 'C:\\workspace\\soptx-results\\run'
    assert _windows_path(Path('/home/user/run')) == '/home/user/run'
