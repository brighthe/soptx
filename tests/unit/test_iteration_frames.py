"""``soptx.postprocess.iteration_frames`` 逐轮写帧与迁移的测试 (小型二维网格)."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from soptx.mesh import QuadrangleMesh
from soptx.postprocess.iteration_frames import IterationFrameWriter
from soptx.postprocess.vtk_export import read_vtu_cell_data


def _writer(tmp_path, n_frames=3, finished=True, name='run'):
    """在 ``tmp_path/wsl/<name>`` 下写 ``n_frames`` 帧, 第 k 帧密度全为 k/10."""
    output = tmp_path / 'wsl' / name
    output.mkdir(parents=True)
    (output / 'config.json').write_text('{"grid": [2, 1]}')
    mesh = QuadrangleMesh.from_box([0.0, 2.0, 0.0, 1.0], 2, 1)
    frames = IterationFrameWriter(output, mesh)
    for k in range(1, n_frames + 1):
        frames.write(k, cell_data={'density': np.full(2, k / 10)})
    if finished:
        (output / 'summary.json').write_text('{}')
    return frames


def _pvd_files(path):
    return [d.get('file') for d in ET.parse(path).getroot().findall('./Collection/DataSet')]


def test_write_publishes_complete_frames_and_pvd(tmp_path) -> None:
    frames = _writer(tmp_path, finished=False)
    output = frames.output_dir
    expected = [f'iterations/iter_{k:04d}.vtu' for k in range(1, 4)]
    assert _pvd_files(output / 'evolution.pvd') == expected
    for k, name in enumerate(expected, 1):
        assert np.array_equal(read_vtu_cell_data(output / name), np.full(2, k / 10))
    # 临时文件改名后不残留
    leftovers = [p.name for p in [*output.iterdir(), *frames.iteration_dir.iterdir()] if p.name.startswith('.')]
    assert leftovers == []


def test_write_returns_relative_path_and_requires_increasing_iteration(tmp_path) -> None:
    frames = _writer(tmp_path, n_frames=0, finished=False)
    assert frames.write(5, cell_data={'density': np.zeros(2)}) == 'iterations/iter_0005.vtu'
    for iteration in (5, 4):
        with pytest.raises(ValueError, match='严格递增'):
            frames.write(iteration, cell_data={'density': np.zeros(2)})


def test_publish_moves_frames_to_view_directory(tmp_path) -> None:
    frames = _writer(tmp_path)
    (tmp_path / 'win').mkdir()
    destination = frames.publish(tmp_path / 'win' / 'experiment')
    assert destination == tmp_path / 'win' / 'experiment' / 'run'
    assert len(_pvd_files(destination / 'evolution.pvd')) == 3
    assert np.array_equal(read_vtu_cell_data(destination / 'iterations' / 'iter_0003.vtu'), np.full(2, 0.3))
    output = frames.output_dir
    assert not (output / 'evolution.pvd').exists() and not (output / 'iterations').exists()
    assert (output / 'visualization_location.json').is_file()


def test_publish_skips_when_view_parent_is_missing(tmp_path, capsys) -> None:
    frames = _writer(tmp_path)
    assert frames.publish(tmp_path / 'missing' / 'experiment') is None
    assert '跳过迁移' in capsys.readouterr().out
    assert len(list(frames.iteration_dir.iterdir())) == 3


def test_publish_replaces_previous_view_of_same_run(tmp_path) -> None:
    view_root = tmp_path / 'win' / 'experiment'
    view_root.parent.mkdir()
    _writer(tmp_path, n_frames=3).publish(view_root)
    # 重跑: 同名运行目录清空重建, 帧数与内容都变
    shutil.rmtree(tmp_path / 'wsl' / 'run')
    second = _writer(tmp_path, n_frames=2)
    (second.output_dir / 'config.json').write_text('{"grid": [2, 1], "rerun": true}')
    destination = second.publish(view_root)
    assert destination is not None
    assert _pvd_files(destination / 'evolution.pvd') == ['iterations/iter_0001.vtu', 'iterations/iter_0002.vtu']
    assert sorted(p.name for p in (destination / 'iterations').iterdir()) == ['iter_0001.vtu', 'iter_0002.vtu']


def test_publish_refuses_to_clear_foreign_entries(tmp_path, capsys) -> None:
    frames = _writer(tmp_path)
    foreign = tmp_path / 'win' / 'experiment' / 'run' / 'notes.txt'
    foreign.parent.mkdir(parents=True)
    foreign.write_text('keep')
    assert frames.publish(tmp_path / 'win' / 'experiment') is None
    assert 'notes.txt' in capsys.readouterr().out
    assert foreign.read_text() == 'keep'
    assert len(list(frames.iteration_dir.iterdir())) == 3


def test_publish_failure_keeps_frames_and_does_not_raise(tmp_path, capsys) -> None:
    # 没有 summary.json 时 sync_visualization 拒绝迁移
    frames = _writer(tmp_path, finished=False)
    (tmp_path / 'win').mkdir()
    assert frames.publish(tmp_path / 'win' / 'experiment') is None
    assert '迁移查看副本失败' in capsys.readouterr().out
    assert (frames.output_dir / 'evolution.pvd').is_file()
    assert len(list(frames.iteration_dir.iterdir())) == 3
