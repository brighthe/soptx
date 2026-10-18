"""ALM-MMA 实验的逐轮结果输出, 不参与优化计算."""

from pathlib import Path
import json

import numpy as np

from soptx.backend import backend_manager as bm
from soptx.postprocess.iteration_frames import IterationFrameWriter
from soptx.postprocess.vtk_export import write_vtu


class ALMIterationOutput:
    """保存与已接受设计对齐的场和历史.

    Parameters
    ----------
    output : Path
        已创建的独立运行目录.
    mesh : Mesh
        用于 VTU 输出的分析网格.
    fields : sequence of str
        逐轮 VTU 中需要保存的单元场名称.

    Notes
    -----
    每个 JSON 文件单独通过临时文件替换, 不保证多个文件同时提交.
    输出失败直接终止运行, 已完成文件保留. 这些文件不构成续算检查点.
    """

    def __init__(self, output, mesh, fields):
        self.output = Path(output)
        self.mesh = mesh
        self.fields = tuple(fields)
        self.frames = IterationFrameWriter(self.output, mesh)
        self.outer_history = []

    def _write_json(self, name, payload):
        """序列化后通过同目录临时文件替换目标文件."""
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        target = self.output / name
        temporary = target.with_name('.' + target.name + '.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(target)

    def __call__(self, event, history, record):
        """消费优化器已完成的记录, 不修改优化状态.

        Parameters
        ----------
        event : str
            initial, iteration 或 outer.
        history : OptimizationHistory
            优化器已记录的密度、标量与应力场.
        record : dict or None
            外层更新诊断, 其他事件为 None.
        """
        if event == 'initial':
            cells = {'density': np.asarray(
                bm.to_numpy(history.initial_physical_density[:])).reshape(-1)}
            self.frames.write(0, cell_data=cells)
            write_vtu(mesh=self.mesh, filepath=str(self.output / 'density_initial'),
                      cell_data=cells)
            self._write_json('outer_history.json', self.outer_history)
        elif event == 'iteration':
            cells = {'density': np.asarray(
                bm.to_numpy(history.physical_densities[-1][:])).reshape(-1)}
            stress = np.asarray(bm.to_numpy(history.field_histories['von_mises_stress'][-1]))
            cells['von_mises_normalized'] = stress.reshape(len(stress), -1).max(axis=1)
            self.frames.write(history.iter_indices[-1], cell_data={
                name: cells[name] for name in self.fields})
        elif event == 'outer':
            self.outer_history.append(dict(record))
            self._write_json('outer_history.json', self.outer_history)
            return
        else:
            raise ValueError(f'未知输出事件: {event}')
        self._write_json('history.json', {
            'iter_indices': history.iter_indices,
            'changes': history.changes,
            'iteration_times': history.iteration_times,
            'scalar_histories': history.scalar_histories,
        })
