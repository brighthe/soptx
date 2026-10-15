"""逐轮写出拓扑优化的 ParaView 帧, 运行结束后迁移到查看目录.

计算在 WSL 中进行, 每轮帧先写在运行目录的 ``iterations/iter_XXXX.vtu``, 由同级的
``evolution.pvd`` 串成时间序列; 写本地盘快, 不把跨文件系统的写入计入逐轮耗时. 运行结束后
经 ``sync_visualization`` 迁移到 Windows 本地查看目录 (WSL 下为 ``/mnt/c/...``), 运行目录只
保留 json, npy 等核查依据.

用法::

    frames = IterationFrameWriter(output, mesh)
    for iteration in ...:
        frame_file = frames.write(iteration, cell_data={'density': rho})
    ...  # 写出 summary.json
    frames.publish(VIEW_ROOT)
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET

from soptx.postprocess.sync_visualization import sync_visualization
from soptx.postprocess.vtk_export import write_vtu

# 查看目录中允许由本工具清空的条目; 出现其他条目时拒绝清空, 以免误删他人文件
_VIEW_ENTRIES = frozenset({'iterations', 'evolution.pvd', '.visualization_sync.json'})


class IterationFrameWriter:
    """逐轮写出 ParaView 帧, 运行结束后迁移到查看目录.

    帧写在运行目录的 ``iterations/iter_XXXX.vtu``, 由同级的 ``evolution.pvd`` 串成时间序列.
    帧与 PVD 都先写临时文件再改名, 运行中断不会留下半个文件, 已写出的帧可用
    ``python -m soptx.postprocess.sync_visualization --move`` 补迁.

    Parameters
    ----------
    output_dir : 运行目录, 须已存在; 构造时在其下创建 ``iterations/``.
    mesh : soptx.mesh 网格, 传给 ``write_vtu``.
    """

    def __init__(self, output_dir: Path, mesh: Any) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.mesh = mesh
        self.iteration_dir = self.output_dir / 'iterations'
        self.iteration_dir.mkdir()
        self._root = ET.Element('VTKFile', type='Collection', version='0.1', byte_order='LittleEndian')
        self._collection = ET.SubElement(self._root, 'Collection')
        self._last_iteration: Optional[int] = None

    def write(self, iteration: int, cell_data: Optional[Dict[str, Any]] = None,
              point_data: Optional[Dict[str, Any]] = None) -> str:
        """写出第 ``iteration`` 轮的帧并更新 ``evolution.pvd``.

        Parameters
        ----------
        iteration : 迭代步号, 用作帧文件名与 PVD 时间步; 须严格递增.
        cell_data : 单元场字典, 可为 None.
        point_data : 节点场字典, 可为 None.

        Returns
        -------
        frame_file : 帧相对运行目录的路径, 如 ``'iterations/iter_0001.vtu'``.

        Raises
        ------
        ValueError
            ``iteration`` 不大于上一次写出的步号.
        """
        if self._last_iteration is not None and iteration <= self._last_iteration:
            raise ValueError(f'迭代步号须严格递增: 上一帧 {self._last_iteration}, 本帧 {iteration}')
        frame = self.iteration_dir / f'iter_{iteration:04d}.vtu'
        # write_vtu 自动补 .vtu 扩展名
        temporary_base = self.iteration_dir / f'.iter_{iteration:04d}.tmp'
        temporary_vtu = Path(str(temporary_base) + '.vtu')
        try:
            write_vtu(mesh=self.mesh, filepath=str(temporary_base),
                      cell_data=cell_data or None, point_data=point_data or None)
            temporary_vtu.replace(frame)
        finally:
            temporary_vtu.unlink(missing_ok=True)
        self._last_iteration = iteration

        # VTU 写完再更新 PVD, PVD 只引用完整的帧
        frame_file = frame.relative_to(self.output_dir).as_posix()
        ET.SubElement(self._collection, 'DataSet', timestep=str(iteration), group='', part='0', file=frame_file)
        temporary_pvd = self.output_dir / '.evolution.pvd.tmp'
        try:
            ET.ElementTree(self._root).write(temporary_pvd, encoding='utf-8', xml_declaration=True)
            temporary_pvd.replace(self.output_dir / 'evolution.pvd')
        finally:
            temporary_pvd.unlink(missing_ok=True)
        return frame_file

    def publish(self, view_root: Path) -> Optional[Path]:
        """把帧与 PVD 迁移到 ``view_root / <运行目录名>``, 须在 ``summary.json`` 写出之后调用.

        Parameters
        ----------
        view_root : 实验级查看根目录, 如 ``/mnt/c/workspace/soptx-results/topopt_simp_fem``.

        Returns
        -------
        destination : 迁移后的查看目录; 跳过或失败时为 None, 帧留在运行目录.

        Notes
        -----
        - ``view_root.parent`` 不存在 (如不在 WSL 或未挂载 C 盘) 时跳过并打印提示.
        - 目标目录已存在时先清空, 与运行目录重跑即清空的规则一致; 但目标中出现
          ``iterations/``, ``evolution.pvd``, ``.visualization_sync.json`` 以外的条目时
          拒绝清空, 打印警告并跳过, 以免误删他人文件.
        - 迁移调用 ``sync_visualization(move=True)``; 任何异常只打印警告不上抛, 运行结果
          已完整写出, 不应因查看副本失败而报错.
        """
        view_root = Path(view_root)
        destination = view_root / self.output_dir.name
        if not view_root.parent.is_dir():
            print(f'[查看副本] {view_root.parent} 不存在, 跳过迁移, 帧留在 {self.output_dir}', flush=True)
            return None
        try:
            if destination.exists():
                unknown = sorted(p.name for p in destination.iterdir() if p.name not in _VIEW_ENTRIES)
                if unknown:
                    print(f'[警告] 查看目录 {destination} 含非本工具条目 {unknown}, 拒绝清空, '
                          f'跳过迁移, 帧留在 {self.output_dir}', flush=True)
                    return None
                shutil.rmtree(destination)
            sync_visualization(self.output_dir, destination, move=True)
        except Exception as error:
            print(f'[警告] 迁移查看副本失败, 帧留在 {self.output_dir}: {error!r}', flush=True)
            return None
        return destination
