from typing import Any, Dict, List, Optional
from math import ceil

from fealpy.backend import backend_manager as bm
from fealpy.mesh import HomogeneousMesh
from fealpy.sparse import COOTensor
from soptx.core import BaseLogged, timer

class FilterMatrixBuilder(BaseLogged):
    """负责构建拓扑优化中使用的稀疏权重矩阵 H

    权重函数有两套实现:

    - 结构化快路径 (``_compute_weighted_matrix_2d/3d``): 线性锥形权重
      ``max(0, rmin - d)``, 只对均匀笛卡尔网格成立 (单元按 ``i*ny + j`` 的
      字典序编号);
    - 通用路径 (``_compute_weighted_matrix_general``): KD-tree 近邻查询 +
      ``(1 - d/rmin)**q`` 权重, 对任意网格成立。

    两条路径在 ``q = 1`` 时权重函数一致 (相差常数因子 rmin, 被行归一化
    ``H / Hs`` 约掉); ``q > 1`` 是 PolyFilter (Giraldo-Londono & Paulino,
    2020) 的非线性权重, 与结构化路径不可比, 故 q 必须由调用方显式给定。
    """
    def __init__(self,
                mesh: HomogeneousMesh,
                rmin: float,
                density_location: str,
                q: int = 1,
                enable_logging: bool = False,
                logger_name: Optional[str] = None,
            ) -> None:
        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        if rmin <= 0:
            raise ValueError("Filter radius must be positive")
        if q < 1:
            raise ValueError(f"过滤权重幂次 q 必须为正整数, 当前 q={q}")

        self._mesh = mesh
        self._rmin = rmin
        self._density_location = density_location
        self._q = q

        self._device = mesh.device

    def build(self) -> COOTensor:
        """构建并返回权重矩阵 H

        按 **网格能否提供结构化元数据** 分派, 而不是按 ``mesh_type`` 字符串:
        只有当密度定义在单元上、``meshdata`` 同时给出 ``nx/ny(/nz)`` 与
        ``hx/hy(/hz)``、且单元数恰好等于 ``nx*ny(*nz)`` (设计变量确实排在一张
        均匀笛卡尔网格上) 时, 才走结构化快路径; 其余一律走 KD-tree 通用路径,
        后者对任意非结构网格 (gmsh 三角/四面体网格等) 同样成立。

        ``meshdata`` 不是 fealpy 网格的固有属性, 而是各 experiment 的 pipeline
        手工挂上去的元数据字典, 因此这里一律用 ``get`` 探测: 缺键时安静退回通
        用路径, 不再直接 KeyError。
        """
        meshdata: Dict[str, Any] = dict(getattr(self._mesh, 'meshdata', None) or {})
        NC = self._mesh.number_of_cells()
        cell_centered = self._density_location in ('element', 'element_multiresolution')

        nx, ny, nz = meshdata.get('nx'), meshdata.get('ny'), meshdata.get('nz')
        hx, hy, hz = meshdata.get('hx'), meshdata.get('hy'), meshdata.get('hz')

        if cell_centered and None not in (nx, ny, hx, hy):
            if nx * ny == NC and nz in (None, 1):
                return self._compute_weighted_matrix_2d(self._rmin, nx, ny, hx, hy)

            if None not in (nz, hz) and nx * ny * nz == NC:
                return self._compute_weighted_matrix_3d(self._rmin, nx, ny, nz, hx, hy, hz)

        return self._compute_weighted_matrix_general(
                                    rmin=self._rmin,
                                    domain=self._bounding_box(meshdata),
                                    q=self._q,
                                )

    def _bounding_box(self, meshdata: Dict[str, Any]) -> List[float]:
        """计算域包围盒 ``[xmin, xmax, ymin, ymax, ...]``

        ``meshdata['domain']`` 缺失时 (非结构网格的常态) 由节点坐标现算。该值
        只在 ``bm.query_point`` 打开周期性时才会被用到, 而通用路径固定
        ``periodic=[False, False, False]``, 因此它当前对结果没有影响。
        """
        domain = meshdata.get('domain')
        if domain is not None:
            return [float(v) for v in domain]

        node = bm.device_put(self._mesh.entity('node'), 'cpu')
        box: List[float] = []
        for d in range(node.shape[1]):
            box.append(float(bm.min(node[:, d])))
            box.append(float(bm.max(node[:, d])))

        return box
        
    def _compute_weighted_matrix_general(self, 
                                        rmin: float,
                                        domain: List[float],
                                        q: int = 1,
                                        periodic: List[bool]=[False, False, False],
                                        enable_timing: bool = False,
                                    ) -> COOTensor:
            """
            计算任意网格的过滤权重矩阵, 即使设备选取为 GPU, 该函数也会先将其转移到 CPU 进行计算

            支持线性过滤 (q=1) 和非线性过滤 (q>1):
                - 线性过滤: w_ij = max(0, rmin - dist_ij)
                - 非线性过滤: w_ij = (1 - dist_ij / rmin)^q, dist_ij <= rmin

            非线性过滤参考:
                PolyFilter.m from PolyStress (Giraldo-Londoño & Paulino, 2020)

            Parameters
            ----------
            rmin: 过滤半径
            domain: 计算域的边界
            q: 过滤权重的幂次参数, 默认为 1 (线性过滤).
                当 q=1 时, 权重函数为 w = max(0, 1 - d/rmin), 等价于线性锥形过滤.
                当 q>1 时, 权重函数为 w = (1 - d/rmin)^q, 提供更集中的过滤效果,
                与结构化快路径的线性锥形权重不可比, 故不设非 1 的默认值.
            periodic: 各方向是否周期性, 默认为 [False, False, False]
                
            Returns
            -------
            H: 过滤矩阵 (行归一化)
            """
            t = None
            if enable_timing:
                t = timer(f"Filter_general")
                next(t)

            if self._density_location in ['element']:
                density_mesh = self._mesh
                density_coords = density_mesh.entity_barycenter('cell')

            elif self._density_location in ['element_multiresolution']:
                sub_density_mesh = self._mesh
                density_coords = sub_density_mesh.entity_barycenter('cell')

            elif self._density_location in ['node']:
                density_mesh = self._mesh
                density_coords = density_mesh.entity_barycenter('node')

            else:
                self._log_error(f"Unsupported density location for general filter: {self._density_location}")

            # 使用 KD-tree 查询邻近点
            density_coords = bm.device_put(density_coords, 'cpu')        
            density_indices, neighbor_indices = bm.query_point(
                                                    x=density_coords, y=density_coords, h=rmin, 
                                                    box_size=domain, mask_self=False, periodic=periodic
                                                )
            
            if enable_timing:
                t.send('KD-tree 查询时间')

            # 自由度总数
            gdof = density_coords.shape[0]

            # 对角线元素 (自身距离为 0, 权重为 1.0^q = 1.0), 向量化赋值
            diag_indices = bm.arange(gdof, dtype=bm.int32)

            if enable_timing:
                t.send('对角线向量化计算时间')

            # 批量计算所有邻居对的物理距离
            diffs = density_coords[density_indices] - density_coords[neighbor_indices]
            dists = bm.sqrt(bm.sum(diffs**2, axis=1))

            # 筛选距离严格小于 rmin 的邻居对, 计算非线性权重
            mask = dists < rmin
            valid_i = density_indices[mask]
            valid_j = neighbor_indices[mask]
            valid_w = (1.0 - dists[mask] / rmin) ** q

            if enable_timing:
                t.send('非对角线向量化计算时间')

            # 拼接对角线元素与邻居权重, 构建稀疏矩阵
            all_i = bm.concatenate([diag_indices, valid_i])
            all_j = bm.concatenate([diag_indices, valid_j])
            all_s = bm.concatenate([bm.ones(gdof, dtype=bm.float64), valid_w])

            H = COOTensor(
                    indices=bm.astype(bm.stack((all_i, all_j), axis=0), bm.int32),
                    values=all_s,
                    spshape=(gdof, gdof)
                )

            if enable_timing:
                t.send('稀疏矩阵构建时间')
                t.send(None)

            return H

    def _compute_weighted_matrix_2d(self,
                                    rmin: float,
                                    nx: int, ny: int,
                                    hx: float, hy: float,
                                    enable_timing: bool = False,
                                ) -> COOTensor:
        """
        计算四边形网格的过滤权重矩阵, 即使设备选取为 GPU, 该函数也会先将其转移到 CPU 进行计算

        SRTO - 设计变量 = 单元密度中心点
        MRTO - 设计变量 = 密度子单元中心点 - 要求设计变量网格 = 密度子单元网格

        Parameters
        ----------
        rmin: 过滤半径 (物理长度尺度), 与 hx, hy 同单位
        nx, ny : 设计变量网格剖分数
        hx, hy : 设计变量网格单元大小 
            
        Returns
        -------
        H: 过滤矩阵
        """
        # 单元中心坐标偏移量
        coord_offset_x = 0.5 * hx
        coord_offset_y = 0.5 * hy
    
        # 总自由度数
        N_total = nx * ny  
        
        t = None
        if enable_timing:
            t = timer(f"Filter_2d_{self._density_location}")
            next(t)
        
        search_radius_x = ceil(rmin/hx)
        search_radius_y = ceil(rmin/hy)
        
        # 批处理单元, 避免一次处理所有单元耗尽内存
        batch_size = min(10000, N_total) 
        n_batches = (N_total + batch_size - 1) // batch_size 
        
        # 创建一个映射函数，从线性索引转换为 2D 坐标
        def linear_to_2d(linear_idx):
            i = linear_idx // ny
            j = linear_idx % ny
            return i, j
        
        # 预计算所有自由度的物理坐标
        all_coords = bm.zeros((N_total, 2), dtype=bm.float64, device='cpu')
        
        for idx in range(N_total):
            i, j = linear_to_2d(idx)
            # 使用偏移量来区分单元中心和节点坐标
            all_coords[idx, 0] = i * hx + coord_offset_x
            all_coords[idx, 1] = j * hy + coord_offset_y
        
        if enable_timing:
            t.send('预处理')
        
        # 初始化存储结果的列表
        all_rows = [] 
        all_cols = []  
        all_vals = []  
        
        # 分批处理所有单元
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, N_total)
            
            batch_rows = []
            batch_cols = []
            batch_vals = []
            
            # 获取当前批次单元的物理坐标
            batch_coords = all_coords[start_idx:end_idx]
            
            # 处理当前批次中的每个单元
            for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
                i, j = linear_to_2d(global_idx)
                row = global_idx
                
                # 计算搜索范围 - 与原始函数完全相同
                ii1 = max(0, i - (search_radius_x - 1))
                ii2 = min(nx, i + search_radius_x)
                jj1 = max(0, j - (search_radius_y - 1))
                jj2 = min(ny, j + search_radius_y)
                
                # 创建搜索范围内所有自由度的线性索引
                search_indices = []
                for ii in range(ii1, ii2):
                    for jj in range(jj1, jj2):
                        col = ii * ny + jj
                        search_indices.append(col)
                
                if not search_indices:
                    continue
                
                # 获取搜索范围内的坐标
                search_coords = all_coords[search_indices]
                
                # 计算与当前单元的距离
                current_coords = batch_coords[local_idx].reshape(1, 2)
                diffs = search_coords - current_coords
                squared_dists = bm.sum(diffs * diffs, axis=1)
                distances = bm.sqrt(squared_dists)
                
                # 计算过滤因子
                factors = rmin - distances
                valid_mask = factors > 0
                
                # 只保留有效的单元对
                if bm.any(valid_mask):
                    valid_cols = bm.array(search_indices, device='cpu')[valid_mask]
                    
                    # 线性权重
                    valid_factors = factors[valid_mask]
                    
                    # 收集结果
                    batch_rows.extend([row] * len(valid_cols))
                    batch_cols.extend(valid_cols.tolist())
                    batch_vals.extend(valid_factors.tolist())
            
            # 将当前批次结果添加到总结果
            all_rows.extend(batch_rows)
            all_cols.extend(batch_cols)
            all_vals.extend(batch_vals)
        
        if enable_timing:
            t.send('计算距离和过滤矩阵')
        
        # 构建稀疏矩阵
        if all_rows:
            iH = bm.tensor(all_rows, dtype=bm.int32, device='cpu')
            jH = bm.tensor(all_cols, dtype=bm.int32, device='cpu')
            sH = bm.tensor(all_vals, dtype=bm.float64, device='cpu')
        else:
            iH = bm.tensor([], dtype=bm.int32, device='cpu')
            jH = bm.tensor([], dtype=bm.int32, device='cpu')
            sH = bm.tensor([], dtype=bm.float64, device='cpu')
        
        H = COOTensor(
                    indices=bm.stack((iH, jH), axis=0),
                    values=sH,
                    spshape=(N_total, N_total)
                )
        
        if enable_timing:
            t.send('矩阵构建')
            t.send(None)
        
        return H
    

    def _compute_weighted_matrix_3d(self,
                                    rmin: float, 
                                    nx: int, ny: int, nz: int, 
                                    hx: float, hy: float, hz: float,
                                    enable_timing: bool = False,
                                ) -> COOTensor:
        """计算六面体网格的过滤权重矩阵.

        Parameters
        ----------
        rmin : 过滤半径.
        nx, ny, nz : 设计变量网格在三个方向的剖分数.
        hx, hy, hz : 设计变量网格单元在三个方向的尺寸.
        enable_timing : 是否输出各阶段细分计时.

        Returns
        -------
        H : 过滤矩阵.

        Notes
        -----
        即使设备选为 GPU, 本函数也会先把数据转到 CPU 上计算.

        设计变量的取法随分辨率策略而异: SRTO 取单元密度中心点; MRTO 取密度子单元中心
        点, 因此要求设计变量网格与密度子单元网格一致.
        """
        t = None
        if enable_timing:
            t = timer(f"Filter_3d_{self._density_location}")
            next(t)
        
        search_radius_x = ceil(rmin/hx)
        search_radius_y = ceil(rmin/hy)
        search_radius_z = ceil(rmin/hz)
        
        # 批处理单元, 避免一次处理所有单元耗尽内存
        batch_size = min(10000, nx * ny * nz)  
        n_batches = (nx * ny * nz + batch_size - 1) // batch_size
        
        # 创建一个映射函数，从线性索引转换为 3D 坐标
        def linear_to_3d(linear_idx):
            i = linear_idx // (ny * nz)
            j = (linear_idx % (ny * nz)) // nz
            k = linear_idx % nz
            return i, j, k
        
        # 预计算每个格子的物理坐标 
        all_coords = bm.zeros((nx * ny * nz, 3), dtype=bm.float64, device='cpu')
        
        for idx in range(nx * ny * nz):
            i, j, k = linear_to_3d(idx)
            all_coords[idx, 0] = i * hx
            all_coords[idx, 1] = j * hy
            all_coords[idx, 2] = k * hz
                
        if enable_timing:
            t.send('预处理')

        # 初始化存储结果的列表
        all_rows = []
        all_cols = []
        all_vals = []
        
        # 分批处理所有单元
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, nx * ny * nz)
            
            batch_rows = []
            batch_cols = []
            batch_vals = []
            
            # 获取当前批次单元的坐标
            batch_coords = all_coords[start_idx:end_idx]
            
            # 处理当前批次中的每个单元
            for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
                i, j, k = linear_to_3d(global_idx)
                row = global_idx
                
                # 计算搜索范围 - 与原始函数完全相同
                ii1 = max(0, i - (search_radius_x - 1))
                ii2 = min(nx, i + search_radius_x)
                jj1 = max(0, j - (search_radius_y - 1))
                jj2 = min(ny, j + search_radius_y)
                kk1 = max(0, k - (search_radius_z - 1))
                kk2 = min(nz, k + search_radius_z)
                
                # 创建搜索范围内所有单元的线性索引
                search_indices = []
                for ii in range(ii1, ii2):
                    for jj in range(jj1, jj2):
                        for kk in range(kk1, kk2):
                            col = kk + jj * nz + ii * ny * nz
                            search_indices.append(col)
                
                if not search_indices:
                    continue
                    
                # 获取搜索单元的物理坐标
                search_coords = all_coords[search_indices]
                
                # 计算与当前单元的距离
                current_coords = batch_coords[local_idx].reshape(1, 3) 
                diffs = search_coords - current_coords  
                squared_dists = bm.sum(diffs * diffs, axis=1) 
                distances = bm.sqrt(squared_dists) 
                
                # 计算滤波因子
                factors = rmin - distances 
                valid_mask = factors > 0  
                
                if bm.any(valid_mask):
                    valid_cols = bm.array(search_indices, device='cpu')[valid_mask]
                    valid_factors = factors[valid_mask]
                    
                    # 收集结果
                    batch_rows.extend([row] * len(valid_cols))
                    batch_cols.extend(valid_cols.tolist())
                    batch_vals.extend(valid_factors.tolist())
            
            # 添加批次结果到总结果
            all_rows.extend(batch_rows)
            all_cols.extend(batch_cols)
            all_vals.extend(batch_vals)
                
        if enable_timing:
            t.send('计算距离和过滤因子')

        if all_rows:
            iH = bm.tensor(all_rows, dtype=bm.int32, device='cpu')
            jH = bm.tensor(all_cols, dtype=bm.int32, device='cpu')
            sH = bm.tensor(all_vals, dtype=bm.float64, device='cpu')
        else:
            iH = bm.tensor([], dtype=bm.int32, device='cpu')
            jH = bm.tensor([], dtype=bm.int32, device='cpu')
            sH = bm.tensor([], dtype=bm.float64, device='cpu')
        
        H = COOTensor(
                    indices=bm.stack((iH, jH), axis=0),
                    values=sH,
                    spshape=(nx * ny * nz, nx * ny * nz)
                )

        if enable_timing:
            t.send('矩阵构建')
            t.send(None)
        
        return H
