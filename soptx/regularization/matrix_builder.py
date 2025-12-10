from typing import Tuple, Set, List
from math import ceil, sqrt

from fealpy.backend import backend_manager as bm
from fealpy.mesh import HomogeneousMesh
from fealpy.typing import TensorLike
from fealpy.sparse import COOTensor, CSRTensor
from soptx.utils import timer

class FilterMatrixBuilder:
    """负责构建拓扑优化中使用的稀疏权重矩阵 H"""
    def __init__(self, 
                mesh: HomogeneousMesh, 
                rmin: float, 
                density_location: str, 
                # integration_order: int = None,
                # interpolation_order: int = None
            ) -> None:
        if rmin <= 0:
            raise ValueError("Filter radius must be positive")
        
        self._mesh = mesh
        self._rmin = rmin
        self._density_location = density_location
        
        # self._integration_order = integration_order
        # self._interpolation_order = interpolation_order

        self._device = mesh.device

    def build(self) -> CSRTensor:
        """构建并返回权重矩阵 H"""
        if self._mesh.meshdata['mesh_type'] == 'uniform_quad':
            if self._density_location in ['element', 'element_multiresolution']:
                H = self._compute_weighted_matrix_2d(
                                            self._rmin,
                                            self._mesh.meshdata['nx'], self._mesh.meshdata['ny'],
                                            self._mesh.meshdata['hx'], self._mesh.meshdata['hy'], 
                                        )
                return H        

            elif self._density_location in ['node', 'node_multiresolution']:
                H = self._compute_weighted_matrix_general(
                                            rmin=self._rmin, 
                                            domain=self._mesh.meshdata['domain']
                                        )
                return H
        
        elif self._mesh.meshdata['mesh_type'] == 'uniform_hex':
            if self._density_location in ['element', 'element_multiresolution']:
                H = self._compute_weighted_matrix_3d(
                                            self._rmin,
                                            self._mesh.meshdata['nx'], self._mesh.meshdata['ny'], self._mesh.meshdata['nz'],
                                            self._mesh.meshdata['hx'], self._mesh.meshdata['hy'], self._mesh.meshdata['hz'], 
                                        )
                return H
            
            elif self._density_location in ['node', 'node_multiresolution']:
                H = self._compute_weighted_matrix_general(
                                                rmin=self._rmin,
                                                domain=self._mesh.meshdata['domain'], 
                                            )
                return H
            
        elif self._mesh.meshdata['mesh_type'] in ['uniform_aligned_tri', 'uniform_crisscross_tri', 'uniform_tet']:
            H = self._compute_weighted_matrix_general(
                                            rmin=self._rmin, 
                                            domain=self._mesh.meshdata['domain']
                                        )
            return H
        
    def _compute_weighted_matrix_general(self, 
                                        rmin: float,
                                        domain: List[float],
                                        periodic: List[bool]=[False, False, False],
                                        enable_timing: bool = False,
                                    ) -> Tuple[COOTensor, TensorLike]:
        """
        计算任意网格的过滤权重矩阵, 即使设备选取为 GPU, 该函数也会先将其转移到 CPU 进行计算

        SRTO - 设计变量 = 单元密度中心点 / 节点密度
        MRTO - 设计变量 = 子单元密度中心点 / 节点密度  - 要求设计变量网格 = 子单元密度网格
        
        Parameters:
        -----------
        rmin: 过滤半径
        domain: 计算域的边界, 
        periodic: 各方向是否周期性, 默认为 [False, False, False]
            
        Returns:
        --------
        H: 过滤矩阵
        Hs: 过滤矩阵行和
        """
        t = None
        if enable_timing:
            t = timer(f"Filter_general")
            next(t)

        if self._density_location in ['element']:
            
            density_mesh = self._mesh
            density_coords = density_mesh.entity_barycenter('cell') # (NC, GD)

        elif self._density_location in ['element_multiresolution']:

            sub_density_mesh = self._mesh
            density_coords = sub_density_mesh.entity_barycenter('cell') # (NC*n_sub, GD)

        elif self._density_location in ['node']:

            density_mesh = self._mesh
            density_coords = density_mesh.entity_barycenter('node') # (NN, GD)

        elif self._density_location in ['node_multiresolution']:

            sub_density_mesh = self._mesh
            density_coords = sub_density_mesh.entity_barycenter('node') # (NN*, GD)

        # 使用 KD-tree 查询临近点
        density_coords = bm.device_put(density_coords, 'cpu')        
        density_indices, neighbor_indices = bm.query_point(
                                                x=density_coords, y=density_coords, h=rmin, 
                                                box_size=domain, mask_self=False, periodic=periodic
                                            )
        
        if enable_timing:
            t.send('KD-tree 查询时间')

        # 自由度总数
        gdof = density_coords.shape[0]
        
        # 准备存储过滤器矩阵的数组
        # 预估非零元素的数量（包括对角线元素）
        max_nnz = len(density_indices) + gdof
        iH = bm.zeros(max_nnz, dtype=bm.int32)
        jH = bm.zeros(max_nnz, dtype=bm.int32)
        sH = bm.zeros(max_nnz, dtype=bm.float64)
        
        # 首先添加对角线元素
        for i in range(gdof):
            iH[i] = i
            jH[i] = i
            sH[i] = rmin  # 自身权重为 rmin（最大权重）
        
        # 当前非零元素计数
        nnz = gdof

        if enable_timing:
            t.send('对角线循环计算时间')
        
        # 填充其余非零元素 (邻居点)
        # TODO 耗时非常久, 需要修改
        for idx in range(len(density_indices)):
            i = density_indices[idx]
            j = neighbor_indices[idx]
            
            # 计算节点间的物理距离
            physical_dist = bm.sqrt(bm.sum((density_coords[i] - density_coords[j])**2))
            
            # 计算权重因子
            fac = rmin - physical_dist
            
            if fac > 0:
                iH[nnz] = i
                jH[nnz] = j
                sH[nnz] = fac
                nnz += 1

        if enable_timing:
            t.send('非对角线循环计算时间')
        
        # 创建稀疏矩阵
        H = COOTensor(
                indices=bm.astype(bm.stack((iH[:nnz], jH[:nnz]), axis=0), bm.int32),
                values=sH[:nnz],
                spshape=(gdof, gdof)
            )
        
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device='cpu')
        
        H = H.tocsr()
        H = H.device_put(self._device)
        Hs = bm.device_put(Hs, self._device)
        
        if enable_timing:
            t.send('稀疏矩阵构建时间')
            t.send(None)

        return H
        # return H, Hs
        

    def _compute_weighted_matrix_2d(self,
                                    rmin: float,
                                    nx: int, ny: int,
                                    hx: float, hy: float,
                                    enable_timing: bool = False,
                                ) -> CSRTensor:
        """
        计算四边形网格的过滤权重矩阵, 即使设备选取为 GPU, 该函数也会先将其转移到 CPU 进行计算

        SRTO - 设计变量 = 单元密度中心点
        MRTO - 设计变量 = 密度子单元中心点 - 要求设计变量网格 = 密度子单元网格

        Parameters:
        -----------
        rmin: 过滤半径
        nx, ny : 设计变量网格剖分数
        hx, hy : 设计变量网格单元大小 
            
        Returns:
        --------
        H: 过滤矩阵
        """
        # 单元中心坐标需要偏移半个网格
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
        
        #! PyTorch 后端对 COOTensor 的支持更好
        # Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device='cpu')
        
        H = H.tocsr()
        H = H.device_put(self._device)
        # Hs = bm.device_put(Hs, self._device)
        
        if enable_timing:
            t.send('矩阵构建')
            t.send(None)
        
        return H

    def _compute_weighted_matrix_2d_math(self,
                                rmin: float, 
                                nx: int, ny: int, 
                                hx: float, hy: float,
                                enable_timing: bool = False,
                            ) -> Tuple[COOTensor, TensorLike]:
        """计算 2D 滤波矩阵 - 数学原理的直接实现版本
            即使设备选取为 GPU, 该函数也会先将其转移到 CPU 进行计算"""

        NC = self.mesh.number_of_cells()
        expected_NC = nx * ny
        assert NC == expected_NC, (
            f"'_compute_filter_2d' is strictly for simple quadrangle grids. "
            f"Expected {expected_NC} cells (nx*ny), but found {NC}."
        )

        t = None
        if enable_timing:
            t = timer(f"Filter_2d_math")
            next(t)

        min_h = min(hx, hy)
        max_cells = ceil(rmin/min_h)
        nfilter = int(nx * ny * ((2 * (max_cells - 1) + 1) ** 2))
        
        iH = bm.zeros(nfilter, dtype=bm.int32, device='cpu')   
        jH = bm.zeros(nfilter, dtype=bm.int32, device='cpu')   
        sH = bm.zeros(nfilter, dtype=bm.float64, device='cpu') 
        cc = 0

        if enable_timing:
            t.send('预处理')

        for i in range(nx):
            for j in range(ny):
                # 单元的编号顺序: y->x 
                row = i * ny + j
                # 根据物理距离计算搜索范围
                kk1 = int(max(i - (ceil(rmin/hx) - 1), 0))
                kk2 = int(min(i + ceil(rmin/hx), nx))
                ll1 = int(max(j - (ceil(rmin/hy) - 1), 0))
                ll2 = int(min(j + ceil(rmin/hy), ny))
                
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        # 单元的编号顺序: y->x 
                        col = k * ny + l
                        # 计算实际物理距离
                        physical_dist = sqrt((i - k)**2 * hx**2 + (j - l)**2 * hy**2)
                        fac = rmin - physical_dist
                        if fac > 0:
                            # ! bm.set_at 的性能相比直接赋值更慢
                            # iH[cc] = row
                            # jH[cc] = col
                            # sH[cc] = max(0.0, fac)
                            iH = bm.set_at(iH, cc, row)
                            jH = bm.set_at(jH, cc, col)
                            sH = bm.set_at(sH, cc, max(0.0, fac))
                            cc += 1

        if enable_timing:
            t.send('计算距离和过滤矩阵')

        H = COOTensor(
                indices=bm.astype(bm.stack((iH[:cc], jH[:cc]), axis=0), bm.int32),
                values=sH[:cc],
                spshape=(nx * ny, nx * ny)
            )
        
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device='cpu')
        
        H = H.tocsr()
        H = H.device_put(self.device)
        Hs = bm.device_put(Hs, self.device)

        if enable_timing:
            t.send('矩阵构建')
            t.send(None)
        
        return H, Hs


    def _compute_weighted_matrix_3d(self,
                                    rmin: float, 
                                    nx: int, ny: int, nz: int, 
                                    hx: float, hy: float, hz: float,
                                    enable_timing: bool = False,
                                ) -> COOTensor:
        """
        计算六面体网格的过滤权重矩阵, 即使设备选取为 GPU, 该函数也会先将其转移到 CPU 进行计算

        SRTO - 设计变量 = 单元密度中心点
        MRTO - 设计变量 = 密度子单元中心点 - 要求设计变量网格 = 密度子单元网格

        Parameters:
        -----------
        rmin: 过滤半径
        nx, ny, nz : 设计变量网格剖分数
        hx, hy, hz : 设计变量网格单元大小 
            
        Returns:
        --------
        H: 过滤矩阵
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
        
        #! PyTorch 后端对 COOTensor 的支持更好
        # Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device='cpu')
        
        H = H.tocsr()
        H = H.device_put(self._device)
        # Hs = bm.device_put(Hs, self._device)

        if enable_timing:
            t.send('矩阵构建')
            t.send(None)
        
        return H
    

    def _compute_filter_3d_math(self,
                                rmin: float, 
                                nx: int, ny: int, nz: int, 
                                hx: float, hy: float, hz: float,
                                enable_timing: bool = False,
                            ) -> Tuple[COOTensor, TensorLike]:
        """计算 3D 滤波矩阵 - 数学原理的直接实现版本"""

        t = None
        if enable_timing:
            t = timer(f"Filter_3d_math")
            next(t)

        min_h = min(hx, hy, hz)
        max_cells = ceil(rmin/min_h)
        nfilter = nx * ny * nz * ((2 * (max_cells - 1) + 1) ** 3)
        
        iH = bm.zeros(nfilter, dtype=bm.int32, device='cpu')
        jH = bm.zeros(nfilter, dtype=bm.int32, device='cpu')
        sH = bm.zeros(nfilter, dtype=bm.float64, device='cpu')
        cc = 0

        if enable_timing:
            t.send('预处理')

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 单元的编号顺序: z -> y -> x
                    row = k + j * nz + i * ny * nz
                    ii1 = int(max(i - (ceil(rmin/hx) - 1), 0))
                    ii2 = int(min(i + ceil(rmin/hx), nx))
                    jj1 = int(max(j - (ceil(rmin/hy) - 1), 0))
                    jj2 = int(min(j + ceil(rmin/hy), ny))
                    kk1 = int(max(k - (ceil(rmin/hz) - 1), 0))
                    kk2 = int(min(k + ceil(rmin/hz), nz))
                    
                    for ii in range(ii1, ii2):
                        for jj in range(jj1, jj2):
                            for kk in range(kk1, kk2):
                                # 单元的编号顺序: z -> y -> x
                                col = kk + jj * nz + ii * ny * nz
                                # 计算实际物理距离 
                                physical_dist = sqrt(
                                                    (i - ii)**2 * hx**2 + 
                                                    (j - jj)**2 * hy**2 + 
                                                    (k - kk)**2 * hz**2
                                                )
                                fac = rmin - physical_dist
                                if fac > 0:
                                    # ! bm.set_at 的性能相比直接赋值更慢
                                    # iH[cc] = row
                                    # jH[cc] = col
                                    # sH[cc] = max(0.0, fac)
                                    iH = bm.set_at(iH, cc, row)
                                    jH = bm.set_at(jH, cc, col)
                                    sH = bm.set_at(sH, cc, max(0.0, fac))
                                    cc += 1

        if enable_timing:
            t.send('计算距离和过滤因子')

        H = COOTensor(
            indices=bm.astype(bm.stack((iH[:cc], jH[:cc]), axis=0), bm.int32),
            values=sH[:cc],
            spshape=(nx * ny * nz, nx * ny * nz)
        )
        H = H.tocsr()
        H = H.device_put(self.device)
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device=self.device)

        if enable_timing:
            t.send('矩阵构建')
            t.send(None)

        return H, Hs