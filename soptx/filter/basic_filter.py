from abc import ABC, abstractmethod
from typing import Optional, Literal, Tuple, List
from math import ceil, sqrt

from fealpy.backend import backend_manager as bm
from fealpy.mesh import StructuredMesh, UniformMesh2d, UniformMesh3d
from fealpy.typing import TensorLike
from fealpy.sparse import COOTensor, CSRTensor

from soptx.utils import timer

class BasicFilter(ABC):
    """基础滤波器抽象基类"""
    
    def __init__(self, mesh: StructuredMesh, rmin: float, domain: List):
        """
        Parameters
        - mesh : 网格
        - rmin : 滤波半径 (物理距离)
        - domain : 计算域的边界
        """
        if rmin <= 0:
            raise ValueError("Filter radius must be positive")
        
        self.mesh = mesh
        self.rmin = rmin
        self.domain = domain
        self.device = mesh.device
        
        self._H, self._Hs = self._compute_filter_matrix()
        self._cell_measure = self.mesh.entity_measure('cell')
        self._normalize_factor = self._H.matmul(self._cell_measure)

    @property
    def H(self) -> COOTensor:
        """滤波矩阵"""
        return self._H

    @property
    def Hs(self) -> TensorLike:
        """滤波矩阵行和向量"""
        return self._Hs
    
    def _compute_filter_matrix(self) -> Tuple[COOTensor, TensorLike]:
        """计算线性衰减的滤波器内核"""
        if isinstance(self.mesh, UniformMesh2d):
            return self._compute_filter_2d(
                                self.mesh.nx, self.mesh.ny,
                                self.mesh.h[0], self.mesh.h[1], 
                                self.rmin
                            )
        elif isinstance(self.mesh, UniformMesh3d):
            return self._compute_filter_3d(
                                self.mesh.nx, self.mesh.ny, self.mesh.nz,
                                self.mesh.h[0], self.mesh.h[1], self.mesh.h[2],
                                self.rmin
                            )
        else:
            return self._compute_filter_general(
                                self.mesh.entity_barycenter('cell'), 
                                self.rmin, 
                                domain=self.domain,
                            )
        
    def _compute_filter_general(self, 
                cell_centers: TensorLike,
                rmin: float,
                domain=None,
                periodic=[False, False, False]
            ) -> Tuple[COOTensor, TensorLike]:
        """计算任意网格的滤波矩阵
        
        Parameters:
        - cell_centers: 单元中心点坐标, 形状为 (NC, GD)
        - rmin: 滤波半径
        - domain: 计算域的边界, 
            例如 [xmin, xmax, ymin, ymax] 或 [xmin, xmax, ymin, ymax, zmin, zmax]
        - periodic: 各方向是否周期性, 默认为 [False, False, False]
            
        Returns:
        - H: 滤波矩阵, 形状为 (NC, NC)
        - Hs: 滤波矩阵行和, 形状为 (NC, )
        """
        t = None
        t = timer(f"General Timing")
        next(t)
        # 使用 KD-tree 查询临近点
        cell_indices, neighbor_indices = bm.query_point(
                                            x=cell_centers, y=cell_centers, h=rmin, 
                                            box_size=domain, mask_self=False, periodic=periodic
                                        )
        t.send('KD-tree 查询时间')
        # 计算节点总数
        NC = cell_centers.shape[0]
        
        # 准备存储过滤器矩阵的数组
        # 预估非零元素的数量（包括对角线元素）
        max_nnz = len(cell_indices) + NC
        iH = bm.zeros(max_nnz, dtype=bm.int32)
        jH = bm.zeros(max_nnz, dtype=bm.int32)
        sH = bm.zeros(max_nnz, dtype=bm.float64)
        
        # 首先添加对角线元素（自身单元）
        for i in range(NC):
            iH[i] = i
            jH[i] = i
            # 自身权重为 rmin（最大权重）
            sH[i] = rmin
        
        # 当前非零元素计数
        nnz = NC
        
        # 填充其余非零元素（邻居点）
        for idx in range(len(cell_indices)):
            i = cell_indices[idx]
            j = neighbor_indices[idx]
            
            # 计算节点间的物理距离
            physical_dist = bm.sqrt(bm.sum((cell_centers[i] - cell_centers[j])**2))
            
            # 计算权重因子
            fac = rmin - physical_dist
            
            if fac > 0:
                iH[nnz] = i
                jH[nnz] = j
                sH[nnz] = fac
                nnz += 1
        t.send('循环计算时间')
        # 创建稀疏矩阵（只使用有效的非零元素）
        H = COOTensor(
            indices=bm.astype(bm.stack((iH[:nnz], jH[:nnz]), axis=0), bm.int32),
            values=sH[:nnz],
            spshape=(NC, NC)
        )
        
        # 转换为 CSR 格式以便于后续操作
        H = H.tocsr()
        
        # 计算滤波矩阵行和
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64)
        t.send('稀疏矩阵构建时间')
        t.send(None)
        return H, Hs

    def _compute_filter_2d_math(self, 
                nx: int, ny: int, 
                hx: float, hy: float,
                rmin: float
            ) -> Tuple[COOTensor, TensorLike]:
        """计算 2D 滤波矩阵 - 数学原理的直接实现版本
        
        计算在 CPU 上进行以获得更好的性能
        """
        min_h = min(hx, hy)
        max_cells = ceil(rmin/min_h)
        nfilter = int(nx * ny * ((2 * (max_cells - 1) + 1) ** 2))
        
        iH = bm.zeros(nfilter, dtype=bm.int32, device='cpu')   
        jH = bm.zeros(nfilter, dtype=bm.int32, device='cpu')   
        sH = bm.zeros(nfilter, dtype=bm.float64, device='cpu') 
        cc = 0

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

        H = COOTensor(
                indices=bm.astype(bm.stack((iH[:cc], jH[:cc]), axis=0), bm.int32),
                values=sH[:cc],
                spshape=(nx * ny, nx * ny)
            )
        H = H.tocsr()
        H = H.device_put(self.device)
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64)
        
        return H, Hs
    
    def _compute_filter_3d_math(self, 
                nx: int, ny: int, nz: int, 
                hx: float, hy: float, hz: float,
                rmin: float
            ) -> Tuple[COOTensor, TensorLike]:
        """计算 3D 滤波矩阵 - 数学原理的直接实现版本
        
        计算在 CPU 上进行以获得更好的性能
        """
        min_h = min(hx, hy, hz)
        max_cells = ceil(rmin/min_h)
        nfilter = nx * ny * nz * ((2 * (max_cells - 1) + 1) ** 3)
        
        iH = bm.zeros(nfilter, dtype=bm.int32, device='cpu')
        jH = bm.zeros(nfilter, dtype=bm.int32, device='cpu')
        sH = bm.zeros(nfilter, dtype=bm.float64, device='cpu')
        cc = 0

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

        H = COOTensor(
            indices=bm.astype(bm.stack((iH[:cc], jH[:cc]), axis=0), bm.int32),
            values=sH[:cc],
            spshape=(nx * ny * nz, nx * ny * nz)
        )
        H = H.tocsr()
        H = H.device_put(self.device)
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device=self.device)

        return H, Hs
    
    def _compute_filter_2d(self, 
            nx: int, ny: int, 
            hx: float, hy: float,
            rmin: float
        ) -> Tuple[COOTensor, TensorLike]:
        """高性能优化的 2D 滤波矩阵计算 - 分块处理版本
        
        """
        t = timer(f"2D Filter Optimized Timer")
        next(t)
        
        search_radius_x = ceil(rmin/hx)
        search_radius_y = ceil(rmin/hy)
        
        # 批处理参数设置 - 防止处理大模型时内存溢出
        batch_size = min(10000, nx * ny)  # 单批次最大处理单元数
        n_batches = (nx * ny + batch_size - 1) // batch_size  # 总批次数
        
        # 从线性索引到 2D 网格坐标的转换函数
        def linear_to_2d(linear_idx):
            i = linear_idx // ny  # x索引
            j = linear_idx % ny   # y索引
            return i, j
        
        # 优化1: 预计算所有网格单元的物理坐标 (避免重复计算)
        all_coords = bm.zeros((nx * ny, 2), dtype=bm.float64, device='cpu')
        for idx in range(nx * ny):
            i, j = linear_to_2d(idx)
            all_coords[idx, 0] = i * hx  # x坐标
            all_coords[idx, 1] = j * hy  # y坐标
        
        t.send('预处理')
        
        # 优化2: 使用Python列表动态收集结果，避免预分配过大数组
        all_rows = []  # 存储行索引
        all_cols = []  # 存储列索引
        all_vals = []  # 存储权重值
        
        # 分批处理所有单元
        for batch_idx in range(n_batches):
            # 计算当前批次的单元范围
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, nx * ny)
            
            # 当前批次的结果收集器
            batch_rows = []
            batch_cols = []
            batch_vals = []
            
            # 获取当前批次单元的物理坐标
            batch_coords = all_coords[start_idx:end_idx]
            
            # 处理当前批次中的每个单元
            for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
                # 获取网格坐标和线性索引
                i, j = linear_to_2d(global_idx)
                row = global_idx  # 当前单元在矩阵中的行索引
                
                # 计算搜索范围 - 与数学实现版本保持一致
                ii1 = max(0, i - (search_radius_x - 1))
                ii2 = min(nx, i + search_radius_x)
                jj1 = max(0, j - (search_radius_y - 1))
                jj2 = min(ny, j + search_radius_y)
                
                # 创建搜索范围内所有单元的线性索引
                search_indices = []
                for ii in range(ii1, ii2):
                    for jj in range(jj1, jj2):
                        col = ii * ny + jj
                        search_indices.append(col)
                
                # 处理没有邻居的特殊情况
                if not search_indices:
                    continue
                
                # 获取搜索单元的物理坐标
                search_coords = all_coords[search_indices]
                
                # 优化3: 向量化计算距离 (一次性计算所有距离)
                current_coords = batch_coords[local_idx].reshape(1, 2)  # [1, 2]
                diffs = search_coords - current_coords  # [n_search, 2]
                squared_dists = bm.sum(diffs * diffs, axis=1)  # [n_search]
                distances = bm.sqrt(squared_dists)  # [n_search]
                
                # 计算滤波因子并应用过滤条件
                factors = rmin - distances 
                valid_mask = factors > 0  
                
                # 只保留有效的单元对
                if bm.any(valid_mask):
                    # 提取有效的列索引和权重
                    valid_cols = bm.array(search_indices, device='cpu')[valid_mask]
                    valid_factors = factors[valid_mask]
                    
                    # 优化4: 批量添加结果
                    batch_rows.extend([row] * len(valid_cols))
                    batch_cols.extend(valid_cols.tolist())
                    batch_vals.extend(valid_factors.tolist())
            
            # 将当前批次结果添加到总结果
            all_rows.extend(batch_rows)
            all_cols.extend(batch_cols)
            all_vals.extend(batch_vals)
        
        t.send('计算距离和滤波')
        
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
            spshape=(nx * ny, nx * ny)
        )
        
        H = H.tocsr()
        H = H.device_put(self.device)
        
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device=self.device)
        
        t.send('矩阵构建')
        t.send(None)
        
        return H, Hs
    
    def _compute_filter_3d(self, 
                    nx: int, ny: int, nz: int, 
                    hx: float, hy: float, hz: float,
                    rmin: float) -> Tuple[COOTensor, TensorLike]:
        """高性能优化的 3D 滤波矩阵计算 - 分块处理版本
        
        计算在 CPU 上进行以获得更好的性能
        """
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
                
        # 初始化存储结果的列表
        all_rows = []
        all_cols = []
        all_vals = []
        
        # 批处理所有单元
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, nx * ny * nz)
            
            batch_rows = []
            batch_cols = []
            batch_vals = []
            
            # 获取批次单元的坐标
            batch_coords = all_coords[start_idx:end_idx]
            
            # 处理每个单元
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
                    
                # 获取搜索单元的坐标
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
                
        if all_rows:
            iH = bm.tensor(all_rows, dtype=bm.int32, device='cpu')
            jH = bm.tensor(all_cols, dtype=bm.int32, device='cpu')
            sH = bm.tensor(all_vals, dtype=bm.float64, device='cpu')
        else:
            # 处理边缘情况
            iH = bm.tensor([], dtype=bm.int32, device='cpu')
            jH = bm.tensor([], dtype=bm.int32, device='cpu')
            sH = bm.tensor([], dtype=bm.float64, device='cpu')
        
        H = COOTensor(
            indices=bm.stack((iH, jH), axis=0),
            values=sH,
            spshape=(nx * ny * nz, nx * ny * nz)
        )
        
        H = H.tocsr()
        H = H.device_put(self.device)
        
        Hs = H @ bm.ones(H.shape[1], dtype=bm.float64, device=self.device)
        
        return H, Hs
    
    @abstractmethod
    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> None:
        """
        获取初始的物理密度场
        
        Parameters
        - x : 初始设计变量
        - xPhys : 初始物理变量 (输出)
        """
        pass

    abstractmethod
    def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> None:
        """
        对设计变量进行滤波得到物理变量
        
        Parameters
        - x : 原始设计变量
        - xPhys : 过滤后的物理变量 (输出)
        """
        pass

    @abstractmethod
    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> None:
        """
        过滤目标函数的灵敏度
        
        Parameters
        - xPhys : 过滤后的物理变量
        - dobj : 原始的目标函数灵敏度
        """
        pass

    @abstractmethod
    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> None:
        """
        过滤约束函数的灵敏度
        
        Parameters
        - xPhys : 过滤后的物理变量
        - dcos : 原始的约束函数灵敏度
        """
        pass

class SensitivityBasicFilter(BasicFilter):
    """灵敏度滤波器"""
    def __init__(self, mesh: StructuredMesh, rmin: float, domain: List):
        super().__init__(mesh, rmin, domain=domain)
    
    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> None:
        """灵敏度滤波器的初始物理密度等于设计变量"""
        # xPhys[:] = x
        xPhys = bm.set_at(xPhys, slice(None), x)
        return xPhys
    
    def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> None:
        # xPhys[:] = x
        xPhys = bm.set_at(xPhys, slice(None), x)
        return xPhys

    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> None:
        # 计算密度加权的目标函数灵敏度
        weighted_dobj = bm.einsum('c, c -> c', xPhys, dobj)
        # 应用滤波矩阵
        filtered_dobj = self._H.matmul(weighted_dobj)
        # 计算修正因子
        correction_factor = self._Hs * bm.maximum(bm.tensor(0.001, dtype=bm.float64), xPhys)
        # 过滤后的目标函数灵敏度
        # dobj[:] = filtered_dobj / correction_factor
        dobj = bm.set_at(dobj, slice(None), filtered_dobj / correction_factor)
        return dobj

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> None:
        return dcons

class DensityBasicFilter(BasicFilter):
    """密度滤波器"""
    def __init__(self, mesh: StructuredMesh, rmin: float, domain: List):
        super().__init__(mesh, rmin, domain=domain)
    
    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> None:
        """密度滤波器的初始物理密度等于设计变量"""
        # xPhys[:] = x
        xPhys = bm.set_at(xPhys, slice(None), x)
        return xPhys

    def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> None:
        '''
        TODO 需要进一步优化 filtered_x = self._H.matmul(weigthed_x) 的计算效率,
        因为 OC 中二分法求解 Lagrange 乘子的时候会多次调用这个函数
        '''
        # 计算加权密度
        weigthed_x = x * self._cell_measure
        # 应用滤波矩阵
        filtered_x = self._H.matmul(weigthed_x)
        # 返回标准化后的密度
        # xPhys[:] = filtered_x / self._normalize_factor
        xPhys = bm.set_at(xPhys, slice(None), filtered_x / self._normalize_factor)
        return xPhys

    def filter_objective_sensitivities(self, xPhys: TensorLike, dobj: TensorLike) -> None:
        # 计算单元测度加权的目标函数灵敏度
        weighted_dobj = self._cell_measure * dobj
        # 应用滤波矩阵
        # dobj[:] = self._H.matmul(weighted_dobj / self._normalize_factor)
        dobj = bm.set_at(dobj, slice(None), self._H.matmul(weighted_dobj / self._normalize_factor))
        return dobj

    def filter_constraint_sensitivities(self, xPhys: TensorLike, dcons: TensorLike) -> None:
        # 计算单元测度加权的约束函数灵敏度
        weighted_dcons = self._cell_measure * dcons
        # 应用滤波矩阵
        # dcons[:] = self._H.matmul(weighted_dcons / self._normalize_factor)
        dcons = bm.set_at(dcons, slice(None), self._H.matmul(weighted_dcons / self._normalize_factor))
        return dcons

class HeavisideProjectionBasicFilter(BasicFilter):
    """Heaviside 投影滤波器"""
    def __init__(self, mesh: StructuredMesh, rmin: float, domain: List,
                beta: float = 1.0, max_beta: float = 512, continuation_iter: int = 50):
        """
        Parameters
        - mesh : 均匀网格
        - rmin : 滤波半径 (物理距离)
        - beta : Heaviside 投影参数
        """
        super().__init__(mesh, rmin, domain=domain)
        if beta <= 0:
            raise ValueError("Heaviside beta must be positive")
            
        self.beta = beta
        self.max_beta = max_beta
        self.continuation_iter = continuation_iter
        self._xTilde = None  # 存储中间密度场

        self._beta_iter = 0  # 用于追踪 continuation 的内部状态

    def get_initial_density(self, x: TensorLike, xPhys: TensorLike) -> None:
        """Heaviside 投影滤波器的初始物理密度需要投影"""
        self._xTilde = x 
        # xPhys[:] = (1 - bm.exp(-self.beta * self._xTilde) + 
        #           self._xTilde * bm.exp(-self.beta))
        xPhys = bm.set_at(xPhys, slice(None), (1 - bm.exp(-self.beta * self._xTilde) + 
                   self._xTilde * bm.exp(-self.beta)))
        return xPhys
    
    def filter_variables(self, x: TensorLike, xPhys: TensorLike) -> None:
        weighted_x = self._cell_measure * x
        filtered_x = self._H.matmul(weighted_x)
        self._xTilde = filtered_x / self._normalize_factor

        # xPhys[:] = (1 - bm.exp(-self.beta * self._xTilde) + 
        #                 self._xTilde * bm.exp(-self.beta))
        xPhys = bm.set_at(xPhys, slice(None), (1 - bm.exp(-self.beta * self._xTilde) + 
                        self._xTilde * bm.exp(-self.beta)))
        return xPhys

    def filter_objective_sensitivities(self, 
                                    xPhys: TensorLike, dobj: TensorLike) -> None:        
        # 计算 Heaviside 投影的导数
        dx = self.beta * bm.exp(-self.beta * self._xTilde) + bm.exp(-self.beta)
        # 修改灵敏度并应用密度滤波
        weighted_dobj = dobj * dx * self._cell_measure
        # dobj[:] = self._H.matmul(weighted_dobj / self._normalize_factor)
        dobj = bm.set_at(dobj, slice(None), self._H.matmul(weighted_dobj / self._normalize_factor))
        return dobj

    def filter_constraint_sensitivities(self, 
                                    xPhys: TensorLike, dcons: TensorLike) -> None:        
        # 计算 Heaviside 投影的导数
        dx = self.beta * bm.exp(-self.beta * self._xTilde) + bm.exp(-self.beta)
        # 修改灵敏度并应用密度滤波
        weighted_dcons = dcons * dx * self._cell_measure
        # dcons[:] = self._H.matmul(weighted_dcons / self._normalize_factor)
        dcons = bm.set_at(dcons, slice(None), self._H.matmul(weighted_dcons / self._normalize_factor))
        return dcons


    def continuation_step(self, change: float) -> Tuple[float, bool]:
        """
        执行一步 beta continuation
        
        Parameters
        - change : 当前的收敛变化量
        
        Returns
        - new_change : 更新后的收敛变化量
        - continued : 是否执行了 continuation

        """
        self._beta_iter += 1
        
        if (self.beta < self.max_beta and 
                (self._beta_iter >= self.continuation_iter or change <= 0.01)):
            # 增加 beta 值
            self.beta *= 2
            # 重置计数器
            self._beta_iter = 0
            print(f"Beta increased to {self.beta}")
            return 1.0, True
        
        # 如果没有执行 continuation，返回原始的 change 值和 False
        return change, False
            