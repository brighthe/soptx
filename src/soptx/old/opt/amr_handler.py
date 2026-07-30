from typing import Tuple, List, Dict, Any, Callable, Optional
from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.mesh import TriangleMesh, TetrahedronMesh, UniformMesh2d

class AMRHandler:
    """自适应网格加密/粗化处理类"""
    
    def __init__(self, 
                 rho_solid_threshold: float = 0.5,  # 固体单元密度阈值
                 rho_min: float = 1e-3,             # 最小密度
                 amr_radius: float = 0.1,           # AMR 判断半径 (默认为滤波半径)
                 max_level: int = 3):               # 最大加密等级
        """
        初始化AMR处理器
        
        Parameters
        ----------
        rho_solid_threshold : float
            判断单元是否为固体的密度阈值
        rho_min : float
            最小密度值，通常为拓扑优化中的epsilon值
        amr_radius : float
            判断周围单元是否有固体单元的半径
        max_level : int
            最大加密等级
        """
        self.rho_s = rho_solid_threshold
        self.rho_o = rho_min
        self.r_amr = amr_radius
        self.max_level = max_level

    def find_elements_in_radius(self, 
                               mesh,
                               element_idx: int, 
                               radius: float) -> List[int]:
        """
        找出给定单元半径范围内的所有单元
        
        Parameters
        ----------
        mesh : Mesh对象
            网格对象，拥有get_element_center等方法
        element_idx : int
            中心单元索引
        radius : float
            搜索半径
            
        Returns
        -------
        List[int]
            半径范围内的单元索引列表
        """
        # 获取中心单元的坐标
        center = mesh.get_element_center(element_idx)
        
        # 获取所有单元的中心坐标
        all_centers = mesh.get_all_element_centers()
        
        # 计算距离
        distances = bm.linalg.norm(all_centers - center, axis=1)
        
        # 返回距离小于radius的单元索引
        return bm.where(distances <= radius)[0].tolist()

    def mark_elements(self, 
                    mesh, 
                    rho: TensorLike) -> Tuple[List[int], List[int]]:
        """
        标记需要加密和需要粗化的单元, 返回加密和粗化的单元索引
        """
        mesh.celldata['density'] = rho
        mesh.to_vtk('density.vts')

        NC = mesh.number_of_cells()

        refine_mask = bm.zeros(NC, dtype=bm.bool)
        derefine_mask = bm.zeros(NC, dtype=bm.bool)

        solid_mask = rho >= self.rho_s
        refine_mask = bm.logical_or(refine_mask, solid_mask)
        
        void_mask = bm.logical_not(solid_mask)
        void_indices = bm.where(void_mask)[0]
        derefine_mask = bm.logical_or(derefine_mask, void_mask)

        cell_centers = mesh.entity_barycenter('cell')
        derefine_centers = mesh.entity_barycenter('cell', derefine_mask)
        domain = [0, 64, 0, 32]
        cell_indices, neighbor_indices = bm.query_point(
                                    x=cell_centers, y=derefine_centers, h=self.r_amr, 
                                    box_size=domain, mask_self=False, periodic=[False, False, False]
                                )
        # 处理每个 void 单元
        for i, void_idx in enumerate(void_indices):
            # 找出当前 void 单元对应的所有查询结果
            mask = cell_indices == i
            cell_neighbors = neighbor_indices[mask]
            
            # 检查邻居中是否有 solid 单元
            if bm.any(solid_mask[cell_neighbors]):
                # 如果有 solid 单元在附近，标记为加密而非粗化
                refine_mask[void_idx] = True
                derefine_mask[void_idx] = False


        refine_indices = bm.where(refine_mask)[0]
        derefine_indices = bm.where(derefine_mask)[0]
        
        return refine_indices, derefine_indices
    
    def perform_amr(self, 
                  mesh, 
                  rho: TensorLike) -> Tuple[Any, TensorLike]:
        """
        执行自适应网格加密/粗化
        
        Parameters
        ----------
        mesh : Mesh对象
            网格对象
        rho : TensorLike
            密度场
            
        Returns
        -------
        Tuple[Any, TensorLike]
            更新后的网格和密度场
        """
        # 1. 标记需要加密和粗化的单元
        to_refine, to_derefine = self.mark_elements(mesh, rho)
        
        # 2. 调整粗化标记，确保网格兼容性
        final_derefine = self.adjust_derefinement_marks(mesh, to_derefine)
        
        # 3. 执行网格加密
        mesh, old_to_new_map = mesh.refine_elements(to_refine)
        
        # 4. 更新密度场
        new_rho = self._update_density_after_refinement(rho, old_to_new_map)
        
        # 5. 执行网格粗化
        mesh, new_to_old_map = mesh.derefine_elements(final_derefine)
        
        # 6. 再次更新密度场
        new_rho = self._update_density_after_derefinement(new_rho, new_to_old_map)
        
        return mesh, new_rho
    
    def create_amr_callback(self) -> Callable:
        """
        创建用于OC优化器的AMR回调函数
        
        Returns
        -------
        Callable
            AMR回调函数
        """
        def amr_callback(rho: TensorLike, rho_phys: TensorLike) -> Tuple[TensorLike, TensorLike]:
            """
            AMR回调函数
            
            Parameters
            ----------
            rho : TensorLike
                设计变量密度场
            rho_phys : TensorLike
                物理密度场
                
            Returns
            -------
            Tuple[TensorLike, TensorLike]
                更新后的设计变量和物理密度场
            """
            # 获取当前网格（这里需要从外部获取，实际应用中可能需要调整）
            mesh = self.current_mesh
            
            # 执行AMR
            new_mesh, new_rho = self.perform_amr(mesh, rho_phys)
            
            # 更新当前网格
            self.current_mesh = new_mesh
            
            # 根据新的物理密度更新设计变量
            # 如果使用过滤器，这里的逻辑可能需要调整
            new_rho_design = new_rho.copy()
            
            return new_rho_design, new_rho
        
        return amr_callback