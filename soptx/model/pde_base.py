from abc import ABC, abstractmethod
from typing import Optional, List

from fealpy.mesh import HomogeneousMesh
from fealpy.typing import TensorLike

from ..utils.base_logged import BaseLogged

class PDEBase(BaseLogged, ABC):
    """PDE 基类，提供网格管理功能"""
    
    def __init__(self,
                domain: List[float] = [0, 1, 0, 1],
                mesh_type: str = 'uniform_tri',
                enable_logging: bool = False, 
                logger_name: Optional[str] = None 
                ):

        super().__init__(enable_logging=enable_logging, logger_name=logger_name)

        self._domain = domain
        self._mesh_type = mesh_type

    
    #######################################################################################################################
    # 访问器
    #######################################################################################################################
    
    @property
    def domain(self) -> List[float]:
        """获取计算域"""
        return self._domain
    
    @property
    def plane_type(self) -> str:
        """平面类型 (plane_stress, plane_strain, 3d)"""
        return getattr(self, '_plane_type', 'unknown')
    
    @property
    def force_type(self) -> str:
        """载荷类型 (continuous, concentrated)"""
        return getattr(self, '_force_type', 'unknown')
    
    @property
    def boundary_type(self) -> str:
        """边界条件类型 (dirichlet, neumann, robin)"""
        return getattr(self, '_boundary_type', 'unknown')


    #######################################################################################################################
    # 内部方法
    #######################################################################################################################

    def _save_mesh(self, mesh: HomogeneousMesh, mesh_type: str, **params) -> None:
        """保存网格数据"""
        self._mesh = mesh
        self.mesh_type = mesh_type

        nx = params.get('nx', 10)  
        ny = params.get('ny', 10)
        nz = params.get('nz', None)

        hx = (self._domain[1] - self._domain[0]) / nx
        hy = (self._domain[3] - self._domain[2]) / ny
        
        mesh.meshdata = getattr(mesh, 'meshdata', {})
        metadata = {
            'domain': self._domain,
            'mesh_type': mesh_type, 
            'nx': nx, 'ny': ny,
            'hx': hx, 'hy': hy,
        }

        if nz is not None and len(self._domain) >= 6:
            hz = (self._domain[5] - self._domain[4]) / nz
            metadata.update({'nz': nz, 'hz': hz})

        mesh.meshdata.update(metadata)

        self._log_info(f"Meshdata saved: {mesh_type} ({nx}x{ny}), "
                      f"cells: {mesh.number_of_cells()}")
        
    @abstractmethod
    def init_mesh(self, **kwargs) -> HomogeneousMesh:
        """初始化网格"""
        pass

    @abstractmethod
    def body_force(self, points: TensorLike) -> TensorLike:
        """体力"""
        pass