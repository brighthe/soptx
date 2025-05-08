from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Union
from fealpy.mesh import HomogeneousMesh, SimplexMesh, TensorMesh, StructuredMesh
from fealpy.functionspace import TensorFunctionSpace
from fealpy.fem import LinearElasticIntegrator, BilinearForm, DirichletBC
from fealpy.fem.dirichlet_bc import apply_csr_matrix
from fealpy.sparse import CSRTensor
from fealpy.solver import cg, spsolve

from soptx.material import BaseElasticMaterialInstance
from soptx.utils import timer

@dataclass
class IterativeSolverResult:
    """迭代求解器的结果"""
    displacement: TensorLike
    solve_info: dict

@dataclass
class DirectSolverResult:
    """直接求解器的结果"""
    displacement: TensorLike

class AssemblyMethod(Enum):
    """矩阵组装方法的枚举类"""
    STANDARD = auto()             # 标准组装
    VOIGT = auto()                # Voigt 格式组装
    FAST = auto()                 # 快速组装
    SYMBOLIC = auto()             # 符号组装

class ElasticFEMSolver:
    """专门用于求解线弹性问题的有限元求解器
    
    该求解器负责：
    1. 密度状态的管理
    2. 对应密度下的材料实例管理
    3. 有限元离散化和求解
    4. 边界条件的处理
    """
    
    def __init__(self, 
                materials: BaseElasticMaterialInstance, 
                tensor_space: TensorFunctionSpace,
                pde,
                assembly_method: AssemblyMethod,
                solver_type: str,
                solver_params: Optional[dict]):
        """
        Parameters
        - materials : 材料
        - tensor_space : 张量函数空间
        - pde : 包含荷载和边界条件的 PDE 模型
        - assembly_method : 矩阵组装方法
        - solver_type : 求解器类型, 'cg' 或 'direct' 
        - solver_params : 求解器参数
            - cg: maxiter, atol, rtol
            - direct: solver_type
        """
        self.materials = materials
        self.tensor_space = tensor_space
        self.pde = pde
        self.assembly_method = assembly_method
        self.solver_type = solver_type
        self.solver_params = solver_params or {}

        self._integrator = self._create_integrator()

        # 状态管理
        self._current_density = None
            
        # 缓存
        self._base_local_stiffness_matrix = None
        self._base_local_trace_matrix = None

    #---------------------------------------------------------------------------
    # 公共属性
    #---------------------------------------------------------------------------
    @property
    def get_current_density(self) -> Optional[TensorLike]:
        """获取当前密度"""
        return self._current_density
    
    @property
    def get_current_material(self) -> Optional[BaseElasticMaterialInstance]:
        """获取当前材料实例"""
        if self.materials is None:
            raise ValueError("Material not initialized. Call update_density first.")
        
        return self.materials

    #----------------------------------------------------------------------------------
    # 状态管理相关方法
    #----------------------------------------------------------------------------------
    def update_status(self, density: TensorLike) -> None:
        """更新相关状态"""
        if density is None:
            raise ValueError("'density' cannot be None")
            
        # 1. 更新密度场
        self._current_density = density
        # 2. 根据新密度更新材料属性
        self.materials.update_elastic_modulus(self._current_density)
    
    def get_base_local_stiffness_matrix(self) -> TensorLike:
        """获取基础材料的局部刚度矩阵 (会被缓存)"""
        if self._base_local_stiffness_matrix is None:
            base_material = self.materials.get_base_material()
            integrator = LinearElasticIntegrator(
                                material=base_material,
                                q=self.tensor_space.p+3,
                                method=None
                            )
            self._base_local_stiffness_matrix = integrator.assembly(space=self.tensor_space)

        return self._base_local_stiffness_matrix
    
    def get_base_local_trace_matrix(self) -> TensorLike:
        """获取基础材料的局部迹矩阵 (会被缓存)"""
        if self._base_local_trace_matrix is None:
            base_material = self.materials.get_base_material()
            integrator = LinearElasticIntegrator(
                            material=base_material,
                            q=self.tensor_space.p+3,
                            method=None
                        )
            
            scalar_space = self.tensor_space.scalar_space
            mesh = getattr(scalar_space, 'mesh', None)
            
            cm, bcs, ws, gphi, detJ = integrator.fetch_assembly(self.tensor_space)
            
            D = base_material.elastic_matrix(bcs)
            
            GD = mesh.geo_dimension()
            if GD == 2:
                D00 = D[..., 0, 0, None]  # E/(1-ν²) 或 2μ+λ
                D01 = D[..., 0, 1, None]  # νE/(1-ν²) 或 λ
                trace_coef = D00 + D01    # 2(λ+μ) 或 E/(1-ν)
            else:  # GD == 3
                D00 = D[..., 0, 0, None]       # 2μ+λ
                D01 = D[..., 0, 1, None]       # λ
                D02 = D[..., 0, 2, None]       # λ
                trace_coef = D00 + D01 + D02   # 3λ + 2μ
            
            # 计算基础矩阵
            if isinstance(mesh, SimplexMesh):
                A_xx = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 0], gphi[..., 0], cm)
                A_yy = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 1], gphi[..., 1], cm)
                A_xy = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 0], gphi[..., 1], cm)
                A_yx = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 1], gphi[..., 0], cm)
                
                if GD == 3:
                    A_zz = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 2], gphi[..., 2], cm)
                    A_xz = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 0], gphi[..., 2], cm)
                    A_zx = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 2], gphi[..., 0], cm)
                    A_yz = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 1], gphi[..., 2], cm)
                    A_zy = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 2], gphi[..., 1], cm)
            else:
                A_xx = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 0], gphi[..., 0], detJ)
                A_yy = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 1], gphi[..., 1], detJ)
                A_xy = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 0], gphi[..., 1], detJ)
                A_yx = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 1], gphi[..., 0], detJ)
                
                if GD == 3:
                    A_zz = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 2], gphi[..., 2], detJ)
                    A_xz = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 0], gphi[..., 2], detJ)
                    A_zx = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 2], gphi[..., 0], detJ)
                    A_yz = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 1], gphi[..., 2], detJ)
                    A_zy = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 2], gphi[..., 1], detJ)
            
            # 初始化迹矩阵
            NC = mesh.number_of_cells()
            ldof = scalar_space.number_of_local_dofs()
            KTr = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64, device=mesh.device)
            
            # 根据维度和自由度排序计算迹矩阵
            if GD == 2:
                if self.tensor_space.dof_priority:
                     # 填充对角块
                    KTr = bm.set_at(KTr, (slice(None), slice(0, ldof), slice(0, ldof)), A_xx)
                    KTr = bm.set_at(KTr, (slice(None), slice(ldof, 2*ldof), slice(ldof, 2*ldof)), A_yy)

                    # 填充非对角块
                    KTr = bm.set_at(KTr, (slice(None), slice(0, ldof), slice(ldof, 2*ldof)), A_xy)
                    KTr = bm.set_at(KTr, (slice(None), slice(ldof, 2*ldof), slice(0, ldof)), A_yx)

                    KTr = trace_coef * KTr
                else:
                    # 填充对角块
                    KTr = bm.set_at(KTr, (slice(None), slice(0, KTr.shape[1], GD), slice(0, KTr.shape[2], GD)), A_xx)
                    KTr = bm.set_at(KTr, (slice(None), slice(1, KTr.shape[1], GD), slice(1, KTr.shape[2], GD)), A_yy)
                    
                    # 填充非对角块
                    KTr = bm.set_at(KTr, (slice(None), slice(0, KTr.shape[1], GD), slice(1, KTr.shape[2], GD)), A_xy)
                    KTr = bm.set_at(KTr, (slice(None), slice(1, KTr.shape[1], GD), slice(0, KTr.shape[2], GD)), A_yx)

                    KTr = trace_coef * KTr
            else: 
                if self.tensor_space.dof_priority:
                    # 填充对角块
                    KTr = bm.set_at(KTr, (slice(None), slice(0, ldof), slice(0, ldof)), A_xx)
                    KTr = bm.set_at(KTr, (slice(None), slice(ldof, 2*ldof), slice(ldof, 2*ldof)), A_yy)
                    KTr = bm.set_at(KTr, (slice(None), slice(2*ldof, 3*ldof), slice(2*ldof, 3*ldof)), A_zz)
                    
                    # 填充非对角块
                    KTr = bm.set_at(KTr, (slice(None), slice(0, ldof), slice(ldof, 2*ldof)), A_xy)
                    KTr = bm.set_at(KTr, (slice(None), slice(ldof, 2*ldof), slice(0, ldof)), A_yx)
                    KTr = bm.set_at(KTr, (slice(None), slice(0, ldof), slice(2*ldof, 3*ldof)), A_xz)
                    KTr = bm.set_at(KTr, (slice(None), slice(2*ldof, 3*ldof), slice(0, ldof)), A_zx)
                    KTr = bm.set_at(KTr, (slice(None), slice(ldof, 2*ldof), slice(2*ldof, 3*ldof)), A_yz)
                    KTr = bm.set_at(KTr, (slice(None), slice(2*ldof, 3*ldof), slice(ldof, 2*ldof)), A_zy)
                    
                    KTr = trace_coef * KTr
                else:
                    # 填充对角块
                    KTr = bm.set_at(KTr, (slice(None), slice(0, KTr.shape[1], GD), slice(0, KTr.shape[2], GD)), A_xx)
                    KTr = bm.set_at(KTr, (slice(None), slice(1, KTr.shape[1], GD), slice(1, KTr.shape[2], GD)), A_yy)
                    KTr = bm.set_at(KTr, (slice(None), slice(2, KTr.shape[1], GD), slice(2, KTr.shape[2], GD)), A_zz)
                    
                    # 填充非对角块
                    KTr = bm.set_at(KTr, (slice(None), slice(0, KTr.shape[1], GD), slice(1, KTr.shape[2], GD)), A_xy)
                    KTr = bm.set_at(KTr, (slice(None), slice(1, KTr.shape[1], GD), slice(0, KTr.shape[2], GD)), A_yx)
                    KTr = bm.set_at(KTr, (slice(None), slice(0, KTr.shape[1], GD), slice(2, KTr.shape[2], GD)), A_xz)
                    KTr = bm.set_at(KTr, (slice(None), slice(2, KTr.shape[1], GD), slice(0, KTr.shape[2], GD)), A_zx)
                    KTr = bm.set_at(KTr, (slice(None), slice(1, KTr.shape[1], GD), slice(2, KTr.shape[2], GD)), A_yz)
                    KTr = bm.set_at(KTr, (slice(None), slice(2, KTr.shape[1], GD), slice(1, KTr.shape[2], GD)), A_zy)
                    
                    KTr = trace_coef * KTr
                    
            self._base_local_trace_matrix = KTr
            
        return self._base_local_trace_matrix
    
    def compute_local_stiffness_matrix(self, 
            density: Optional[TensorLike] = None
        ) -> TensorLike:
        """计算特定密度下的局部刚度矩阵"""
        # 确定要使用的密度
        if density is not None:
            if not bm.is_tensor(density):
                raise TypeError(
                    f"密度参数类型错误: 期望 TensorLike 类型, 但得到{type(density).__name__}. "
                    "请提供有效的张量类型."
                )
            use_density = density
        else:
            # 使用当前状态的密度
            if self._current_density is None:
                raise ValueError(
                    "当前密度未设置. 请先调用 update_status(density) 设置密度,"
                    "或者在调用 compute_local_stiffness_matrix(density) 时提供密度参数."
                )
            use_density = self._current_density
        
        base_K = self.get_base_local_stiffness_matrix()
        E = self.materials.calculate_elastic_modulus(use_density)
    
        if len(E.shape) > 0:
            KE = bm.einsum('c, cij -> cij', E, base_K)
        else:
            KE = E * base_K
        
        return KE

    def compute_local_stiffness_matrix_old(self) -> TensorLike:
        """计算当前材料的局部刚度矩阵 (每次重新计算)"""
        integrator = self._integrator
 
        # 根据 assembly_config.method 选择对应的组装函数
        method_map = {
            AssemblyMethod.STANDARD: integrator.assembly,
            AssemblyMethod.VOIGT: integrator.voigt_assembly,
            AssemblyMethod.FAST: integrator.fast_assembly,
            AssemblyMethod.SYMBOLIC: integrator.symbolic_assembly,
        }
        
        try:
            assembly_func = method_map[self.assembly_method]
        except KeyError:
            raise RuntimeError(f"Unsupported assembly method: {self.assembly_method}")
        
        # 调用选定的组装函数
        KE = assembly_func(space=self.tensor_space)
        
        return KE

    #----------------------------------------------------------------------------------
    # 内部方法：组装和边界条件处理
    #----------------------------------------------------------------------------------
    def _create_integrator(self) -> LinearElasticIntegrator:
        """创建适当的积分器实例"""
        # 确定积分方法
        method_map = {
            AssemblyMethod.STANDARD: 'assembly',
            AssemblyMethod.VOIGT: 'voigt',
            AssemblyMethod.FAST: 'fast',
            AssemblyMethod.SYMBOLIC: 'symbolic',
        }
        
        method = method_map[self.assembly_method]

        # 创建积分器
        q = self.tensor_space.p + 3
        integrator = LinearElasticIntegrator(
                            material=self.materials, 
                            q=q, method=method
                        )
        integrator.keep_data(True)
        
        return integrator
    
    def _assemble_global_stiffness_matrix(self) -> CSRTensor:
        """组装全局刚度矩阵"""    
        integrator = self._integrator
        bform = BilinearForm(self.tensor_space)
        bform.add_integrator(integrator)
        K = bform.assembly(format='csr')

        return K
    
    def _assemble_global_force_vector(self) -> TensorLike:
        """组装全局载荷向量

        Returns:
        - 单载荷情况: 形状为 (tgdof,) 的一维张量
        - 多载荷情况: 形状为 (nloads, tgdof) 的二维张量
        """
        force = self.pde.force
        force_at_nodes = force(self.tensor_space.mesh.entity('node'))

        is_multi_load = len(force_at_nodes.shape) == 3

        if is_multi_load:
            nloads = force_at_nodes.shape[0]
            tgdof = self.tensor_space.number_of_global_dofs()
            kwargs = bm.context(force_at_nodes)
            F = bm.zeros((nloads, tgdof), **kwargs)

            for i in range(nloads):
                single_force = lambda p: force_at_nodes[i]
                F_i = self.tensor_space.interpolate(single_force)
                F[i] = F_i
        else:
            F = self.tensor_space.interpolate(force)

        return F

        # # ! 多载荷情况
        # force = self.pde.force
        # force_tensor = force(self.tensor_space.mesh.entity('node'))
        # nloads = force_tensor.shape[0]
        # tgdof = self.tensor_space.number_of_global_dofs()
        # kwargs = bm.context(force_tensor)
        # F = bm.zeros((nloads, tgdof), **kwargs)
        # for i in range(nloads):
        #     single_force = lambda p: force_tensor[i]
        #     F_i = self.tensor_space.interpolate(single_force)
        #     F[i:, ] = F_i

        # return F
    
    def _apply_matrix(self, A: CSRTensor, isDDof: TensorLike):
        """
        FEALPy 中的 apply_matrix 使用了 D0@A@D0, 
        不同后端下 @ 会使用大量的 for 循环, 这在 GPU 下非常缓慢 
        """
        isIDof = bm.logical_not(isDDof)
        crow = A.crow
        col = A.col
        indices_context = bm.context(col)
        ZERO = bm.array([0], **indices_context)

        nnz_per_row = crow[1:] - crow[:-1]
        remain_flag = bm.repeat(isIDof, nnz_per_row) & isIDof[col] # 保留行列均为内部自由度的非零元素
        rm_cumsum = bm.concat([ZERO, bm.cumsum(remain_flag, axis=0)], axis=0) # 被保留的非零元素数量累积
        nnz_per_row = rm_cumsum[crow[1:]] - rm_cumsum[crow[:-1]] + isDDof # 计算每行的非零元素数量

        new_crow = bm.cumsum(bm.concat([ZERO, nnz_per_row], axis=0), axis=0)

        NNZ = new_crow[-1]
        non_diag = bm.ones((NNZ,), dtype=bm.bool, device=bm.get_device(isDDof)) # Field: non-zero elements
        loc_flag = bm.logical_and(new_crow[:-1] < NNZ, isDDof)
        non_diag = bm.set_at(non_diag, new_crow[:-1][loc_flag], False)

        # 修复：只选取适当数量的值对应设置
        # 找出所有边界DOF对应的行索引
        bd_rows = bm.where(loc_flag)[0]
        new_col = bm.empty((NNZ,), **indices_context)
        # 设置为相应行的边界DOF位置
        new_col = bm.set_at(new_col, new_crow[:-1][loc_flag], bd_rows)
        # 设置非对角元素的列索引
        new_col = bm.set_at(new_col, non_diag, col[remain_flag])

        new_values = bm.empty((NNZ,), **A.values_context())
        new_values = bm.set_at(new_values, new_crow[:-1][loc_flag], 1.)
        new_values = bm.set_at(new_values, non_diag, A.values[remain_flag])

        return CSRTensor(new_crow, new_col, new_values, A.sparse_shape)
    
    def _apply_boundary_conditions(self, 
                                K: CSRTensor, F: TensorLike, 
                                enable_timing: bool = False
                            ) -> tuple[CSRTensor, TensorLike]:
        """应用边界条件

        Parameters:
        - K: 全局刚度矩阵
        - F: 全局载荷向量, 形状可以是 (tgdof,) 或 (nloads, tgdof)
        - enable_timing: 是否启用计时
        """
        t = None
        if enable_timing:
            t = timer(f"Apply Boundary Timing")
            next(t)

        dirichlet = self.pde.dirichlet
        threshold = self.pde.threshold()

        is_multi_load = len(F.shape) > 1

        gdof = self.tensor_space.number_of_global_dofs()
        uh_bd_base = bm.zeros(gdof, 
                            dtype=bm.float64, device=self.tensor_space.device)
        uh_bd_base, isBdDof = self.tensor_space.boundary_interpolate(
                            gd=dirichlet, threshold=threshold, method='interp')

        if enable_timing:
            t.send('1')

        if is_multi_load:
            nloads = F.shape[0]
            kwargs = bm.context(F)
            uh_bd = bm.zeros((nloads, gdof), **kwargs)

            for i in range(nloads):
                uh_bd[i] = uh_bd_base[:]  
            
            for i in range(nloads):
                F[i] = F[i] - K.matmul(uh_bd[i])
                F[i, isBdDof] = uh_bd[i, isBdDof]
        else:
            F = F - K.matmul(uh_bd_base[:])
            F[isBdDof] = uh_bd_base[isBdDof]

        if enable_timing:
            t.send('2')
        
        K = self._apply_matrix(A=K, isDDof=isBdDof)

        # ! DirichletBC 类在 GPU 下速度非常慢
        # device = self.tensor_space.scalar_space.mesh.device
        # dbc = DirichletBC(space=self.tensor_space, gd=dirichlet,
        #                 threshold=threshold, method='interp')
        # if bm.backend_name == 'jax' or device == 'cuda':
        #     K = self._apply_matrix(A=K, isDDof=isBdDof)
        # else:
        #     K = dbc.apply_matrix(matrix=K, check=True)

        if enable_timing:
            t.send('3')
            t.send(None)
        
        return K, F

    #---------------------------------------------------------------------------
    # 求解方法
    #---------------------------------------------------------------------------
    def solve(self) -> Union[IterativeSolverResult, DirectSolverResult]:
        """统一的求解接口"""
        if self.solver_type == 'cg':
            return self.solve_cg(**self.solver_params)
        elif self.solver_type == 'direct':
            return self.solve_direct(**self.solver_params)
        else:
            raise ValueError(f"Unsupported solver type: {self.solver_type}")
               
    def solve_cg(self, 
            maxiter: int = 5000,
            atol: float = 1e-12,
            rtol: float = 1e-12,    
            x0: Optional[TensorLike] = None,
            enable_timing: bool = False,
        ) -> IterativeSolverResult:
        """使用共轭梯度法求解"""
        if self._current_density is None:
            raise ValueError("Density not set. Call update_density first.")
    
        t = None
        if enable_timing:
            t = timer(f"CG Solver Timing")
            next(t)
            
        K0 = self._assemble_global_stiffness_matrix()
        
        if enable_timing:
            t.send('矩阵组装时间')
            
        F0 = self._assemble_global_force_vector()

        if enable_timing:
            t.send('右端项组装时间')
        
        K, F = self._apply_boundary_conditions(K0, F0)

        if enable_timing:
            t.send('边界条件处理时间')
            
        uh = bm.zeros(F.shape, dtype=bm.float64, device=F.device)

        try:
            uh[:], info = cg(K, F[:], x0=x0,
                            batch_first=True, 
                            atol=atol, rtol=rtol, 
                            maxit=maxiter, returninfo=True)
        except Exception as e:
            raise RuntimeError(f"CG solver failed: {str(e)}")
        
        if enable_timing:
            t.send('求解时间')
            t.send(None)

        return IterativeSolverResult(displacement=uh, solve_info=info)
    
    def solve_direct(self, 
                    solver_type: str = 'mumps', 
                    enable_timing: bool = False,
                ) -> DirectSolverResult:
        """使用直接法求解"""
        if self._current_density is None:
            raise ValueError("Density not set. Call 'update_density' first.")
        
        t = None
        if enable_timing:
            t = timer(f"MUMPS Solver Timing")
            next(t)
            
        K0 = self._assemble_global_stiffness_matrix()
        
        if enable_timing:
            t.send('矩阵组装时间')
            
        F0 = self._assemble_global_force_vector()

        if enable_timing:
            t.send('右端项组装时间')

        K, F = self._apply_boundary_conditions(K0, F0)
        
        if enable_timing:
            t.send('边界条件处理时间')
            
        uh = self.tensor_space.function()

        try:
            # TODO 能否支持批量求解
            uh[:] = spsolve(K, F[:], solver=solver_type)
        except Exception as e:
            raise RuntimeError(f"Direct solver failed: {str(e)}")
        
        if enable_timing:
            t.send('求解时间')
            t.send(None)

        return DirectSolverResult(displacement=uh)