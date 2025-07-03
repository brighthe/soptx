class BaseMaterialStrategy(ABC):
    def __init__(self, material, space: TensorFunctionSpace):
        self.material = material
        self.space = space

    @abstractmethod
    def update_density(self, density: TensorLike):
        """更新策略所使用的密度场。"""
        pass

    @abstractmethod
    def assembly_stiffness_matrix(self) -> "CSRTensor":
        """根据当前密度，使用特定策略组装全局刚度矩阵。"""
        pass

class SingleGridStrategy(BaseMaterialStrategy):
    def __init__(self, material, space: TensorFunctionSpace):
        super().__init__(material, space)
        self._current_density = None
        # 缓存基础刚度矩阵 K_e^0 以提高效率
        self._base_local_stiffness_matrix = self._compute_base_local_stiffness()

    def update_density(self, density: TensorLike):
        # 密度是每个单元一个值 (NC, )
        self._current_density = density

    def _compute_base_local_stiffness(self):
        # ... (此处代码计算并返回 K_e^0) ...
        bform = BilinearForm(self.space)
        # 使用基础材料 E_0 来创建积分子
        integrator = LinearElasticIntegrator(self.material.get_base_material(), q=self.space.p + 3)
        bform.add_integrator(integrator)
        return bform.get_local_matrix() # 假设有方法可以获取局部矩阵

    def assembly_stiffness_matrix(self) -> "CSRTensor":
        if self._current_density is None:
            raise ValueError("Density not set for SingleGridStrategy.")

        # 核心逻辑：对应公式 (5) 
        p = self.material.penalty # 获取惩罚因子 p
        scaling_factor = self._current_density ** p
        
        # 缩放局部矩阵: K_e(ρ_e) = (ρ_e)^p * K_e^0
        K_local_scaled = bm.einsum('c, cij -> cij', scaling_factor, self._base_local_stiffness_matrix)
        
        # 从缩放后的局部矩阵组装全局矩阵
        # (这可能需要 fealpy 提供一个从局部矩阵直接组装的功能)
        # 或者，如果 fealpy 不支持，可以手动组装
        # K_global = self.space.assembly_from_local(K_local_scaled)
        
        # 替代方案：如果 fealpy 的积分子支持传入一个 per-element 的材料
        self.material.update_elastic_modulus(self._current_density)
        bform = BilinearForm(self.space)
        integrator = LinearElasticIntegrator(self.material, q=self.space.p + 3)
        bform.add_integrator(integrator)
        return bform.assembly(format='csr')