"""
SOPTX 全局界面组装与全尺寸有限元求解器
=========================================================
基于 SOPTX 原生的 QuadrangleMesh (四边形网格) 与 LinearElasticIntegrator (线弹性积分器)，
用于全尺寸有限元分析 (FEA) 参考直接求解以及 Scatter-Add 界面组装。

作者: Liang He (大连理工大学博士后) & Antigravity Assistant
日期: 2026-08-06
"""

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, List, Union, Any
from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh, HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem import BilinearForm
from soptx.fem.integrators.linear_elastic_integrator import LinearElasticIntegrator
from soptx.materials import IsotropicLinearElasticMaterial


class GlobalAssembler:
    """
    使用 SOPTX 原生积分器组装全局界面系统以及全尺寸有限元参考系统（支持 2D 及 3D）。
    """
    def __init__(
        self,
        domain_size: Union[Tuple[float, ...], float],
        n_sub: Union[Tuple[int, ...], float, int],
        n_fine: Union[Tuple[int, ...], int],
        *args: Any,
        E_base: float = 1.0,
        nu: float = 0.3
    ) -> None:
        self.E_base: float = E_base
        self.nu: float = nu

        # 灵活解析 2D/3D 参数
        all_args = (domain_size, n_sub, n_fine) + args
        if isinstance(all_args[0], (tuple, list)):
            self.domain_size: Tuple[float, ...] = tuple(all_args[0])
            self.n_sub: Tuple[int, ...] = tuple(all_args[1])
            self.n_fine: Tuple[int, ...] = tuple(all_args[2])
            self.E_base = all_args[3] if len(all_args) > 3 else E_base
            self.nu = all_args[4] if len(all_args) > 4 else nu
        elif len(all_args) >= 6:
            self.domain_size = (float(all_args[0]), float(all_args[1]))
            self.n_sub = (int(all_args[2]), int(all_args[3]))
            self.n_fine = (int(all_args[4]), int(all_args[5]))
            self.E_base = all_args[6] if len(all_args) > 6 else E_base
            self.nu = all_args[7] if len(all_args) > 7 else nu
        else:
            raise ValueError("无法解析的 GlobalAssembler 输入参数")

        self.dim: int = len(self.domain_size)

        # 属性别名
        self.Lx = self.domain_size[0]
        self.Ly = self.domain_size[1]
        self.n_sub_x = self.n_sub[0]
        self.n_sub_y = self.n_sub[1]
        self.n_fine_x = self.n_fine[0]
        self.n_fine_y = self.n_fine[1]
        self.total_fine_x = self.n_sub[0] * self.n_fine[0]
        self.total_fine_y = self.n_sub[1] * self.n_fine[1]

        if self.dim == 3:
            self.Lz = self.domain_size[2]
            self.n_sub_z = self.n_sub[2]
            self.n_fine_z = self.n_fine[2]
            self.total_fine_z = self.n_sub[2] * self.n_fine[2]

        self.total_fine = tuple(self.n_sub[d] * self.n_fine[d] for d in range(self.dim))

        box = [coord for d in range(self.dim) for coord in (0.0, self.domain_size[d])]
        if self.dim == 2:
            self.full_mesh = QuadrangleMesh.from_box(box=box, nx=self.total_fine[0], ny=self.total_fine[1])
            self.material = IsotropicLinearElasticMaterial(youngs_modulus=self.E_base, poisson_ratio=self.nu, hypothesis='plane_stress')
        else:
            self.full_mesh = HexahedronMesh.from_box(box=box, nx=self.total_fine[0], ny=self.total_fine[1], nz=self.total_fine[2])
            self.material = IsotropicLinearElasticMaterial(youngs_modulus=self.E_base, poisson_ratio=self.nu)

        self.sspace_full: LagrangeFESpace = LagrangeFESpace(self.full_mesh, p=1, ctype='C')
        self.space_full: TensorFunctionSpace = TensorFunctionSpace(self.sspace_full, shape=(-1, self.dim))

        self.n_full_nodes_x: int = self.total_fine[0] + 1
        self.n_full_nodes_y: int = self.total_fine[1] + 1
        if self.dim == 3:
            self.n_full_nodes_z: int = self.total_fine[2] + 1

        self.total_full_nodes: int = self.full_mesh.number_of_nodes()
        self.total_full_dofs: int = self.space_full.number_of_global_dofs()

    def get_substructure_global_dofs(self, *args: Any) -> Any:
        """
        将局部子结构的自由度 (DOF) 映射到全局全网格的自由度 (DOF)。
        支持 2D (sx, sy, sub_mesh) 与 3D (sx, sy, sz, sub_mesh) 或 (sub_pos, sub_mesh)。
        """
        if len(args) == 2:
            sub_pos, sub_mesh = args[0], args[1]
        elif len(args) == 3:
            sub_pos, sub_mesh = (args[0], args[1]), args[2]
        else:
            sub_pos, sub_mesh = (args[0], args[1], args[2]), args[3]

        sub_global_dofs: List[int] = []
        if self.dim == 2:
            sx, sy = sub_pos[0], sub_pos[1]
            for ix in range(sub_mesh.n_nodes_x):
                for iy in range(sub_mesh.n_nodes_y):
                    gx = sx * self.n_fine[0] + ix
                    gy = sy * self.n_fine[1] + iy
                    gnode = gx * self.n_full_nodes_y + gy
                    sub_global_dofs.extend([2 * gnode, 2 * gnode + 1])
        else:
            sx, sy, sz = sub_pos[0], sub_pos[1], sub_pos[2]
            for ix in range(sub_mesh.n_nodes_x):
                for iy in range(sub_mesh.n_nodes_y):
                    for iz in range(sub_mesh.n_nodes_z):
                        gx = sx * self.n_fine[0] + ix
                        gy = sy * self.n_fine[1] + iy
                        gz = sz * self.n_fine[2] + iz
                        gnode = (gx * self.n_full_nodes_y + gy) * self.n_full_nodes_z + gz
                        sub_global_dofs.extend([3 * gnode, 3 * gnode + 1, 3 * gnode + 2])
        return bm.array(sub_global_dofs, dtype=bm.int64)

    def solve_fullscale_fea(
        self,
        densities: Union[List[Any], Any],
        load_dof: int,
        load_val: float = -1.0,
        bc_type: str = "cantilever"
    ) -> Tuple[Any, Any]:
        """
        使用 SOPTX 原生的 LinearElasticIntegrator 组装并求解全尺寸精细有限元直接参考解：
        K_full U_full = F_full
        支持 'cantilever' (左端固定) 和 'mbb' (MBB 简支梁) 边界条件。
        """
        # 向量化拼接全局密度网格
        if self.dim == 2:
            full_density_grid = (
                bm.asarray(densities)
                .reshape(self.n_sub[0], self.n_sub[1], self.n_fine[0], self.n_fine[1])
                .transpose(0, 2, 1, 3)
                .reshape(self.total_fine[0], self.total_fine[1])
            )
        else:
            full_density_grid = (
                bm.asarray(densities)
                .reshape(self.n_sub[0], self.n_sub[1], self.n_sub[2], self.n_fine[0], self.n_fine[1], self.n_fine[2])
                .transpose(0, 3, 1, 4, 2, 5)
                .reshape(self.total_fine[0], self.total_fine[1], self.total_fine[2])
            )

        simp_coef = bm.asarray(full_density_grid.flatten()**3.0, dtype=bm.float64)

        integrator = LinearElasticIntegrator(material=self.material)
        integrator.coef = simp_coef

        bform = BilinearForm(self.space_full)
        bform.add_integrator(integrator)

        K_tensor = bform.assembly()
        if hasattr(K_tensor, 'to_scipy'):
            K_full = K_tensor.to_scipy()
        elif hasattr(K_tensor, 'toarray'):
            K_full = sp.csr_matrix(K_tensor.toarray())
        else:
            K_full = sp.csr_matrix(bm.asarray(K_tensor))

        # 边界条件施加
        node_coords = self.full_mesh.entity('node')
        eps = 1e-7

        if bc_type == "mbb":
            # MBB 梁 (Half MBB Beam):
            # 左侧边界 (x=0) 施加对称约束: ux = 0
            # 右下角/右底线 (x=Lx, y=0) 施加简支约束: uy = 0
            left_nodes = [idx for idx, pt in enumerate(node_coords) if abs(pt[0]) < eps]
            rb_nodes = [idx for idx, pt in enumerate(node_coords) if abs(pt[0] - self.Lx) < eps and abs(pt[1]) < eps]
            fixed_dofs = bm.sort(bm.concat([
                bm.array([self.dim * n for n in left_nodes], dtype=bm.int64),
                bm.array([self.dim * n + 1 for n in rb_nodes], dtype=bm.int64)
            ]))
        else:
            # Cantilever: 左侧全固定 (x = 0)
            left_nodes = [idx for idx, pt in enumerate(node_coords) if abs(pt[0]) < eps]
            left_nodes_arr = bm.array(left_nodes, dtype=bm.int64)
            fixed_dofs = bm.sort(bm.concat([self.dim * left_nodes_arr + k for k in range(self.dim)]))

        F_full = bm.zeros(self.total_full_dofs, dtype=bm.float64)
        F_full[load_dof] = load_val

        free_dofs = bm.setdiff1d(bm.arange(self.total_full_dofs), fixed_dofs)
        K_free = K_full[free_dofs[:, None], free_dofs]
        F_free = F_full[free_dofs]

        U_free = spla.spsolve(K_free, F_free)

        U_full_ref = bm.zeros(self.total_full_dofs, dtype=bm.float64)
        U_full_ref[free_dofs] = U_free

        return U_full_ref, free_dofs

    def build_interface_dofs(self, sub_meshes: List[Any]) -> tuple:
        """
        收集所有子结构边界 DOF 的全局编号，构建接口系统的 DOF 集合。

        返回接口 DOF 数组和全局→接口的映射字典。
        """
        interface_set = set()
        n_sub_y = self.n_sub[1]

        for sub_idx, sub_mesh in enumerate(sub_meshes):
            # 根据 sub_idx 恢复 (sx, sy) 或 (sx, sy, sz)
            if self.dim == 2:
                sx = sub_idx // n_sub_y
                sy = sub_idx % n_sub_y
                sub_pos = (sx, sy)
            else:
                n_sub_yz = self.n_sub[1] * self.n_sub[2]
                sx = sub_idx // n_sub_yz
                rem = sub_idx % n_sub_yz
                sy = rem // self.n_sub[2]
                sz = rem % self.n_sub[2]
                sub_pos = (sx, sy, sz)

            # 获取该子结构的全局 DOF 编号
            sub_global_dofs = self.get_substructure_global_dofs(sub_pos, sub_mesh)
            # 取边界 DOF
            b_global = sub_global_dofs[sub_mesh.b_dofs]
            for gdof in b_global:
                interface_set.add(int(gdof))

        # 排序后创建接口 DOF 数组
        interface_global_dofs = bm.array(sorted(interface_set), dtype=bm.int64)
        # 创建反向映射
        global_to_interface = {
            int(gdof): idx for idx, gdof in enumerate(interface_global_dofs)
        }
        return interface_global_dofs, global_to_interface

    def solve_condensed_fea(
        self,
        densities: Union[List[Any], Any],
        sub_meshes: List[Any],
        condensors: List[Any],
        load_dof: int,
        load_val: float = -1.0,
        bc_type: str = "cantilever"
    ) -> Tuple[Any, Any]:
        """
        真缩聚求解：组装接口系统并求解。

        参数
        -------
        densities : 保留（暂未使用）
        sub_meshes : 所有子结构网格列表
        condensors : FEAStaticCondensation 对象列表，已调用 condense()
        load_dof : 荷载作用的全局 DOF 编号
        load_val : 荷载值
        bc_type : "cantilever" 或 "mbb"

        返回
        -------
        U_full : 完整位移向量（包括内部）
        interface_free : 接口系统中未固定的 DOF 索引
        """
        # Step 1: 构建接口 DOF
        interface_global_dofs, global_to_interface = self.build_interface_dofs(sub_meshes)
        n_interface = len(interface_global_dofs)

        # Step 2: 计算接口系统上的固定 DOF
        node_coords = self.full_mesh.entity('node')
        eps = 1e-7

        if bc_type == "mbb":
            if self.dim == 2:
                # 2D HalfMBB: ux=0 at x=0; uy=0 at (Lx,0)
                left_nodes = [idx for idx, pt in enumerate(node_coords) if abs(pt[0]) < eps]
                rb_nodes = [idx for idx, pt in enumerate(node_coords)
                           if abs(pt[0] - self.Lx) < eps and abs(pt[1]) < eps]
                fixed_dofs = bm.sort(bm.concat([
                    bm.array([2 * n for n in left_nodes], dtype=bm.int64),
                    bm.array([2 * n + 1 for n in rb_nodes], dtype=bm.int64),
                ]))
            else:
                # 3D FullMBBBeam3d (与路径 A 完全一致):
                #   ux=0 at (x=0, y=0)          左下底线
                #   uy=0 at (y=0) & (x=0 | x=Lx) 底部两端底线
                #   uz=0 at (y=0, z=Lz/2)       底面中心线
                z_mid = (self.domain_size[2]) / 2.0
                fixed_dofs = bm.sort(bm.concat([
                    bm.array([3 * n + 0 for n in range(len(node_coords))
                              if abs(node_coords[n, 0]) < eps and abs(node_coords[n, 1]) < eps],
                             dtype=bm.int64),
                    bm.array([3 * n + 1 for n in range(len(node_coords))
                              if (abs(node_coords[n, 0]) < eps or abs(node_coords[n, 0] - self.Lx) < eps)
                              and abs(node_coords[n, 1]) < eps],
                             dtype=bm.int64),
                    bm.array([3 * n + 2 for n in range(len(node_coords))
                              if abs(node_coords[n, 1]) < eps and abs(node_coords[n, 2] - z_mid) < eps],
                             dtype=bm.int64),
                ]))
        else:  # cantilever
            left_nodes = [idx for idx, pt in enumerate(node_coords) if abs(pt[0]) < eps]
            left_arr = bm.array(left_nodes, dtype=bm.int64)
            fixed_dofs = bm.sort(bm.concat(
                [self.dim * left_arr + k for k in range(self.dim)]
            ))

        # 过滤到接口系统
        interface_fixed_list = []
        for d in fixed_dofs:
            gd = int(d)
            if gd in global_to_interface:
                interface_fixed_list.append(global_to_interface[gd])
        interface_fixed = bm.array(sorted(interface_fixed_list), dtype=bm.int64)

        # Step 3: Scatter-add K_s 到 K_global
        K_global = sp.lil_matrix((n_interface, n_interface), dtype=bm.float64)
        n_sub_y = self.n_sub[1]

        for sub_idx, (sub_mesh, condensor) in enumerate(zip(sub_meshes, condensors)):
            # 恢复 (sx, sy) 或 (sx, sy, sz)
            if self.dim == 2:
                sx = sub_idx // n_sub_y
                sy = sub_idx % n_sub_y
                sub_pos = (sx, sy)
            else:
                n_sub_yz = self.n_sub[1] * self.n_sub[2]
                sx = sub_idx // n_sub_yz
                rem = sub_idx % n_sub_yz
                sy = rem // self.n_sub[2]
                sz = rem % self.n_sub[2]
                sub_pos = (sx, sy, sz)

            sub_global_dofs = self.get_substructure_global_dofs(sub_pos, sub_mesh)
            b_dofs_local = sub_mesh.b_dofs
            b_global = sub_global_dofs[b_dofs_local]
            # 映射到接口 DOF 索引
            b_interface = bm.array(
                [global_to_interface[int(g)] for g in b_global], dtype=bm.int64
            )
            n_b = len(b_dofs_local)

            # Scatter-add K_s
            K_s = condensor.K_s
            for i_local in range(n_b):
                i_glob = int(b_interface[i_local])
                for j_local in range(n_b):
                    j_glob = int(b_interface[j_local])
                    K_global[i_glob, j_glob] += K_s[i_local, j_local]

        K_global = K_global.tocsr()

        # Step 4: 构建右端项
        F_interface = bm.zeros(n_interface, dtype=bm.float64)
        if load_dof in global_to_interface:
            F_interface[global_to_interface[load_dof]] = load_val
        else:
            raise ValueError(f"load_dof {load_dof} not found on any substructure boundary")

        # Step 5: 施加 BC 并求解
        interface_free = bm.setdiff1d(bm.arange(n_interface), interface_fixed)
        K_free = K_global[interface_free[:, None], interface_free]
        F_free = F_interface[interface_free]

        u_b_free = spla.spsolve(K_free, F_free)

        u_b = bm.zeros(n_interface, dtype=bm.float64)
        u_b[interface_free] = u_b_free

        # Step 6: 恢复全场位移
        U_full = bm.zeros(self.total_full_dofs, dtype=bm.float64)

        # 写入接口位移
        for i_idx, gdof in enumerate(interface_global_dofs):
            U_full[int(gdof)] = u_b[i_idx]

        # 恢复内部位移
        for sub_idx, (sub_mesh, condensor) in enumerate(zip(sub_meshes, condensors)):
            if self.dim == 2:
                sx = sub_idx // n_sub_y
                sy = sub_idx % n_sub_y
                sub_pos = (sx, sy)
            else:
                n_sub_yz = self.n_sub[1] * self.n_sub[2]
                sx = sub_idx // n_sub_yz
                rem = sub_idx % n_sub_yz
                sy = rem // self.n_sub[2]
                sz = rem % self.n_sub[2]
                sub_pos = (sx, sy, sz)

            sub_global_dofs = self.get_substructure_global_dofs(sub_pos, sub_mesh)
            b_global = sub_global_dofs[sub_mesh.b_dofs]
            b_interface_indices = bm.array(
                [global_to_interface[int(g)] for g in b_global], dtype=bm.int64
            )
            u_sub_b = u_b[b_interface_indices]
            u_sub_i = condensor.recover(u_sub_b)
            U_full[sub_global_dofs[sub_mesh.i_dofs]] = u_sub_i

        return U_full, interface_free




