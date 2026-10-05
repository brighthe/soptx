# 移植自 brighthe/fealpy ``fealpy/functionspace/tensor_space.py`` @ f474a5775.
# FEALPy Copyright (C) Huayi Wei, GPL-3.0-or-later; 此后以 SOPTX 本文件为准演化.
"""由标量空间构造的张量值函数空间."""

from typing import Tuple, Union, Callable,Optional
from math import prod

from ..backend import backend_manager as bm
from ..typing import TensorLike, Size, _S
from .functional import generate_tensor_basis, generate_tensor_grad_basis
from .space import FunctionSpace, _S, Index, Function
from .utils import to_tensor_dof
from ..decorator import barycentric, cartesian


class TensorFunctionSpace(FunctionSpace):
    """由标量空间构造的张量值函数空间, 每个标量自由度携带一个张量分量.

    Parameters
    ----------
    scalar_space : FunctionSpace
        标量空间.
    shape : tuple of int
        自由度形状, 首或末元素须为 ``-1`` 以标明自由度的排列优先级: 末元素为
        ``-1`` (如 ``(2, -1)``) 时自由度优先, 即同一分量的全部自由度相邻; 首元素
        为 ``-1`` (如 ``(-1, 2)``) 时分量优先.

    Attributes
    ----------
    dof_shape : tuple of int
        去掉 ``-1`` 后的分量形状.
    dof_priority : bool
        是否自由度优先.

    Raises
    ------
    ValueError
        ``shape`` 少于两个元素, 或首末都不是 ``-1``.
    """
    def __init__(self, scalar_space: FunctionSpace, shape: Tuple[int, ...]) -> None:
        self.scalar_space = scalar_space
        self.shape = shape

        if len(shape) < 2:
            raise ValueError('shape must be a tuple of at least two element')

        if shape[0] == -1:
            self.dof_shape = tuple(shape[1:])
            self.dof_priority = False
        elif shape[-1] == -1:
            self.dof_shape = tuple(shape[:-1])
            self.dof_priority = True
        else:
            raise ValueError('`-1` is required as the first or last element')
        
        self.p = self.scalar_space.p

    @property
    def mesh(self):
        """标量空间的网格."""
        return self.scalar_space.mesh

    @property
    def device(self):
        """标量空间的设备."""
        return self.scalar_space.device
    @property
    def ftype(self):
        """标量空间的浮点类型."""
        return self.scalar_space.ftype
    @property
    def itype(self):
        """标量空间的整数类型."""
        return self.scalar_space.itype

    @property
    def dof_numel(self) -> int:
        """每个标量自由度的分量数."""
        return prod(self.dof_shape)

    @property
    def dof_ndim(self) -> int:
        """分量形状的维数."""
        return len(self.dof_shape)

    def number_of_global_dofs(self) -> int:
        """全局自由度个数, 即分量数乘以标量自由度数."""
        return self.dof_numel * self.scalar_space.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='cell') -> int:
        """``doftype`` 类实体上的局部自由度个数."""
        return self.dof_numel * self.scalar_space.number_of_local_dofs(doftype)

    def basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike:
        """单元积分点处的张量基函数, 形状 ``(1, NQ, ldof*numel, *dof_shape)``."""
        phi = self.scalar_space.basis(p, index, **kwargs) # (NC, NQ, ldof)
        return generate_tensor_basis(phi, self.dof_shape, self.dof_priority)
    
    def face_basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike:
        """面积分点处的张量面基函数."""
        phi = self.scalar_space.face_basis(p, index, **kwargs)
        return generate_tensor_basis(phi, self.dof_shape, self.dof_priority)


    def grad_basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike:
        """单元积分点处的张量基函数梯度, 形状 ``(NC, NQ, ldof*numel, *dof_shape, GD)``."""
        gphi = self.scalar_space.grad_basis(p, index, **kwargs)
        return generate_tensor_grad_basis(gphi, self.dof_shape, self.dof_priority)
    
     
    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        """单元到张量自由度的映射, 形状 ``(NC, ldof*dof_numel)``."""
        return to_tensor_dof(
            self.scalar_space.cell_to_dof(),
            self.dof_numel,
            self.scalar_space.number_of_global_dofs(),
            self.dof_priority
        )[index]

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        """面到张量自由度的映射, 形状 ``(NF, ldof*dof_numel)``."""
        return to_tensor_dof(
            self.scalar_space.face_to_dof(),
            self.dof_numel,
            self.scalar_space.number_of_global_dofs(),
            self.dof_priority
        )[index]
    
    def edge_to_dof(self, index: Index=_S) -> TensorLike:
        """边到张量自由度的映射, 形状 ``(NE, ldof*dof_numel)``."""
        return to_tensor_dof(
            self.scalar_space.edge_to_dof(),
            self.dof_numel,
            self.scalar_space.number_of_global_dofs(),
            self.dof_priority
        )[index]
    
    def entity_to_dof(self, etype: int, index: Index=_S):
        """按拓扑维数取单元、面或边到张量自由度的映射."""
        TD = self.mesh.top_dimension()
        if etype == TD:
            return self.cell_to_dof(index=index)
        elif etype == TD-1:
            return self.face_to_dof(index=index)
        elif etype == 1:
            return self.edge_to_dof(index=index)
        else:
            raise ValueError(f"Unknown entity type: {etype}")

    def interpolation_points(self) -> TensorLike:
        """标量空间的插值点坐标."""

        return self.scalar_space.interpolation_points()
    
    def interpolate(self, u: Union[Callable[..., TensorLike], TensorLike], ) -> TensorLike:
        """把张量值函数插值到空间中, 按 ``dof_priority`` 排列后展平."""

        if self.dof_priority:
            uI = self.scalar_space.interpolate(u)
            ndim = len(self.shape)
            uI = bm.swapaxes(uI, ndim-1, ndim-2) 
        else:
            uI = self.scalar_space.interpolate(u)   

        return self.function(uI.reshape(-1))
    
    def is_boundary_dof(self, threshold=None, method='interp') -> TensorLike:
        """标记边界上的张量自由度.

        Parameters
        ----------
        threshold : TensorLike, callable, tuple or None, optional
            长度为全局自由度数的布尔张量时原样返回; None 或函数时由标量空间判定
            后扩展到全部分量; 元组时按分量分别给出标量空间的筛选条件 (仅支持
            向量值空间).
        method : str, optional
            传给标量空间的判定方式, 默认 'interp'.

        Returns
        -------
        TensorLike
            长度为全局自由度数的布尔张量.

        Raises
        ------
        ValueError
            ``threshold`` 类型未知, 或张量长度不等于全局自由度数.
        """
        scalar_space = self.scalar_space

        scalar_gdof = scalar_space.number_of_global_dofs()
        if bm.is_tensor(threshold):
            index = threshold
            if (index.dtype == bm.bool) and (len(index) == self.number_of_global_dofs()):
                return index
            else:
                raise ValueError("len(threshold) must equal tensorspace gdof")
        elif (threshold is None) | (callable(threshold)) :
            scalar_is_bd_dof = scalar_space.is_boundary_dof(threshold, method=method)

            if  self.dof_priority:
                is_bd_dof = bm.reshape(scalar_is_bd_dof, (-1,) * self.dof_ndim + (scalar_gdof,))
                is_bd_dof = bm.broadcast_to(is_bd_dof, self.dof_shape + (scalar_gdof,))
            else:
                is_bd_dof = bm.reshape(scalar_is_bd_dof, (scalar_gdof,) + (-1,) * self.dof_ndim)
                is_bd_dof = bm.broadcast_to(is_bd_dof, (scalar_gdof,) + self.dof_shape)
        elif isinstance(threshold, tuple):
            ### 只处理了向量型的 tensorspace 空间
            assert self.dof_numel == len(threshold)
            scalar_is_bd_dof = [scalar_space.is_boundary_dof(i, method=method) for i in threshold] 
            if self.dof_priority:
                return bm.concatenate(scalar_is_bd_dof)
            else:
                return bm.stack(scalar_is_bd_dof, axis=1).flatten()                        
        else:
            raise ValueError(f"Unknown type of threshold {type(threshold)}")
        return is_bd_dof.reshape(-1)   
   
    def boundary_interpolate(self,
        gd: Union[Callable, int, float, TensorLike],
        uh: Optional[TensorLike]=None,
        *,
        threshold: Union[Callable, TensorLike, None]=None, method=None) -> TensorLike:
        """在边界张量自由度上设置第一类 (Dirichlet) 边界值.

        Parameters
        ----------
        gd : int, float, TensorLike, Function or callable
            边界值: 常数、长度为全局自由度数的张量或函数, 或在插值点处求值、
            末轴为分量的函数.
        uh : TensorLike, optional
            写入边界值的有限元函数, 默认新建.
        threshold : TensorLike, callable, tuple or None, optional
            边界自由度的筛选条件, 见 ``is_boundary_dof``.
        method : str, optional
            传给 ``is_boundary_dof`` 的判定方式.

        Returns
        -------
        tuple
            ``(uh, isTensorBDof)``: 写入边界值后的函数与边界自由度的布尔掩码.

        Raises
        ------
        ValueError
            ``gd`` 或 ``threshold`` 类型未知.

        Notes
        -----
        ``gd`` 为函数且 ``threshold`` 为 None 或函数时, 分支只算出边界值, 由函数末尾
        if 链之后的赋值写入 ``uh``; 这是各分析器走的主路径.
        """
        ipoints = self.interpolation_points()
        scalar_space = self.scalar_space
        if uh is None:
            uh = self.function()
        # 根据不同gd类型进行判断
        if isinstance(gd, (int, float)):
            if bm.is_tensor(threshold):
                assert len(threshold) == self.number_of_global_dofs()
                isTensorBDof = threshold
            else : # threshold 为函数或 None
                isTensorBDof = self.is_boundary_dof(threshold=threshold, method=method)
            uh[isTensorBDof] = gd
            return uh, isTensorBDof

        elif (bm.is_tensor(gd)) or (isinstance(gd, Function)):
            assert len(gd[:]) == self.number_of_global_dofs()
            if bm.is_tensor(threshold):
                assert len(threshold) == self.number_of_global_dofs()
                isTensorBDof = threshold
            else : # threshold 为函数或 None
                isTensorBDof = self.is_boundary_dof(threshold=threshold, method=method)
            uh[isTensorBDof] = gd[isTensorBDof]
            return uh, isTensorBDof

        elif callable(gd):
            if bm.is_tensor(threshold):
                assert len(threshold) == self.number_of_global_dofs()
                gd_tensor = gd(ipoints)
                assert gd_tensor.shape[-1] == self.dof_numel
                isTensorBDof = threshold
                if  self.dof_priority:
                    uh[:] = bm.set_at(uh[:], isTensorBDof, gd_tensor.T.reshape(-1)[isTensorBDof])
                else:
                    uh[:] = bm.set_at(uh[:], isTensorBDof, gd_tensor.reshape(-1)[isTensorBDof])
                return uh, isTensorBDof

            elif (threshold is None) | (callable(threshold)):
                isScalarBDof = scalar_space.is_boundary_dof(threshold=threshold, method=method)
                gd_tensor = gd(ipoints[isScalarBDof])
                assert gd_tensor.shape[-1] == self.dof_numel
                isTensorBDof = self.is_boundary_dof(threshold = threshold, method=method)

            elif isinstance(threshold, tuple):
                assert len(threshold) == self.dof_numel
                isScalarBDof = [scalar_space.is_boundary_dof(i, method=method) for i in threshold]
                gd_tensor = [gd(ipoints[isScalarBDof[i]])[..., i] for i in range(self.dof_numel)]
                isTensorBDof = self.is_boundary_dof(threshold=threshold, method=method)
                if self.dof_priority:
                    gd_tensor = bm.concatenate(gd_tensor)
                else:
                    # ! 注意这里的效率没有 dof_priority 的高
                    scalar_gdof = scalar_space.number_of_global_dofs()
                    full_values = bm.zeros((scalar_gdof, self.dof_numel), 
                                            dtype=bm.float64, device=self.device)
                    
                    for i in range(self.dof_numel):
                        indices = bm.where(isScalarBDof[i])[0]
                        full_values = bm.set_at(full_values, (indices, i), gd_tensor[i])
                    
                    gd_tensor = full_values.reshape(-1)
                    gd_tensor = gd_tensor[isTensorBDof]

                    # gd_tensor = bm.stack(gd_tensor, axis=1).flatten()
                    # scalar_gdof = scalar_space.number_of_global_dofs()
                    # # 使用列表推导式重新排列数据
                    # gd_values = []
                    # for i in range(scalar_gdof):
                    #     for j in range(self.dof_numel):
                    #         # 从 gd_tensor[j] 中获取第 i 个标量基函数的值
                    #         gd_values.append(gd_tensor[j][i] if i < len(gd_tensor[j]) else 0.0)
                    # gd_tensor = bm.array(gd_values, device=self.device)
                    # gd_tensor = gd_tensor[isTensorBDof]
                uh[:] = bm.set_at(uh[:], isTensorBDof, gd_tensor)
                return uh, isTensorBDof
            else:
                raise ValueError(f"Unknown type of threshold {type(threshold)}")
        else:
            raise ValueError(f"Unknown type of gd {type(gd)}")

        if  self.dof_priority:
            uh[:] = bm.set_at(uh[:], isTensorBDof, gd_tensor.T.reshape(-1))
            
        else:
            uh[:] = bm.set_at(uh[:], isTensorBDof, gd_tensor.reshape(-1))
        return uh, isTensorBDof

    
    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        """有限元函数在积分点处的值, 形状 ``(NC, NQ, *dof_shape)``."""
        if isinstance(bc, tuple):
            TD = sum(item.shape[-1] - 1 for item in bc)
        else :
            TD = bc.shape[-1] - 1
        phi = self.basis(bc, index=index)
        e2dof = self.entity_to_dof(TD, index=index)
        val = bm.einsum('cql..., cl... -> cq...', phi, uh[e2dof, ...])
        return val
    
    @barycentric
    def grad_value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        """有限元函数在积分点处的梯度, 形状 ``(NC, NQ, *dof_shape, GD)``."""
        if isinstance(bc, tuple):
            TD = sum(item.shape[-1] - 1 for item in bc)
        else :
            TD = bc.shape[-1] - 1
        gphi = self.grad_basis(bc, index=index)
        e2dof = self.entity_to_dof(TD, index=index)
        val = bm.einsum('cqlmn..., cl... -> cqmn', gphi, uh[e2dof, ...])
        return val[...]
