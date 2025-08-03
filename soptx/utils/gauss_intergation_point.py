from fealpy.backend import backend_manager as bm

def get_gauss_point_mapping(nx: int, ny: int, nq_per_dim: int = 3):
    """局部高斯积分点和全局高斯积分点的映射"""
    NC = nx * ny
    NQ = nq_per_dim * nq_per_dim
    
    # 创建全局到局部的映射
    global_to_local = bm.zeros(NC * NQ, dtype=bm.int32)
    
    global_idx = 0
    
    # 按照transpose后的维度顺序遍历
    for ix in range(nx):                    # 网格x方向
        for iq_x in range(nq_per_dim):      # 积分点x方向（列）
            for iy in range(ny):            # 网格y方向
                for iq_y in range(nq_per_dim):  # 积分点y方向（行）
                    
                    # 单元索引（列优先存储）
                    cell_idx = iy + ix * ny
                    
                    # 单元内积分点索引（行优先）
                    point_idx = iq_x * nq_per_dim + iq_y
                    
                    # 局部索引
                    local_idx = cell_idx * NQ + point_idx
                    
                    # 全局位置global_idx对应局部位置local_idx
                    global_to_local[global_idx] = local_idx
                    global_idx += 1
    
    # 计算局部到全局的映射
    local_to_global = bm.argsort(global_to_local)
    
    return global_to_local, local_to_global