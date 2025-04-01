__version__ = '1.0.0'
__author__ = 'Liang He'
__all__ = ['pde', 'material', 'solver', 'filter', 'opt']

# 其他必要的初始化代码
def version():
    """返回软件版本信息"""
    return __version__

def import_all():
    """导入所有子模块"""
    from . import pde   
    from . import material
    from . import solver
    from . import filter
    from . import opt
    return locals()