import sys
import numpy as np
from types import ModuleType

def show_error_table(N, errorType, errorMatrix, 
        f='e', pre=4, sep=' & ',
        out=sys.stdout, end='\n'):

    flag = False
    if type(out) == type(''):
        flag = True
        out = open(out, 'w')

    string = ''
    n = errorMatrix.shape[1] + 1
    print('\\begin{table}[!htdp]', file=out, end='\n')
    print('\\begin{tabular}[c]{|'+ n*'c|' + '}\\hline', file=out, end='\n')

    s = 'Dof' + sep + np.array2string(N, separator=sep,
            )
    s = s.replace('\n', '')
    s = s.replace('[', '')
    s = s.replace(']', '')
    print(s, file=out, end=end)
    print('\\\\\\hline', file=out)

    n = len(errorType)
    ff = '%.'+str(pre)+f
    for i in range(n):
        first = errorType[i]
        line = errorMatrix[i]
        s = first + sep + np.array2string(line, separator=sep,
                precision=pre, formatter=dict( float = lambda x: ff % x ))
        
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        print(s, file=out, end=end)
        print('\\\\\\hline', file=out)

        order = np.log(line[0:-1]/line[1:])/np.log(2)
        s = 'Order' + sep + '--' + sep + np.array2string(order,
                separator=sep, precision=2)
        s = s.replace('\n', '')
        s = s.replace('[', '')
        s = s.replace(']', '')
        print(s, file=out, end=end)
        print('\\\\\\hline', file=out)

    print('\\end{tabular}', file=out, end='\n')
    print('\\end{table}', file=out, end='\n')

    if flag:
        out.close()

def showmultirate(plot, k, N, errorMatrix, labellist, optionlist=None, lw=1,
        ms=4, propsize=10, computerate=True):
    """
    在同一张图上绘制多条误差收敛曲线，并自动计算和显示收敛阶.
    
    该函数主要用于有限元方法等数值计算中, 展示不同类型的误差 (如 L2 误差、H1 误差等)
    随着网格加密或自由度增加的收敛情况. 函数会自动在对数坐标系下绘制误差曲线,
    并可选择性地计算和显示数值收敛阶. 
    
    Parameters
    ----------
    plot : matplotlib.pyplot 模块或 matplotlib.axes.Axes 对象
        - 如果传入 matplotlib.pyplot 模块，函数会创建新的图形
        - 如果传入 axes 对象，会在现有的坐标轴上绘图
    
    k : int 或 array-like
        用于计算收敛阶的起始索引位置
        - 如果是整数：从第 k 个数据点开始计算收敛阶（适用于前几个点可能不稳定的情况）
        - 如果是数组：指定用于拟合的数据点索引
    
    N : 1D array 或 2D array
        横坐标数据，表示网格数量、自由度数或网格尺寸
        - 1D array：所有误差曲线共用相同的横坐标
        - 2D array：每条误差曲线有独立的横坐标（每行对应一条曲线）
    
    errorMatrix : 2D array, shape (m, n)
        误差数据矩阵
        - m：误差类型的数量（如L2误差、H1误差等）
        - n：每种误差的数据点数量
        - errorMatrix[i, j] 表示第 i 种误差在第 j 个网格下的值
    
    labellist : list of strings
        每条误差曲线的标签列表，长度应等于 errorMatrix 的行数
        例如：['$|| u - u_h ||_{L^2}$', '$|| \\nabla(u - u_h) ||_{L^2}$']
    
    optionlist : list of strings, optional
        每条曲线的绘图样式字符串列表
        - 默认提供12种不同的线型、颜色和标记组合
        - 格式如 'k-*' 表示：黑色(k)、实线(-)、星号标记(*)
    
    lw : float, default=1
        线宽（line width）
    
    ms : float, default=4
        标记大小（marker size）
    
    propsize : int, default=10
        图例字体大小
    
    computerate : bool, default=True
        是否计算并显示收敛阶
        - True：会在图上添加一条拟合线，并在图例中显示收敛阶
        - False：仅绘制原始误差曲线
    
    Returns
    -------
    axes : matplotlib.axes.Axes
        返回绘图使用的坐标轴对象，可用于进一步自定义
    """
    # 判断 plot 参数类型，创建或获取绘图坐标轴
    if isinstance(plot, ModuleType):
        # 如果传入的是 matplotlib.pyplot 模块，创建新图形
        fig = plot.figure()
        fig.set_facecolor('white')  # 设置图形背景为白色
        axes = fig.gca()  # 获取当前坐标轴
    else:
        # 如果传入的已经是 axes 对象，直接使用
        axes = plot
    
    # 设置默认的绘图样式列表
    if optionlist is None:
        optionlist = ['k-*', 'r-o', 'b-D', 'g-->', 'k--8', 'm--x',
                      'r-.x', 'b-.+', 'b-.h', 'm:s', 'm:p', 'm:h']
        # 格式说明：颜色-线型-标记
        # 颜色：k(黑)、r(红)、b(蓝)、g(绿)、m(品红)
        # 线型：-(实线)、--(虚线)、-.(点划线)、:(点线)
        # 标记：*(星)、o(圆)、D(菱形)、>(三角)等

    # 获取误差矩阵的维度
    m, n = errorMatrix.shape  # m: 误差类型数量, n: 数据点数量
    
    # 循环绘制每种误差的收敛曲线
    for i in range(m):
        if len(N.shape) == 1:
            # 如果 N 是一维数组，所有误差曲线共用相同的横坐标
            showrate(axes, k, N, errorMatrix[i], optionlist[i], 
                    label=labellist[i], lw=lw, ms=ms, computerate=computerate)
        else:
            # 如果 N 是二维数组，每条曲线有独立的横坐标
            showrate(axes, k, N[i], errorMatrix[i], optionlist[i], 
                    label=labellist[i], lw=lw, ms=ms, computerate=computerate)
    
    # 添加图例
    axes.legend(loc=3,              # 位置：左下角
                framealpha=0.2,     # 图例框透明度
                fancybox=True,      # 圆角边框
                prop={'size': propsize})  # 字体大小

    return axes

def showrate(axes, k, N, error, option, label=None, lw=1, ms=4, computerate=True):
    """
    在对数坐标系下绘制单条误差收敛曲线，并可选择性地计算和显示收敛阶。
    
    该函数是 showmultirate 的核心辅助函数，用于绘制单条误差曲线。
    它会自动判断横坐标是网格数量（整数）还是网格尺寸（浮点数），
    并相应地调整显示格式和计算方式。
    
    Parameters
    ----------
    axes : matplotlib.axes.Axes
        绘图使用的坐标轴对象
    
    k : int 或 array-like
        用于计算收敛阶的数据点选择
        - int: 从第 k 个点开始的所有数据用于拟合
        - array: 指定用于拟合的数据点索引列表
    
    N : 1D array
        横坐标数据
        - 整数数组：通常表示网格数量或自由度
        - 浮点数数组：通常表示网格尺寸 h
    
    error : 1D array
        误差数据，与 N 长度相同
    
    option : str
        绘图样式字符串，如 'k-*' (黑色实线带星号标记)
    
    label : str, optional
        曲线标签，用于图例显示
    
    lw : float, default=1
        线宽
    
    ms : float, default=4
        标记大小
    
    computerate : bool, default=True
        是否计算并显示收敛阶
    
    工作原理
    --------
    1. 首先绘制原始误差曲线（对数-对数坐标）
    
    2. 如果 computerate=True：
       - 使用最小二乘法拟合 log(error) vs log(N)
       - 拟合得到的斜率即为收敛阶
       - 在图上添加一条拟合直线，显示理论收敛趋势
    
    3. 根据 N 的数据类型调整显示：
       - 整数类型：显示为 $CN^{-p}$ 格式（N 通常是网格数或自由度）
       - 浮点类型：显示为 $Ch^{p}$ 格式（h 通常是网格尺寸）
    
    收敛阶的意义
    -----------
    - 对于有限元方法，理论收敛阶反映了数值解的精度
    - L2 范数误差通常有 O(h^{p+1}) 的收敛阶
    - H1 半范数误差通常有 O(h^p) 的收敛阶
    - 其中 p 是有限元空间的多项式次数
    """
    # 根据 N 的数据类型确定显示格式
    # 整数类型通常表示网格数 N，浮点类型通常表示网格尺寸 h
    pres = '$CN^' if isinstance(N[0], np.int_) else  '$Ch^'
    
    # 绘制原始误差曲线（双对数坐标）
    line0, = axes.loglog(N, error, option, lw=lw, ms=ms, label=label)
    
    if computerate:
        if isinstance(k, int):
            # k 是整数：从第 k 个点开始的所有数据用于拟合
            # 使用 polyfit 在对数空间进行线性拟合
            c = np.polyfit(np.log(N[k:]), np.log(error[k:]), 1)
            # c[0] 是拟合直线的斜率，即收敛阶
            # c[1] 是截距，对应 log(C)
            
            # 计算拟合直线的起始值
            # 0.75 是缩放因子，使拟合线稍微偏下，避免与原曲线重叠
            s = 0.75*error[k]/N[k]**c[0]
            
            # 绘制拟合直线，标签显示收敛阶
            line1, = axes.loglog(N[k:], s*N[k:]**c[0], 
                    label=pres+'{%0.2f}$'%(c[0]),  # 如 $Ch^{2.00}$
                    lw=lw, 
                    ls=line0.get_linestyle(),  # 使用相同线型
                    color=line0.get_color())   # 使用相同颜色
        else:
            # k 是数组：只使用指定索引的数据点进行拟合
            c = np.polyfit(np.log(N[k]), np.log(error[k]), 1)
            s = 0.75*error[k[0]]/N[k[0]]**c[0]
            line1, = axes.loglog(N[k], s*N[k]**c[0], 
                    label=pres+'{%0.2f}$'%(c[0]),
                    lw=lw, 
                    ls=line0.get_linestyle(), 
                    color=line0.get_color())

    # 设置坐标轴范围和格式
    if isinstance(N[0], np.int_):
        # 对于整数类型（网格数），横轴从 N[0]/2 到 N[-1]*2
        axes.set_xlim(left=N[0]/2, right=N[-1]*2)
    elif isinstance(N[0], np.float64):
        # 对于浮点类型（网格尺寸 h）
        from matplotlib.ticker import LogLocator, NullFormatter
        
        # 注意：h 通常是递减的，所以 N[-1] 是最小值，N[0] 是最大值
        axes.set_xlim(left=N[-1]/1.2, right=N[0]*1.2)
        
        # 设置 x 轴为以 2 为底的对数坐标
        axes.set_xscale("log", base=2) 
        
        # 不显示次要刻度的标签
        axes.xaxis.set_minor_formatter(NullFormatter())
        
        # 设置次要刻度位置（在主刻度之间均匀分布）
        minor_locator = LogLocator(base=2, subs=2**np.linspace(-1, 0, 10))
        axes.xaxis.set_minor_locator(minor_locator)