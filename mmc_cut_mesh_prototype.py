import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import Delaunay
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

class MMCComponent:
    def __init__(self, xc, yc, L, W, theta_deg, m=4):
        self.xc = xc
        self.yc = yc
        self.L = L
        self.W = W
        self.theta = np.deg2rad(theta_deg)
        self.m = m

    def evaluate_tdf(self, x, y):
        """
        计算点 (x,y) 的 TDF (拓扑描述函数) 值。
        f(x,y) <= 0 表示实体 (Solid)，f(x,y) > 0 表示空洞 (Void)
        """
        dx = x - self.xc
        dy = y - self.yc
        
        term1 = (np.cos(self.theta)*dx + np.sin(self.theta)*dy) / (self.L / 2)
        term2 = (-np.sin(self.theta)*dx + np.cos(self.theta)*dy) / (self.W / 2)
        
        # 确保 m 为偶数时无需绝对值，但为稳妥起见加一个 abs
        f_val = np.power(np.abs(term1), self.m) + np.power(np.abs(term2), self.m) - 1.0
        return f_val

class MeshCutter:
    def __init__(self, nx, ny, lx, ly, component):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / nx
        self.dy = ly / ny
        self.component = component
        
        self.x_nodes = np.linspace(0, lx, nx + 1)
        self.y_nodes = np.linspace(0, ly, ny + 1)
        
        # 存储分类结果
        self.solid_cells = []
        self.void_cells = []
        self.cut_cells = []
        self.cut_polygons = [] # 用于存储被切割后的实体侧多边形

    def classify_and_cut(self):
        # 遍历所有单元
        for i in range(self.nx):
            for j in range(self.ny):
                x0, x1 = self.x_nodes[i], self.x_nodes[i+1]
                y0, y1 = self.y_nodes[j], self.y_nodes[j+1]
                
                # 单元四个角点 (按逆时针：左下，右下，右上，左上)
                corners = np.array([
                    [x0, y0],
                    [x1, y0],
                    [x1, y1],
                    [x0, y1]
                ])
                
                # 计算角点 TDF
                f_vals = np.array([self.component.evaluate_tdf(x, y) for x, y in corners])
                
                if np.all(f_vals <= 0):
                    self.solid_cells.append(corners)
                elif np.all(f_vals > 0):
                    self.void_cells.append(corners)
                else:
                    # 发生切割
                    self.cut_cells.append(corners)
                    polygon = self._extract_solid_polygon(corners, f_vals)
                    if polygon is not None:
                        self.cut_polygons.append(polygon)

    def _extract_solid_polygon(self, corners, f_vals):
        """利用线性插值重构切割单元内的实体多边形"""
        points = []
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        
        for i in range(4):
            # 如果原节点是实体，则保留
            if f_vals[i] <= 0:
                points.append(corners[i])
            
            # 检查边上是否有交点 (变号)
            a, b = edges[i]
            fa, fb = f_vals[a], f_vals[b]
            if fa * fb < 0:
                t = fa / (fa - fb)
                p = corners[a] + t * (corners[b] - corners[a])
                points.append(p)
                
        if len(points) < 3:
            return None
            
        points = np.array(points)
        # 去重
        _, idx = np.unique(points.round(decimals=8), axis=0, return_index=True)
        points = points[np.sort(idx)]
        
        if len(points) < 3:
            return None
            
        # 极角排序以确保凸多边形顶点顺序正确
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        points = points[np.argsort(angles)]
        return points

class IntegrationPointGenerator:
    def __init__(self):
        self.points = []
        self.weights = []

    def generate_for_solid(self, corners):
        """为完整实体单元分配 2x2 标准高斯积分点"""
        x0, x1 = corners[0, 0], corners[1, 0]
        y0, y1 = corners[0, 1], corners[2, 1]
        
        gp_1d = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
        wt_1d = np.array([1.0, 1.0])
        
        area_det = (x1 - x0)/2 * (y1 - y0)/2
        
        for i in range(2):
            for j in range(2):
                xg = (x0 + x1)/2 + (x1 - x0)/2 * gp_1d[i]
                yg = (y0 + y1)/2 + (y1 - y0)/2 * gp_1d[j]
                wg = wt_1d[i] * wt_1d[j] * area_det
                self.points.append([xg, yg])
                self.weights.append(wg)

    def generate_for_cut(self, polygon):
        """对切割得到的多边形进行三角化，并在每个三角形内布置高斯积分点 (3点求积)"""
        # 利用 Delaunay 进行三角化
        tri = Delaunay(polygon)
        
        for simplex in tri.simplices:
            pts = polygon[simplex]
            A, B, C = pts[0], pts[1], pts[2]
            
            # 计算三角形面积
            area = 0.5 * np.abs(A[0]*(B[1]-C[1]) + B[0]*(C[1]-A[1]) + C[0]*(A[1]-B[1]))
            if area < 1e-10:
                continue
                
            # 三点高斯积分法则 (各边中点，权重为 area/3)
            p1 = (A + B) / 2
            p2 = (B + C) / 2
            p3 = (C + A) / 2
            
            for p in [p1, p2, p3]:
                self.points.append(p)
                self.weights.append(area / 3.0)

def main():
    # 1. 设定参数 (对应 PPT 推断数据)
    nx, ny = 40, 20
    lx, ly = 2.0, 1.0
    comp = MMCComponent(xc=1.0, yc=0.5, L=1.2, W=0.4, theta_deg=30, m=4)
    
    # 2. 网格切割
    cutter = MeshCutter(nx, ny, lx, ly, comp)
    cutter.classify_and_cut()
    
    print(f"单元分类结果:")
    print(f"  Solid 单元: {len(cutter.solid_cells)}")
    print(f"  Void 单元: {len(cutter.void_cells)}")
    print(f"  Cut 单元: {len(cutter.cut_cells)}")
    
    # 3. 积分点生成
    ig = IntegrationPointGenerator()
    
    for cell in cutter.solid_cells:
        ig.generate_for_solid(cell)
        
    for poly in cutter.cut_polygons:
        ig.generate_for_cut(poly)
        
    pts_array = np.array(ig.points)
    print(f"总计生成高阶积分点: {len(pts_array)} 个")
    
    # 4. 可视化绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("MMC Explicit Geometry & Cut-Cell Integration (Frame 10)", fontsize=14, pad=15)
    
    # 绘制背景网格
    for x in cutter.x_nodes:
        ax.axvline(x, color='lightgray', linestyle='-', linewidth=0.5)
    for y in cutter.y_nodes:
        ax.axhline(y, color='lightgray', linestyle='-', linewidth=0.5)
        
    # 绘制 MMC 零等值线
    xx, yy = np.meshgrid(np.linspace(0, lx, 200), np.linspace(0, ly, 200))
    zz = comp.evaluate_tdf(xx, yy)
    ax.contour(xx, yy, zz, levels=[0], colors='blue', linewidths=2, label='TDF f(x,y)=0')
    
    # 绘制积分点
    ax.scatter(pts_array[:, 0], pts_array[:, 1], s=4, c='green', alpha=0.7, label='Gauss Integration Points')
    
    ax.set_xlim(0, lx)
    ax.set_ylim(0, ly)
    ax.set_aspect('equal')
    
    # 修复 legend warning
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, loc='upper left')
    
    # --- 添加局部放大图 (Inset) ---
    # 选择右上方的一块切割区域进行放大
    axins = zoomed_inset_axes(ax, zoom=4, loc='lower right', borderpad=2)
    
    # 放大图内部重绘内容
    for x in cutter.x_nodes:
        axins.axvline(x, color='lightgray', linestyle='-', linewidth=1)
    for y in cutter.y_nodes:
        axins.axhline(y, color='lightgray', linestyle='-', linewidth=1)
    axins.contour(xx, yy, zz, levels=[0], colors='blue', linewidths=3)
    axins.scatter(pts_array[:, 0], pts_array[:, 1], s=15, c='green')
    
    # 画出 Cut 单元的子三角多边形边缘以展示精确切分
    for poly in cutter.cut_polygons:
        poly_patch = Polygon(poly, closed=True, fill=False, edgecolor='red', linestyle='--', linewidth=0.8, alpha=0.5)
        axins.add_patch(poly_patch)
        
    # 设置放大的坐标域
    x1, x2, y1, y2 = 1.35, 1.6, 0.65, 0.8
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    
    # 隐藏坐标轴刻度
    axins.set_xticks([])
    axins.set_yticks([])
    
    # 连接放大区域
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    
    # 避免 warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
        
    plt.savefig("c:\\workspace\\soptx_heliang\\mmc_integration_result.png", dpi=300, bbox_inches='tight')
    print("验证图已保存为 c:\\workspace\\soptx_heliang\\mmc_integration_result.png")
    
if __name__ == "__main__":
    main()
