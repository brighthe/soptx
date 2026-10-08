"""固定最终物理密度的完整细网格验证, 不过滤或更新密度."""
from pathlib import Path
from time import perf_counter
import json
import hashlib
import gc
import numpy as np
from scipy.sparse.linalg import cg, LinearOperator
from soptx.backend import backend_manager as bm
from soptx.mesh import HexahedronMesh
from soptx.fem.analyzers import LagrangeFEMAnalyzer
from soptx.fem.matrix import build_csr_pattern, assemble_csr
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import FullMBBBeam3d

out = Path(__file__).resolve().parent
source = out.parent.parent / "outputs" / "linear_corner_mumps"
cfg = json.loads((source / 'config.json').read_text())
started = perf_counter()
def log(message):
    print(f'[{perf_counter()-started:.1f}s] {message}', flush=True)

bm.set_backend('numpy')
rho = np.load(source / 'density_final.npy')
grid = tuple(cfg['grid'])
assert rho.shape == grid and np.isfinite(rho).all() and rho.min() >= 0 and rho.max() <= 1
assert cfg['spacing'] == [1.0, 1.0, 1.0]
problem = FullMBBBeam3d(domain=tuple(cfg['domain']), P=cfg['load'], E=cfg['E0'], nu=cfg['nu'], support=cfg['support'], load_subdivisions=(grid[0], grid[2]))
material = IsotropicLinearElasticMaterial(youngs_modulus=cfg['E0'], poisson_ratio=cfg['nu'], hypothesis='3D', enable_logging=False)
def analyzer_for(mesh):
    return LagrangeFEMAnalyzer(disp_mesh=mesh, pde=problem, material=material, space_degree=1, integration_order=cfg['integration_order'], assembly_method='fast', operator_level='fa', solve_method='cg', enable_logging=False)

log('创建完整细网格及 Q1 空间')
mesh = HexahedronMesh.from_box(cfg['domain'], *grid)
analyzer = analyzer_for(mesh)
space = analyzer.tensor_space
assert not space.dof_priority
nodes = np.asarray(mesh.entity('node'))
cells = np.asarray(mesh.entity('cell'))
# 通过单元中心坐标显式映射物理密度, 不假定单元排列顺序.
centers = nodes[cells].mean(axis=1)
ijk = np.floor(centers).astype(np.int64)
rho_cell = rho[ijk[:,0], ijk[:,1], ijk[:,2]]
assert np.unique(np.ravel_multi_index(ijk.T, grid)).size == rho.size
del centers, ijk
small = HexahedronMesh.from_box([0,1,0,1,0,1], 1,1,1)
ke = np.asarray(analyzer_for(small).compute_solid_stiffness_matrix())
# 均匀单位立方体采用同一参考单元, 核对局部节点顺序.
reference_offsets = np.asarray(small.entity('node'))[np.asarray(small.entity('cell'))[0]]
for begin in range(0,len(cells),16384):
    xyz = nodes[cells[begin:begin+16384]]
    assert np.allclose(xyz-xyz.min(axis=1,keepdims=True),reference_offsets,rtol=0,atol=1e-12)
coef = cfg['Emin']/cfg['E0'] + (1-cfg['Emin']/cfg['E0'])*rho_cell**cfg['penalty']
log(f'构建 CSR 模式: {rho.size} 单元, {space.number_of_global_dofs()} 自由度')
pattern = build_csr_pattern(space)
log(f'数值装配: nnz={pattern.nnz}')
K = assemble_csr(np.broadcast_to(ke,(rho.size,24,24)),pattern,scale=coef)
log('施加原工况的载荷与齐次支承')
A_backend, force = analyzer.apply_bc(K, analyzer.assemble_body_force_vector())
A = A_backend.to_scipy()
b = np.asarray(force[:]).copy()
del K, A_backend, pattern
# 原恢复位移按全局结构化节点坐标映射到当前 Q1 节点.
node_grid = tuple(n+1 for n in grid)
node_index = np.rint(nodes).astype(np.int64)
flat = np.ravel_multi_index(node_index.T,node_grid)
u0 = np.load(source/'displacement_final.npy').reshape(-1,3)[flat].reshape(-1).copy()
fixed = np.stack([np.asarray(fn(nodes)) for fn in problem.is_dirichlet_boundary()],axis=1).reshape(-1)
assert np.max(np.abs(u0[fixed])) < 1e-10
u0[fixed]=0
normb = np.linalg.norm(b)
coarse = float(b@u0)
energy0 = float(u0@(A@u0))
expected = json.loads((source/'summary.json').read_text())['compliance']
assert abs(coarse-expected)/expected < 1e-7
assert abs(energy0-expected)/expected < 1e-7
log(f'恢复场独立核对: FTu={coarse:.12g}, uTKu={energy0:.12g}, 细网格残差={np.linalg.norm(b-A@u0)/normb:.3e}')
diag = A.diagonal()
assert np.all(diag>0)
M = LinearOperator(A.shape,matvec=lambda x:x/diag,dtype=np.float64)
gc.collect()
iteration = 0
progress = []
def monitor(u):
    global iteration
    iteration += 1
    if iteration % 50 == 0:
        row = dict(iteration=iteration,relative_residual=float(np.linalg.norm(b-A@u)/normb),compliance=float(b@u),seconds=perf_counter()-started)
        progress.append(row)
        (out/'progress.json').write_text(json.dumps(row,indent=2))
        log(str(row))
log('开始 Jacobi-PCG, 容差 1e-8, 最多 20000 步, 使用恢复位移初值')
u, info = cg(A,b,x0=u0,rtol=1e-8,atol=0,maxiter=20000,M=M,callback=monitor)
residual = float(np.linalg.norm(b-A@u)/normb)
fine = float(b@u)
energy = float(u@(A@u))
summary = dict(converged=bool(info==0 and residual<=1e-8),scipy_info=int(info),iterations=iteration,relative_residual=residual,compliance_fine=fine,compliance_linear_corner=coarse,compliance_ratio=fine/coarse,energy_relative_error=abs(energy-fine)/abs(fine),max_support_displacement=float(np.max(np.abs(u[fixed]))),volume_fraction=float(rho.mean()),density_sha256=hashlib.sha256((source/'density_final.npy').read_bytes()).hexdigest(),total_seconds=perf_counter()-started,grid=grid,support=cfg['support'])
np.save(out/'displacement_fine.npy',u)
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
(out/'convergence.json').write_text(json.dumps(progress,indent=2))
log(str(summary))
if not summary['converged']:
    raise RuntimeError('完整细网格求解未达到真实残差验收标准.')
