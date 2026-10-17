"""比较完整 Schur 投影与角点直接缩聚的数值结果和耗时."""
import argparse
import os
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from soptx.backend import backend_manager as bm
from soptx.mesh import HexahedronMesh
from soptx.fem.analyzers import SubstructureAnalyzer
from soptx.fem.substructure import StructuredSubstructureLayout
from soptx.fem.substructure.streaming import assemble_exact_interface_system
from soptx.materials import IsotropicLinearElasticMaterial
from soptx.problems import FullMBBBeamPoint3d
from soptx.topology.interpolation import MaterialInterpolationScheme
from soptx.topology.objectives import ComplianceObjective


def main():
    """在固定线程环境下验证两种缩聚路径, 独立保存报告."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iter94", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bm.set_backend("numpy")
    grid = (210, 35, 35) if args.iter94 else (60, 10, 10)
    domain = tuple(v for n in grid for v in (0., float(n)))
    layout = StructuredSubstructureLayout(
        domain_size=tuple(float(n) for n in grid), n_sub=tuple(n//5 for n in grid),
        n_fine=(5,5,5), degree=1, E_base=1., nu=.3, hypothesis="3D",
        full_mesh=HexahedronMesh.from_box(list(domain), *grid))
    material = IsotropicLinearElasticMaterial(youngs_modulus=1., poisson_ratio=.3, hypothesis="3D")
    interp = MaterialInterpolationScheme(density_location="element", interpolation_method="msimp",
        options=dict(penalty_factor=3., void_youngs_modulus=1e-7, target_variables=["E"]))
    densities = [("uniform", np.full(np.prod(grid), .12)),
                 ("random", np.random.default_rng(42).uniform(.05,1.,np.prod(grid)))]
    if args.iter94:
        import struct
        import xml.etree.ElementTree as ET
        source=Path(__file__).parent/"outputs/linear_corner_hex_210x35x35_m5_cg_jacobi/iterations/iter_0094.vtu"
        with source.open("rb") as f:
            head=f.read(1100); mark=head.index(b"<AppendedData")
            root=ET.fromstring(head[:mark]+b"</VTKFile>")
            start=head.index(b"_",mark)+1
            f.seek(start+int(root.find(".//CellData/DataArray").attrib["offset"]))
            size=struct.unpack("<Q",f.read(8))[0]
            rho=np.frombuffer(f.read(size),dtype="<f8").copy()
        densities=[("iter94",rho)]
    def analyzer():
        """为两条路径构造相同配置的分析器."""
        return SubstructureAnalyzer(layout=layout,
            pde=FullMBBBeamPoint3d(domain=domain,P=-1.,E=1.,nu=.3),
            material=material,trace="linear_corner",chunk_size=64,
            space_degree=1,integration_order=2,solve_method="scipy",
            topopt_algorithm="density_based",interpolation_scheme=interp)
    (args.output/"config.json").write_text(json.dumps(dict(
        grid=grid, n_fine=5, chunk_size=64, solver="scipy", repeats=3,
        thread_environment={k:os.environ.get(k) for k in
            ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","BLIS_NUM_THREADS")},
        note="耗时包含装配与接口求解/位移恢复, 不含边界条件、过滤、输出及 OC; 非完整优化计时."
    ),indent=2))
    old,new=analyzer(),analyzer()
    report=[]
    for label,rho in densities:
        for repeat in range(3):
            results={}
            for name,a in ([("old",old),("new",new)] if repeat%2==0 else [("new",new),("old",old)]):
                started=perf_counter()
                if name=="old":
                    a._update_density_coefficient(rho)
                    a._density_cells=a._to_density_cells(rho)
                    a._corner_recovery_batches=None
                    K=assemble_exact_interface_system(a.prototype,a._density_cells,a.interface_space,chunk_size=64)
                else:
                    K=a.assemble_stiff_matrix(rho)
                assembly=perf_counter()-started
                prepared,rhs=a.apply_bc(K,a.assemble_body_force_vector())
                out=a.tensor_space.function()
                started=perf_counter()
                _,info=a.solve_system(prepared,rhs,out)
                solve=perf_counter()-started
                objective=ComplianceObjective(analyzer=a,state_variable="u",diff_mode="manual")
                state={"displacement":out}
                c=float(objective.fun(density=rho,state=state))
                dc=np.asarray(objective.jac(density=rho,state=state)).copy()
                results[name]=(K.stiffness.to_scipy(),np.asarray(out[:]).copy(),c,dc,assembly,solve,info)
            x,y=results["old"],results["new"]
            assert x[6]["converged"] and y[6]["converged"]
            assert max(x[6]["relres"], y[6]["relres"]) < 1e-8
            errors=dict(stiffness=float(np.linalg.norm((y[0]-x[0]).data)/np.linalg.norm(x[0].data)),
                displacement=float(np.linalg.norm(y[1]-x[1])/np.linalg.norm(x[1])),
                compliance=float(abs(y[2]-x[2])/abs(x[2])),
                sensitivity=float(np.linalg.norm(y[3]-x[3])/np.linalg.norm(x[3])))
            assert max(errors.values())<1e-8,errors
            row=dict(case=label,repeat=repeat,relative_errors=errors,
                old_assembly=x[4],new_assembly=y[4],old_solve_recovery=x[5],new_solve_recovery=y[5],
                old_compliance=x[2],new_compliance=y[2],
                old_residual=x[6]["relres"],new_residual=y[6]["relres"],
                recovery_cache_bytes=sum(t.nbytes for _,_,t in new._corner_recovery_batches))
            report.append(row)
            (args.output/"report.json").write_text(json.dumps(report,indent=2))
            print(json.dumps(row),flush=True)


if __name__=="__main__":
    main()
