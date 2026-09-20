"""固定尺度 AL 对照诊断, 不改变正式求解器及原始验收指标.

用法: python diagnostic_al_controls.py verify --output <新目录>
      python diagnostic_al_controls.py run --method lfem --group A --output <新目录>
A/B 使用原始约束, C/D 使用初始 dg/dr 固定尺度; A/C 内层5步, B/D内层20步.
投影 beta 固定为1; mu每累计20步乘alpha**4; 总预算120步.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from config import bootstrap_source_path
bootstrap_source_path()
from metrics import case_parameters
from pipeline import build_stress_config, build_stress_pipeline
from soptx.topology.objectives import AugmentedLagrangianObjective
from soptx.topology.optimizers import ALMMMAOptimizer


class FixedScaleAL(AugmentedLagrangianObjective):
    """复用原始约束灵敏度, 仅对 AL 目标施加固定正尺度."""

    def __init__(self, original, scale):
        super().__init__(original._volume_objective, original._stress_constraint,
                         original._options, enable_logging=False)
        self.scale = np.array(scale, dtype=float, copy=True)
        if not np.all(np.isfinite(self.scale)) or np.any(self.scale <= 0):
            raise ValueError('尺度必须为有限正数')
        self.scale.setflags(write=False)

    def fun(self, density, state=None, **kwargs):
        f = self._volume_objective.fun(density, state)
        g = self._stress_constraint.fun(density, state)
        if g.shape != self.scale.shape:
            raise ValueError('尺度与约束形状不一致')
        scaled_h = np.maximum(g / self.scale, -self.lamb / self.mu)
        self._cache_g = g
        self._cache_h = self.scale * scaled_h
        return float(f + np.mean(self.lamb * scaled_h + .5 * self.mu * scaled_h**2))

    def jac(self, density, state=None, **kwargs):
        # 转回原始 g 的等效系数, 复用其完整伴随及材料导数.
        self.fun(density, state)
        lamb, mu = self.lamb, self.mu
        try:
            self.lamb = lamb / self.scale
            self.mu = mu / self.scale**2
            return super().jac(density, state, diff_mode='manual')
        finally:
            self.lamb, self.mu = lamb, mu

    def update_multipliers(self):
        self.lamb = np.maximum(0., self.lamb + self.mu * self._cache_g / self.scale)
        # mu由累计 MMA 步数调度, 不按外层轮数增长.


class ControlledOptimizer(ALMMMAOptimizer):
    """对齐两种分组的罚因子调度及渐近线初始化时段."""

    def _update_penalty(self, iter_idx):
        self.diag_step = iter_idx + 1
        self._al_objective.mu = min(self.options.mu_0 * self.options.alpha**(4 * (iter_idx // 20)),
                                    self.options.mu_max)

    def _solve_unconstrained_subproblem(self, dfdz, z, zold1, zold2):
        self._epoch = (self.diag_step - 1) // 5 + 1
        grad = np.asarray(dfdz)
        self.diagnostics.append(dict(before_step=self.diag_step,
            projected_gradient_inf=float(np.max(abs(z - np.clip(z - grad, 0, 1)))),
            mu=float(self._al_objective.mu)))
        return super()._solve_unconstrained_subproblem(dfdz, z, zold1, zold2)

    def _accept_mma_step(self, *args, **kwargs):
        result = super()._accept_mma_step(*args, **kwargs)
        self.last_design = np.array(result[0], copy=True)
        return result


def build(method, scaled, inner=5):
    p = case_parameters()
    p.update(max_al_iterations=120//inner, mma_iters_per_al=inner)
    pipe = build_stress_pipeline(build_stress_config(p), p, method, 2)
    old = pipe.optimizer
    filt = old._filter
    rho = filt.get_initial_density(density=pipe.density_distribution)
    state = pipe.analyzer.solve_state(rho_val=rho)
    g = np.asarray(pipe.stress_constraint.fun(rho, state))
    if method == 'lfem':
        # 保留原多项式缩放公式, 从公共接口获取未加权实体应力偏差.
        dev = np.asarray(
            pipe.stress_constraint.compute_solid_stress_ratio(rho, state)
        ).reshape(g.shape) - 1.0
        scale = np.asarray(state['stiffness_ratio']).reshape(g.shape) * (3*dev*dev+1)
    else:
        scale = np.asarray(state['eta_threshold']).reshape(g.shape)
    # 去除共同量级, 避免把全局罚强度变化误判为单元缩放效果.
    scale = scale / np.median(scale)
    al = FixedScaleAL(pipe.al_objective, scale if scaled else np.ones_like(g))
    opt = ControlledOptimizer(al, filt, old.options, enable_logging=False)
    opt.diagnostics = []
    filt.continuation_step = lambda change: (change, False)
    pipe.optimizer, pipe.al_objective = opt, al
    return pipe


def evaluate(pipe, z, gradient=False):
    rho = pipe.optimizer._filter.filter_design_variable(
        design_variable=z, physical_density=np.empty_like(z))
    state = pipe.analyzer.solve_state(rho_val=rho)
    value = pipe.al_objective.fun(rho, state)
    if not gradient:
        return value
    grad = pipe.al_objective.jac(rho, state)
    grad = pipe.optimizer._filter.filter_objective_sensitivities(design_variable=z, obj_grad_rho=grad)
    return value, np.array(grad, copy=True)


def verify(out):
    records = []
    for method in ('lfem', 'huzhang'):
        pipe = build(method, True)
        al = pipe.al_objective
        # 非零乘子覆盖线性罚项和截断激活集.
        al.lamb[:] = .1
        scale_before = al.scale.copy()
        print(json.dumps(dict(method=method,scale_min=float(al.scale.min()),
            scale_median=float(np.median(al.scale)),scale_max=float(al.scale.max()))),flush=True)
        rng = np.random.default_rng(20260909)
        for label in ('uniform', 'nonuniform'):
            z = np.full(len(pipe.design_variable), .5)
            if label == 'nonuniform':
                z += rng.uniform(-.15, .15, z.shape)
            value, grad = evaluate(pipe, z, True)
            directions = [rng.normal(size=z.shape), grad.copy()]
            for direction_index, direction in enumerate(directions):
                direction /= np.max(abs(direction))
                analytical = float(grad @ direction)
                errors = []
                for h in (1e-3, 1e-4, 1e-5):
                    numerical = (evaluate(pipe, z+h*direction)-evaluate(pipe, z-h*direction))/(2*h)
                    error = abs(numerical-analytical)/max(abs(numerical), abs(analytical), 1e-8)
                    errors.append(dict(h=h, numerical=numerical, relative_error=error))
                row = dict(method=method, state=label, direction=direction_index,
                           analytical=analytical, samples=errors,
                           passed=min(e['relative_error'] for e in errors) < 1e-3)
                records.append(row)
                print(json.dumps(row), flush=True)
            assert np.array_equal(scale_before, al.scale)
        # 单位尺度必须复现原AL目标及梯度.
        al.scale = np.ones_like(al.scale)
        z = np.full(len(pipe.design_variable), .5)
        value, grad = evaluate(pipe, z, True)
        native = AugmentedLagrangianObjective(al._volume_objective, al._stress_constraint,
                                              al._options, initial_lambda=al.lamb.copy(), enable_logging=False)
        native.mu = al.mu
        pipe.al_objective = native
        value0, grad0 = evaluate(pipe, z, True)
        assert np.isclose(value,value0,rtol=1e-12,atol=1e-12)
        assert np.allclose(grad,grad0,rtol=1e-10,atol=1e-12)
    (out/'gradient-check.json').write_text(json.dumps(records,indent=2))
    if not all(row['passed'] for row in records):
        raise RuntimeError('方向导数核对未通过, 不得用于对照运行')


def run_case(out, method, group):
    pipe = build(method, group in ('C','D'), 20 if group in ('B','D') else 5)
    start = perf_counter()
    rho, history = pipe.optimizer.optimize(pipe.design_variable, pipe.density_distribution)
    state = pipe.analyzer.solve_state(rho_val=rho)
    g = np.asarray(pipe.stress_constraint.fun(rho,state)).reshape(-1)
    r = np.asarray(pipe.stress_constraint.compute_relative_violation(rho,state)).reshape(-1)
    np.savez(out/'final-state.npz',density=np.asarray(rho),design=pipe.optimizer.last_design,
             scale=pipe.al_objective.scale,lamb=pipe.al_objective.lamb,mu=pipe.al_objective.mu)
    summary=dict(method=method,group=group,beta=1,total_budget=120,
        elapsed_seconds=perf_counter()-start,max_constraint=float(g.max()),
        max_relative_violation=float(r.max()),violating_cells=int((r>.003).sum()),
        high_density_violating=int(((np.asarray(rho)>=.9)&(r>.003)).sum()),
        low_density_violating=int(((np.asarray(rho)<.01)&(r>.003)).sum()),
        volume_fraction=float(pipe.volume_objective.fun(rho,state)))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'stationarity.json').write_text(json.dumps(pipe.optimizer.diagnostics,indent=2))
    # 保存原生历史中的完整停止指标, 不以AL目标下降判定收敛.
    (out/'history.json').write_text(json.dumps(dict(changes=history.changes,
        scalar_histories=history.scalar_histories),indent=2))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('verify','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',choices=('lfem','huzhang'))
    parser.add_argument('--group',choices=('A','B','C','D'))
    args=parser.parse_args()
    if args.action=='run' and (not args.method or not args.group):
        parser.error('run要求--method及--group')
    args.output.mkdir(parents=True,exist_ok=False)
    if args.action=='verify':verify(args.output)
    else:run_case(args.output,args.method,args.group)
