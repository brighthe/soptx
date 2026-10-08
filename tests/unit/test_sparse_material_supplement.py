"""验证稀疏材料补样的镜像配对与跨批次可复现性."""
from pathlib import Path
import numpy as np
import pytest
from soptx.backend import backend_manager as bm
from soptx.fem.substructure.independent_targets import IndependentTargetProvider


def test_sparse_sampler_batch_independence(monkeypatch):
    """同一种子与指纹集合在不同批量下产生同一组训练镜像家族."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'experiments/topopt_piml_substructure'))
    from supplement_samples import SparseMaterialSampler
    from mixed_sampling import MixedMaterialSampler
    previous = bm.get_current_backend().backend_name
    try:
        bm.set_backend('numpy')
        provider = IndependentTargetProvider(cell_size=(5.,5.,5.),n_fine=(5,5,5))
        output = []
        for chunks in ((8,), (3, 3, 2)):
            rng = np.random.default_rng(2039)
            base = MixedMaterialSampler(provider.prototype,10,rng,1e-7,(1e-5,1.),3.,3.)
            sampler = SparseMaterialSampler(base,8)
            values = np.concatenate([sampler(rng,n,125) for n in chunks])
            assert sampler.cursor == 8 and sampler.pending is None
            assert len(np.unique(sampler.groups)) == 4
            np.testing.assert_array_equal(sampler.groups[::2],sampler.groups[1::2])
            for first, second in zip(values[::2],values[1::2]):
                np.testing.assert_array_equal(base._mirror_and_fingerprint(first)[0],second)
            rho=((values-1e-7)/(1-1e-7))**(1/3)
            assert np.all((rho.mean(axis=1)>=.004)&(rho.mean(axis=1)<=.04))
            assert np.all((rho.max(axis=1)>=.025)&(rho.max(axis=1)<=.15))
            assert np.all(rho.min(axis=1)<=1e-12)
            output.append(values)
        np.testing.assert_array_equal(*output)
        with pytest.raises(ValueError):
            SparseMaterialSampler(base,3)
    finally:
        bm.set_backend(previous)
