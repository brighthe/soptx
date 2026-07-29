"""Run the 2D plane-strain linear-elasticity PINN baseline.

Run this file from the repository root. Use ``--checkpoint_dir <directory>``
to save ``best.pt`` and ``last.pt``.
"""

from fealpy.backend import bm
from model import LinearElasticityPINNModel


bm.set_backend('pytorch')
options = LinearElasticityPINNModel.get_options()
model = LinearElasticityPINNModel(options=options)
model.run()
model.show()
