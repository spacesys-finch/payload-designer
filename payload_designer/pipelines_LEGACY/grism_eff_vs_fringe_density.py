"""Calculate grism efficiency for VPH grism from fringe density, incidence
angle, order of diffraction, grating volume thickness, refractive index of VPH
and its semiamplitude."""

# stdlib
import logging
import logging.config
from pathlib import Path

# external
import numpy as np
import pandas as pd

# project
from payload_designer.components import diffractors
from payload_designer.libs import plotlib, utillib

# region path config
filename = Path(__file__).stem
output_path = Path(f"output/{filename}")
log_path = Path(f"logs/{filename}")
# endregion

# region logging config
log_path.mkdir(parents=True, exist_ok=True)
logging.config.fileConfig(
    fname="log.conf", defaults={"path": log_path}, disable_existing_loggers=False
)
LOG = logging.getLogger(__name__)
# endregion

# region parameter config

# parameters
fringe_density = np.linspace(start=0, stop=5, num=1000)  # [mm^-1]
n_grism = np.linspace(start=1.0, stop=1.5, num=1000)
n_grism_semiamp = np.linspace(start=0.0, stop=0.5, num=1000)
grism_thickness = np.linspace(start=0.0, stop=0.5, num=1000)  # [mm]
grism_length = np.linspace(start=0.0, stop=0.5, num=1000)  # [mm]
incident_angle = np.linspace(start=0.0, stop=np.pi / 2, num=1000)  # [rad]

# constants
a = 90
d = 2.5
eff_mat = 0.85
m = 1
n_1 = 1.0
n_2 = 1.52
n_3 = 1.3
n_g = 0.1
# endregion

if __name__ == "__main__":
    # region vectorization
    a = np.array(a)
    a_in = np.array(a_in)
    d = np.array(d)
    eff_mat = np.array(eff_mat)
    l = np.array(l)
    n_1 = np.array(n_1)
    n_2 = np.array(n_2)
    n_3 = np.array(n_3)
    n_g = np.array(n_g)
    v = np.array(v)

    shape = (
        a_in.size,
        v.size,
        l.size,
        d.size,
        a.size,
        n_g.size,
        n_1.size,
        n_2.size,
        n_3.size,
        eff_mat.size,
    )

    a_in = utillib.orient_and_broadcast(a=a_in, dim=0, shape=shape)
    v = utillib.orient_and_broadcast(a=v, dim=1, shape=shape)
    l = utillib.orient_and_broadcast(a=l, dim=2, shape=shape)
    d = utillib.orient_and_broadcast(a=d, dim=3, shape=shape)
    a = utillib.orient_and_broadcast(a=a, dim=4, shape=shape)
    n_g = utillib.orient_and_broadcast(a=n_g, dim=5, shape=shape)
    n_1 = utillib.orient_and_broadcast(a=n_1, dim=6, shape=shape)
    n_2 = utillib.orient_and_broadcast(a=n_2, dim=7, shape=shape)
    n_3 = utillib.orient_and_broadcast(a=n_3, dim=8, shape=shape)
    eff_mat = utillib.orient_and_broadcast(a=eff_mat, dim=9, shape=shape)
    # endregion

    # region component instantiation
    diffractor = diffractors.VPHGrism(
        m=m,
        l=l,
        v=v,
        a_in=a_in,
        eff_mat=eff_mat,
        a=a,
        d=d,
        n_g=n_g,
        n_3=n_3,
        n_2=n_2,
        n_1=n_1,
    )
    # endregion

    # region pipeline
    n_p = diffractor.get_diffraction_efficiency()
