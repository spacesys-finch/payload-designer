"""Calculate grism efficiency for VPH grism from a_in, fringe frequency,
wavelength, DCG thickness."""

# stdlib
import logging
import logging.config
from pathlib import Path

# external
import numpy as np
import pandas as pd

# project
from payload_designer.components import diffractors
from payload_designer.libs import plotlib

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
l = np.linspace(start=90, stop=1800, num=3)  # [nm]
v = np.linspace(start=900, stop=1200, num=3)  # [L/mm]
a_in = np.linspace(start=0, stop=20, num=3)
d = np.linspace(start=0.25, stop=5, num=3)

n_g = 0.1
n_3 = 1.3
n_2 = 1.52
n_1 = 1.0
a = 90
eff_mat = 0.85
m = 1
# endregion

if __name__ == "__main__":
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
    eff, df = diffractor.get_diffraction_efficiency()
    # endregion

    # region plots
    # df2=df.loc[((df['a_in [°]'] == 0.0) & (df['d [μm]'] == 2.0) & (df['l [nm]'] == 1620) & (df['v [lines/mm]'] == 1200))],
    # LOG.debug(df2.to_string())
    plotlib.line(
        df=df.loc[
            (
                (df["d [um]"] == 2.0)
                & (df["l [nm]"] == 1620.0)
                & (df["v [lines/mm]"] == 1200.000000)
                & (df["a_in [°]"] >= 0.0)
            )
        ],
        x="a_in [°]",
        y="eff",
        title="Grism Efficiency vs Incident Ray Angle with d = 2um, l = 1620nm, v = 1200lines/mm",
    )
    plotlib.line(
        df=df.loc[
            (
                (df["a_in [°]"] == 0.0)
                & (df["d [um]"] == 2.0)
                & (df["l [nm]"] == 1620.0)
                & (df["v [lines/mm]"] > 900)
            )
        ],
        x="v [lines/mm]",
        y="eff",
        title="Grism Efficiency vs Fringe Frequency with d = 2um, l = 1620nm, a_in = 0°",
    )
    plotlib.line(
        df=df.loc[
            (
                (df["a_in [°]"] == 0.0)
                & (df["d [um]"] == 2.0)
                & (df["v [lines/mm]"] == 1200)
                & (df["l [nm]"] > 1440)
            )
        ],
        x="l [nm]",
        y="eff",
        title="Grism Efficiency vs Wavelength with d = 2um, a_in = 0°, v = 1200lines/mm",
    )
    plotlib.line(
        df=df.loc[
            (
                (df["a_in [°]"] == 0.0)
                & (df["v [lines/mm]"] == 1200)
                & (df["l [nm]"] == 1620)
                & (df["d [um]"] > 0.75)
            )
        ],
        x="d [um]",
        y="eff",
        title="Grism Efficiency vs DCG Thickness with l = 1620nm, a_in = 0°, v = 1200lines/mm",
    )

    # endregion
