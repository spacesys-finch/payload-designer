"""Compute mass and volume envelopes."""

# stdlib
import logging
import logging.config
from pathlib import Path

# external
import numpy as np
import pandas as pd
from black import diff

# project
from payload_designer.components import (
    diffractors,
    filters,
    foreoptics,
    lenses,
    sensors,
    slits,
)
from payload_designer.libs import plotlib, utillib

LOG = logging.getLogger(__name__)

# region path config
filename = Path(__file__).stem
output_path = Path(f"output/{filename}")
log_path = Path(f"logs/{filename}")
# endregion

# region logging config
log_path.mkdir(parents=True, exist_ok=True)
logging.config.fileConfig(fname="log.conf", defaults={"path": log_path})
LOG = logging.getLogger(__name__)
# endregion

# region parameter config

masses = np.zeros(7)
Vx = np.zeros(7)
Vy = np.zeros(7)
Vz = np.zeros(7)

# foreoptics_cmount = None
collimator_to_filter = 5.121
filter_to_diffractor = 4.205
diffractor_to_focuser = 6.331
sensor_to_frontplane = 12.55

# endregion

if __name__ == "__main__":
    # region component instantiation
    foreoptic = foreoptics.Foreoptic(
        mass=80, V=(44, 44, 54)
    )  # 54mm includes spacing between foreoptic and slit
    slit = slits.Slit(mass=10, V=(15, 15, 0.1))
    collimator = lenses.AchromLens(
        mass=180, V=(12.5, 12.5, 10)
    )  # mass estimate AC508-075-C
    filter = filters.Filter(mass=40, V=(12.5, 12.5, 1))  # mass estimate FB1590-12
    grating = diffractors.VPHGrating(mass=0.28, V=(25.4, 25.4, 6))  # not grism
    focuser = lenses.AchromLens(
        mass=180, V=(12.5, 12.5, 10)
    )  # mass estimate AC508-075-C
    sensor = sensors.Sensor(M=81, V=(38, 38, 36))

    # collimator_thick = lenses.ThickLens(d=dc, n=nc, R1=R1c, R2=R2c)
    # focuser_thick = lenses.ThickLens(d=df, n=nf, R1=R1f, R2=R2f)
    # endregion

    # region pipeline
    masses[0] = foreoptic.mass
    masses[1] = slit.mass
    masses[2] = collimator.mass
    masses[3] = filter.mass
    masses[4] = grating.mass
    masses[5] = focuser.mass
    masses[6] = sensor.M

    Vx[0], Vy[0], Vz[0] = foreoptic.V[0], foreoptic.V[1], foreoptic.V[2]
    Vx[1], Vy[1], Vz[1] = slit.V[0], slit.V[1], slit.V[2]
    Vx[2], Vy[2], Vz[2] = collimator.V[0], collimator.V[1], collimator.V[2]
    Vx[3], Vy[3], Vz[3] = filter.V[0], filter.V[1], filter.V[2]
    Vx[4], Vy[4], Vz[4] = grating.V[0], grating.V[1], grating.V[2]
    Vx[5], Vy[5], Vz[5] = focuser.V[0], focuser.V[1], focuser.V[2]
    Vx[6], Vy[6], Vz[6] = sensor.V[0], sensor.V[1], sensor.V[2]

    # efl_foreoptic = 100
    # bfl_foreoptic = efl_foreoptic - foreoptics_cmount
    # bfl_foreoptic = 9  # min bfl estimate in #payload (Maggie) - in the plots as a constraint (to see where the bfl is being constrained)

    # efl_collimator = np.linspace(start=1, stop=50, num=100)
    # h1c, h2c = collimator_thick.get_principal_planes()
    # ffl_collimator = efl_collimator - h1c
    ffl_collimator = np.linspace(start=1, stop=50, num=100)

    # efl_focuser = np.linspace(start=1, stop=50, num=100)
    # h1f, h2f = focuser_thick.get_principal_planes()
    # bfl_focuser = efl_focuser - h2f
    bfl_focuser = np.linspace(start=1, stop=50, num=100)

    spacing_tot = (
        ffl_collimator
        + collimator_to_filter
        + filter_to_diffractor
        + diffractor_to_focuser
        + (bfl_focuser - sensor_to_frontplane)
    )

    Vx_max = max(Vx)
    Vy_max = max(Vy)
    Vz_tot = sum(Vz) + spacing_tot

    tot_mass = sum(masses)
    tot_V = (Vx_max, Vy_max, Vz_tot)
    # endregion

    # region plots
    x1 = ffl_collimator
    y1 = Vz_tot
    x2 = bfl_focuser
    y2 = Vz_tot

    dfd1 = {"x1": x1.flatten(), "y1": y1.flatten}
    df1 = pd.DataFrame(data=dfd1)
    LOG.debug(df1)

    dfd2 = {"x2": x2.flatten(), "y2": y2.flatten()}
    df2 = pd.DataFrame(data=dfd2)
    LOG.debug(df2)

    # both plots are the same because ffl_collimator = bfl_focuser - probably not very useful?
    plotlib.line(df=df1, x=x1, y=y1, title="Vz vs. FFL Collimator")
    plotlib.line(df=df2, x=x2, y=y2, title="Vz vs. BFL Focuser")
    # endregion
