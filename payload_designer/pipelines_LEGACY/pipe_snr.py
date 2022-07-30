"""Compute SNR as function of wavelength."""

# stdlib
import logging
import logging.config
from pathlib import Path

# external
import numpy as np
import pandas as pd

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
lmbda = np.linspace(start=900, stop=1700, num=100)  # [nm]
f_n = 1.5
w_s = 1  # slit width [mm]
l_s = 20  # slit length [mm]
d_i = 20  # image diameter from foreoptics incident on slit [mm]

# LUTS
foreoptic_eta = utillib.LUT(Path("data/foreoptic_transmittance.csv"))
collimator_eta = utillib.LUT(Path("data/collimator_transmittance.csv"))
bandfilter_eta = utillib.LUT(Path("data/filter_transmittance.csv"))
diffractor_eta = utillib.LUT(Path("data/vphgrism_transmittance.csv"))
focuser_eta = utillib.LUT(Path("data/focuser_transmittance.csv"))
sensor_eta = utillib.LUT(Path("data/sensor_quantum_efficiency.csv"))

L_target = utillib.LUT(
    path=Path("data/atmos_radiance_max.csv"), scale=(1e9, 1e-9)
)  # (m, W/sr/m2/m) to (nm, W/sr/m2/nm)
# endregion

if __name__ == "__main__":
    # region component instantiation
    foreoptic = foreoptics.Foreoptic(eta=foreoptic_eta, d_i=d_i)
    slit = slits.Slit(w_s=w_s, l_s=l_s)
    collimator = lenses.AchromLens(eta=collimator_eta)
    bandfilter = filters.Filter(eta=bandfilter_eta)
    diffractor = diffractors.VPHGrism(eta=diffractor_eta)
    focuser = lenses.AchromLens(eta=focuser_eta)
    sensor = sensors.TauSWIR()
    # endregion

    # region pipeline
    LOG.debug(f"L_target:\n{L_target(lmbda)}")

    eta_optics = (
        foreoptic.eta(lmbda)
        * collimator.eta(lmbda)
        * bandfilter.eta(lmbda)
        * diffractor.eta(lmbda)
        * focuser.eta(lmbda)
    )
    LOG.info(f"Optical transmittance:\n{eta_optics}%")

    epsilon = slit.get_slit_area() / foreoptic.get_image_area()
    LOG.info(f"Fraction of image not blocked:\n{epsilon}")

    snr, signal, noise = sensor.get_snr(
        L_target=L_target,
        eta_optics=eta_optics,
        epsilon=epsilon,
        f_n=f_n,
        lmbda=lmbda,
    )
    LOG.info(f"SNR:\n{snr}")
    # endregion

    # region plots
    dfd = {r"$\lambda$": lmbda.flatten(), "SNR": snr.flatten()}
    df = pd.DataFrame(data=dfd)
    LOG.debug(f"\n{df}")

    plotlib.line(df=df, x=r"$\lambda$", y="SNR")
    # endregion
