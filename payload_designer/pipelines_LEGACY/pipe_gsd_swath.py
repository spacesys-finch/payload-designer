"""Calculate GSD and Swath.

All formulas are from https://www.desmos.com/calculator/vnclx0hswx (GSD flat Earth,
swath) and https://www.desmos.com/calculator/rsv0oowzsv (GSD curved Earth)

"""

# stdlib
import logging
import logging.config
from pathlib import Path

# external
import numpy as np
import pandas as pd

# project
# from payload_designer.components import sensors
from payload_designer.libs import plotlib

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
# for both GSD and swath
flight_h = np.linspace(start=160, stop=2000, num=100)  # [km]
pix_pitch = 15  # [um] replace with actual value
foc_len = np.linspace(start=1, stop=100, num=100)  # [mm]
skew_angle = -20  # [deg] skew angle from nadir; replace with actual value
wavelength = 800  # [nm] replace with actual value
aperture_d = 10  # [mm] replace with actual value

# for GSD as function of skew angle
theta = np.linspace(start=-70, stop=70, num=100)  # [deg]
R_E = 6.371 * 10 ^ 6  # [m] radius of Earth
r_orb = R_E + (flight_h * 10 ^ 3)  # [m] radius of orbit

# for swath
pix_across = 2000  # replace with actual sensor pixel count in across-track direction
pix_along = 1800  # replace with actual sensor pixel count in along-track direction
sens_hor = pix_across * pix_pitch  # [um] horizontal sensor size
sens_ver = pix_along * pix_pitch  # [um] vertical sensor size
# endregion

# region unit conversions
flight_h = flight_h * 10 ^ 3  # [m]
pix_pitch = pix_pitch * 10 ^ (-6)  # [m]
foc_len = foc_len * 10 ^ (-3)  # [m]
skew_angle = (np.pi * skew_angle) / 180  # [rad]
wavelength = wavelength * 10 ^ (-9)  # [m]
aperture_d = aperture_d * 10 ^ (-3)  # [m]
sens_hor_size = sens_hor * 10 ^ (-6)  # [m]
sens_ver_size = sens_ver * 10 ^ (-6)  # [m]
# endregion

if __name__ == "__main__":
    # region component instantiation
    # sensor = sensors.Sensor(px_x = px_x)
    # endregion

    # region pipeline
    gsd_sen_lim = (flight_h * pix_pitch) / (
        foc_len * np.cos(skew_angle)
    )  # sensor-limited GSD, flat Earth
    gsd_diffrac_lim = (1.22 * wavelength * flight_h) / (
        aperture_d * np.cos(skew_angle)
    )  # diffraction-limited GSD, flat Earth
    gsd_abs = max(gsd_sen_lim, gsd_diffrac_lim)  # actual system GSD, flat Earth

    swath_wid = ((flight_h * sens_hor_size) / (foc_len * np.cos(skew_angle))) * 10 ^ (
        -3
    )  # swath width
    swath_len = ((flight_h * sens_ver_size) / (foc_len * np.cos(skew_angle))) * 10 ^ (
        -3
    )  # swath length

    term_1a = 2 * r_orb * np.cot(theta)
    term_1b = -2 * r_orb * (np.cot(theta)) ^ 2
    term_2a = (
        (4 * R_E ^ 2 * (np.cot(theta)) ^ 2) - (4 * r_orb ^ 2) + (4 * R_E ^ 2)
    ) ^ (0.5)
    term_2b = np.cot(theta) * (
        (4 * R_E ^ 2 * (np.cot(theta)) ^ 2) - (4 * r_orb ^ 2) + (4 * R_E ^ 2)
    ) ^ (0.5)
    term_3 = 2 * ((np.cot(theta)) ^ 2 + 1)

    curv_1 = (
        ((term_1a + term_2a) / term_3) ^ 2 + ((term_1b - term_2b) / term_3) ^ 2
    ) ^ (
        0.5
    )  # curvature
    curv_2 = (
        ((term_1a - term_2a) / term_3) ^ 2 + ((term_1b + term_2b) / term_3) ^ 2
    ) ^ (
        0.5
    )  # curvature
    gsd_skew = (
        min(curv_1, curv_2) * pix_pitch / foc_len
    )  # GSD as function of skew angle
    # endregion

    # region plots
    dfd1 = {
        "focal length": foc_len.flatten(),
        "sensor-limited gsd": gsd_sen_lim.flatten(),
    }
    df1 = pd.DataFram(data=dfd1)
    LOG.debug(df1)

    dfd2 = {
        "focal length": foc_len.flatten(),
        "diffraction-limited gsd": gsd_diffrac_lim.flatten(),
    }
    df2 = pd.DataFram(data=dfd2)
    LOG.debug(df2)

    dfd3 = {"focal length": foc_len.flatten(), "absolute gsd": gsd_abs.flatten()}
    df3 = pd.DataFram(data=dfd3)
    LOG.debug(df3)

    dfd4 = {"skew angle": theta.flatten(), "gsd skew": gsd_skew.flatten()}
    df4 = pd.DataFram(data=dfd4)
    LOG.debug(df4)

    dfd5 = {"focal length": foc_len.flatten(), "swath width": swath_wid.flatten()}
    df5 = pd.DataFram(data=dfd5)
    LOG.debug(df5)

    dfd6 = {"focal length": foc_len.flatten(), "swath length": swath_len.flatten()}
    df6 = pd.DataFram(data=dfd6)
    LOG.debug(df6)

    # GSD - flat earth
    plotlib.line(
        df=df1, x=foc_len, y=gsd_sen_lim, title="Sensor-limited GSD vs Focal Length"
    )
    plotlib.line(
        df=df2,
        x=foc_len,
        y=gsd_diffrac_lim,
        title="Diffraction-limited GSD vs Focal Length",
    )
    plotlib.line(df=df3, x=foc_len, y=gsd_abs, title="Absolute GSD vs Focal Length")

    # GSD - considering Earth curvature
    plotlib.line(df=df4, x=theta, y=gsd_skew, title="Absolute GSD vs Skew Angle")

    # swath
    plotlib.line(df=df5, x=foc_len, y=swath_wid, title="Swath Width vs Focal Length")
    plotlib.line(df=df6, x=foc_len, y=swath_len, title="Swath Length vs Focal Length")
    # endregion
