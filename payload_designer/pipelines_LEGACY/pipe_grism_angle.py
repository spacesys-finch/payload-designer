"""Compute angular range subtended by max and min wavelengths out of grism."""

# stdlib
import logging
import logging.config
from pathlib import Path

# external
import numpy as np
import pandas as pd

# project
from payload_designer.components import lenses, sensors
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
f = np.linspace(start=0.1, stop=50, num=16)  # focal length [mm]
# endregion

# region component instantiation
sensor = sensors.TauSWIR()
focuser = lenses.ThinLens(f=f)
# endregion

if __name__ == "__main__":

    # region pipeline
    h = sensor.size[1]  # height of sensor
    f = focuser.f  # focal length of focuser
    angle = 2 * np.arctan((h / 2) / f)
    # endregion

    # region unit conversions
    angle = np.degrees(angle)  # rad to deg
    # endregion

    # region plots
    dfd = {"f [mm]": f.flatten(), "angle [°]": angle.flatten()}
    df = pd.DataFrame(data=dfd)
    LOG.debug(df)

    plotlib.line(
        df=df,
        x="f [mm]",
        y="angle [°]",
        title="Angular range requirement out of grism vs focal length of focuser",
    )
    # endregion
