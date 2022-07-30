"""Calculate resolution for VPH grism from wavelength, fringe frequency, beam
width."""

# stdlib
import logging
import logging.config
from pathlib import Path

# external
import numpy as np

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
logging.config.fileConfig(fname="log.conf", defaults={"path": log_path})
LOG = logging.getLogger(__name__)
# endregion

# region parameter config
lmbda = 1600  # [nm]
v = np.linspace(start=300, stop=1200, num=100)  # [L/mm]
w = 10  # [mm]
# endregion

if __name__ == "__main__":
    # region component instantiation
    diffractor = diffractors.VPHGrism(m=1, l=lmbda, v=v, w=w)
    # endregion

    # region pipeline
    res = diffractor.get_resolution()
    # endregion

    # region plots
    plotlib.line(x=v, y=res)
    # endregion
