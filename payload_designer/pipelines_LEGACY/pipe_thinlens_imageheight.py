"""Calculate image height for thin lens from focal length and incident angle."""

# stdlib
import logging
import logging.config
from pathlib import Path

# external
import numpy as np

# project
from payload_designer.components import lenses
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
a_in = np.linspace(start=0, stop=10, num=100)  # [°]
f = np.linspace(start=1, stop=30, num=100)  # [mm]

# endregion

if __name__ == "__main__":
    # region component instantiation
    lens = lenses.ThinLens(f=f, a_in=a_in)
    # endregion

    # region pipeline
    h = lens.get_image_height()
    # endregion

    # region plots
    plotlib.surface(x=a_in, y=f, z=h)
    # endregion
