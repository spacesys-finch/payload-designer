"""Component classes."""

# stdlib
import logging

# external
import numpy as np

# project
from payload_designer.libs import physlib

LOG = logging.getLogger(__name__)


class VPHGrism:
    """Volume-Phase Holographic grating grism component.

    Args:
        d (float, optional): DCG thickness. Defaults to None.
        t (float, optional): transmision ratio. Defaults to None.
        a_in (float, optional): incident ray angle in degrees. Defaults to None.
        a_out (float, optional): outgoing ray angle in degrees. Defaults to None.
        R (float, optional): resolvance. Defaults to None.
        l (array_like[float], optional): wavelength in nm. Defaults to None.
        l_g (float, optional): undeviated wavelength in nm. Defaults to None.
        a (float, optional): apex angle. Defaults to None.
        m (int, optional): diffraction order. Defaults to None.
        n_1 (float, optional): external index of refraction. Defaults to None.
        n_2 (float, optional): prism index of refraction. Defaults to None.
        n_3 (float, optional): grating substrate index of refraction.
            Defaults to None.
        v (float, optional): fringe frequency. Defaults to None.
        dl (float, optional): spectral resolution. Defaults to None.
        N (float, optional): Number of illumated fringes. Defaults to None.
    """

    def __init__(
        self,
        d=None,
        t=None,
        a_in=None,
        a_out=None,
        R=None,
        l=None,
        l_g=None,
        a=None,
        m=None,
        n_1=None,
        n_2=None,
        n_3=None,
        v=None,
        dl=None,
        N=None,
    ):
        self.d = d
        self.t = t
        self.a_in = a_in
        self.a_out = a_out
        self.R = R
        self.l = l
        self.l_g = l_g
        self.a = a
        self.m = m
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        self.v = v
        self.dl = dl
        self.N = N

    def get_angle_out(self):
        """Calculates the outgoing angle from the grism.

        Returns:
            float: outgoing angle in degrees.
        """
        assert self.a_in is not None, "a_in is not set."
        assert self.n_1 is not None, "n_1 is not set."
        assert self.n_2 is not None, "n_2 is not set."
        assert self.n_3 is not None, "n_3 is not set."
        assert self.m is not None, "m is not set."
        assert self.a is not None, "a is not set."
        assert self.v is not None, "v is not set."
        assert self.l is not None, "l is not set."

        # region vectorization
        a_in = np.array(self.a_in).reshape(-1, 1)
        n_1 = np.array(self.n_1).reshape(-1, 1)
        n_2 = np.array(self.n_2).reshape(-1, 1)
        n_3 = np.array(self.n_3).reshape(-1, 1)
        m = np.array(self.m).reshape(-1, 1)
        a = np.array(self.a).reshape(-1, 1)
        v = np.array(self.v).reshape(-1, 1)
        l = np.array(self.l).reshape(-1, 1)
        # endregion

        # region unit conversions
        l = l * 10 ** -9  # m to nm
        a_in = np.radians(a_in)  # deg to rad
        a = np.radians(self.a)  # deg to rad
        # endregion

        angle_1 = a_in + a
        angle_2 = physlib.snell_angle_2(angle_1=angle_1, n_1=n_1, n_2=n_2)
        angle_3 = a - angle_2
        angle_4 = physlib.snell_angle_2(angle_1=angle_3, n_1=n_2, n_2=n_3)
        angle_5 = angle_4
        angle_6 = np.arcsin(np.sin(angle_5) - m * np.matmul(v, np.transpose(l)))
        angle_7 = angle_6
        angle_8 = physlib.snell_angle_2(angle_1=angle_7, n_1=n_3, n_2=n_2)
        angle_9 = angle_8 - a
        angle_10 = physlib.snell_angle_2(angle_1=angle_9, n_1=n_2, n_2=n_1)
        angle_out = angle_10 + a

        angle_out = np.degrees(angle_out)

        return angle_out

    def get_resolvance(self):
        """Caculates the grism resolvance.

        Raises:
            ValueError: if required parameters are not set.

        Returns:
            float: resolvance.
        """
        if self.l is not None and self.dl is not None:
            R = self.l / self.dl
        elif self.m is not None and self.N is not None:
            R = self.m * self.N
        else:
            raise ValueError("l and dl or m and N must be set.")

        return R

    def get_resolution(self):
        """Caclulates the grism optically-limited spectral resolution.

        Raises:
            ValueError: if required parameters are not set.

        Returns:
            float: resolution in nm.
        """
        if self.l is not None and self.R is not None:
            dl = self.l / self.R
        elif self.l is not None and self.m is not None and self.N is not None:
            dl = self.l / (self.m * self.N)
        else:
            raise ValueError("l and R or l and m and N must be set.")

        return dl


class ThinFocuser:
    """Thin lens focuser component.

    Args:
        f (float, optional): focal length. Defaults to None.
        h (float, optional): image height above optical axis. Defaults to None.
        a_in (float, optional): incoming ray angle relative to optical axis.
            Defaults to None.
    """

    def __init__(self, f=None, h=None, a_in=None):

        self.f = f
        self.h = h
        self.a_in = a_in

    def get_image_height(self):
        """Calculate the image height along the focal plane.

        Returns:
            float: image height in m.
        """
        assert self.f is not None, "f is not set."
        assert self.a_in is not None, "a_in is not set."

        # region vectorization
        f = np.array(self.f).reshape(-1, 1)
        a_in = np.array(self.a_in).reshape(-1, 1)
        # endregion

        # region unit conversions
        a_in = np.radians(a_in)  # deg to rad
        # endregion

        h = np.matmul(f, np.transpose(np.tan(a_in)))

        return h

    
class AchromDoublet:
    """Achromatic doublet component.

    Args:
        f_1 (float, optional): Focal length of first element (mm). Defaults to None.
        f_2 (float, optional): Focal length of second element (mm). Defaults to None.
        f_eq (float, optional): Effective focal length of system (mm). Defaults to None.
        V_1 (float, optional): Abbe number for first element. Defaults to None.
        V_2 (float, optional): Abbe number for second element. Defaults to None.
    """

    def __init__(
        self,
        f_1=None,
        f_2=None,
        f_eq=None,
        V_1=None,
        V_2=None,
    ):
        self.f_1 = f_1
        self.f_2 = f_2
        self.f_eq = f_eq
        self.V_1 = V_1
        self.V_2 = V_2

    def focal_length_1(self):
        assert self.V_1 is not None, "V_1 is not set."
        assert self.V_2 is not None, "V_2 is not set."
        assert self.f_eq is not None, "f_eq is not set."

        # region unit conversions
        f_eq = self.f_eq * 10 ** -3  # mm to m
        # endregion

        f_1 = f_eq * (self.V_1 - self.V_2) / self.V_1
        return f_1

    def focal_length_2(self):
        assert self.V_1 is not None, "V_1 is not set."
        assert self.V_2 is not None, "V_2 is not set."
        assert self.f_eq is not None, "f_eq is not set."

        # region unit conversions
        f_eq = self.f_eq * 10 ** -3  # mm to m
        # endregion

        f_2 = -f_eq * (self.V_1 - self.V_2) / self.V_2
        return f_2

    def effective_focal_length(self):

        # region unit conversions
        f_1 = self.f_1 * 10 ** -3  # mm to m
        f_2 = self.f_2 * 10 ** -3  # mm to m
        # endregion

        if f_1 is not None:
            f_eq = f_1 * self.V_1 / (self.V_1 - self.V_2)
        elif f_2 is not None:
            f_eq = -f_2 * self.V_2 / (self.V_1 - self.V_2)
        else:
            raise ValueError("f_1 or f_2 must be set.")

        return f_eq
