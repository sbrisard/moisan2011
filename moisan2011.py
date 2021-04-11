"""Periodic-plus-smooth decomposition of an image.

This module implements the periodic-plus-smooth decomposition of an
image introduced by L. Moisan [J. Math. Imag. Vision 39(2), 161-179,
doi:10.1007/s10851-010-0227-1]. The author's version of this paper is
available at

    https://hal.archives-ouvertes.fr/hal-00388020/

Several implementations of Algorithms 1 and 2 (see the above cited paper
for a description of these algorithms):

  - _per  -- reference implementation of Algorithm 1. Unoptimized.
  - per   -- optimized implementation of Algorithm 1.
  - per2  -- implementation of Algorithm 2.
  - rper  -- optimized implementation of Algorithm 1. For real images.
  - rper2 -- optimized implementation of Algorithm 2. For real images.

All these functions share the same interface

    per(u, inverse_dft=True)

where u is the image to be processed.

If inverse_dft is True, then the pair (p, s) is returned (p: periodic
component; s: smooth component).

If inverse_dft is False, then the pair (dft(p), dft(s)) is returned,
where dft denotes the discrete Fourier transform:

  - for _per, per, and per2: dft == numpy.fft.fft2,
  - for rper, rper2:         dft == numpy.fft.rfft2.

In general, users are advised to use the functions per and rper for
complex and real images, respectively (both seem to be more accurate and
efficient).

This module also defines a few supporting functions and objects:

  - energy     -- Moisan's energy of the periodic-plus-smooth decomposition,
  - OperatorQ1 -- Implementation of the quadratic form Q1 as a
                  scipy.sparse.linalg.LinearOperator,
  - OperatorQ  -- Implementation of the quadratic form Q as a
                  scipy.sparse.linalg.LinearOperator.

This is illustrated in the short example below:

>>> import numpy as np
>>> from moisan2011 import *
>>> np.random.seed(20180209)
>>> u = np.ascontiguousarray(np.random.rand(32, 64))
>>> p, s = rper(u, inverse_dft=True)
>>> energy(u, 0)
27.27083292498726
>>> energy(p, s)
13.50399440683221
>>> q1 = OperatorQ1(u.shape)
>>> q = OperatorQ(u.shape)
>>> m, n = u.shape
>>> u_flat = u.reshape((m*n,))
>>> s_flat = s.reshape((m*n,))
>>> u_q1_u = np.dot(q1.matvec(u_flat), u_flat)
>>> s_q1_u = np.dot(q1.matvec(u_flat), s_flat)
>>> s_q_s = np.dot(q.matvec(s_flat), s_flat)
>>> s_q_s-2*s_q1_u+u_q1_u # should be equal to energy(p, s)
13.503994406832206
"""
import numpy as np
import scipy.ndimage as ndi
import scipy.sparse.linalg

__author__ = 'Sebastien Brisard'
__version__ = '1.0'
__release__ = __version__


def _per(u, inverse_dft=True):
    """Compute the periodic component of the 2D image u.

    This function returns the periodic-plus-smooth decomposition of
    the 2D array-like u.

    If inverse_dft is True, then the pair (p, s) is returned
    (p: periodic component; s: smooth component).

    If inverse_dft is False, then the pair

        (numpy.fft.fft2(p), numpy.fft.fft2(s))

    is returned.

    This is a reference (unoptimized) implementation of Algorithm 1.
    """
    u = np.asarray(u, dtype=np.float64)

    v = np.zeros_like(u)
    du = u[-1, :]-u[0, :]
    v[0, :] = du
    v[-1, :] = -du

    du = u[:, -1]-u[:, 0]
    v[:, 0] += du
    v[:, -1] -= du

    v_dft = np.fft.fft2(v)

    m, n = u.shape
    cos_m = np.cos(2.*np.pi*np.fft.fftfreq(m, 1.))
    cos_n = np.cos(2.*np.pi*np.fft.fftfreq(n, 1.))

    k_dft = 2.0*(cos_m[:, None]+cos_n[None, :]-2.0)
    k_dft[0, 0] = 1.0
    s_dft = v_dft/k_dft
    s_dft[0, 0] = 0.0

    if inverse_dft:
        s = np.fft.ifft2(s_dft)
        return u-s, s
    else:
        u_dft = np.fft.fft2(u)
        return u_dft-s_dft, s_dft


def per(u, inverse_dft=True, axes=(-2, -1)):
    """Compute the periodic component of the images u.

    Parameters
    ----------
    u : array_like
        Input image.
    inverse_dft : boolean, optional
        Whether to calculate the final inverse DFT or to forego
        calcuation and to return the Fourier space representation
        of the decomposition
    axes : length-2 sequence of ints, optional
        The axes along which to perform the tranformation.

    Returns
    -------
    p : ndarray
        Periodic component.
    s : ndarray
        Smooth component.

    Notes
    -----

    This function returns the periodic-plus-smooth decomposition of
    the array-like u along axes.

    If inverse_dft is True, then the pair (p, s) is returned
    (p: periodic component; s: smooth component).

    If inverse_dft is False, then the pair

        (numpy.fft.fft2(p), numpy.fft.fft2(s))

    is returned.

    This function implements Algorithm 1.

    References
    ----------
    .. [1] Moisan, Lionel. "Periodic plus smooth image
           decomposition." Journal of Mathematical Imaging and
           Vision 39.2 (2011): 161-179.
           10.1007/s10851-010-0227-1. hal-00388020v2
    """
    u = np.asarray(u, dtype=np.float64)

    ut = np.moveaxis(u, axes, (-2, -1))

    m, n = ut.shape[-2:]

    arg = 2.*np.pi*np.fft.fftfreq(m, 1.)
    cos_m, sin_m = np.cos(arg), np.sin(arg)
    one_minus_exp_m = 1.0-cos_m-1j*sin_m

    arg = 2.*np.pi*np.fft.fftfreq(n, 1.)
    cos_n, sin_n = np.cos(arg), np.sin(arg)
    one_minus_exp_n = 1.0-cos_n-1j*sin_n

    w1 = ut[..., -1] - ut[..., 0]
    w1_dft = np.fft.fft(w1)
    v_dft = w1_dft[..., None] * one_minus_exp_n[..., None, :]

    w2 = ut[..., -1, :] - ut[..., 0, :]
    w2_dft = np.fft.fft(w2)
    v_dft += one_minus_exp_m[..., None]*w2_dft[..., None, :]

    denom = 2.0*(cos_m[:, None]+cos_n[None, :]-2.0)
    denom[..., 0, 0] = 1.0
    s_dft = v_dft/denom
    s_dft[..., 0, 0] = 0.0

    if inverse_dft:
        s = np.fft.ifft2(s_dft)
        s = np.moveaxis(s, (-2, -1), axes)
        return u-s, s
    else:
        u_dft = np.fft.fft2(u, axes=axes)
        s_dft = np.moveaxis(s_dft, (-2, -1), axes)
        return u_dft-s_dft, s_dft


def per2(u, inverse_dft=True, axes=(-2, -1)):
    """Compute the periodic component of the image u.

    Parameters
    ----------
    u : array_like
        Input image.
    inverse_dft : boolean, optional
        Whether to calculate the final inverse DFT or to forego
        calcuation and to return the Fourier space representation
        of the decomposition
    axes : length-2 sequence of ints, optional
        The axes along which to perform the tranformation.

    Returns
    -------
    p : ndarray
        Periodic component.
    s : ndarray
        Smooth component.

    Notes
    -----

    This function returns the periodic-plus-smooth decomposition of
    the array-like u along axes.

    If inverse_dft is True, then the pair (p, s) is returned
    (p: periodic component; s: smooth component).

    If inverse_dft is False, then the pair

        (numpy.fft.fft2(p), numpy.fft.fft2(s))

    is returned.

    This function implements Algorithm 2.

    References
    ----------
    .. [1] Moisan, Lionel. "Periodic plus smooth image
           decomposition." Journal of Mathematical Imaging and
           Vision 39.2 (2011): 161-179.
           10.1007/s10851-010-0227-1. hal-00388020v2
    """
    u = np.asarray(u, dtype=np.float64)

    ut = np.moveaxis(u, axes, (-2, -1))

    m, n = ut.shape[-2:]

    kernel = np.array([[0.0, 1.0, 0.0],
                       [1.0, -4.0, 1.0],
                       [0.0, 1.0, 0.0]],
                      dtype=np.float64, ndmin=ut.ndim)
    kp = ndi.convolve(ut, kernel, mode='wrap')

    du = ut[..., -1, :] - ut[..., 0, :]
    kp[..., 0, :] -= du
    kp[..., -1, :] += du

    du = ut[..., -1] - ut[..., 0]
    kp[..., 0] -= du
    kp[..., -1] += du

    kp_dft = np.fft.fft2(kp)

    cos_m = np.cos(2. * np.pi * np.fft.fftfreq(m, 1.))
    cos_n = np.cos(2. * np.pi * np.fft.fftfreq(n, 1.))
    k_dft = 2.0 * (cos_m[:, None]+cos_n[None, :]-2.0)
    k_dft[..., 0, 0] = 1.0
    p_dft = kp_dft / k_dft
    p_dft[..., 0, 0] = ut.sum(axis=(-1, -2))

    if inverse_dft:
        p = np.fft.ifft2(p_dft)
        p = np.moveaxis(p, (-2, -1), axes)
        return p, u - p
    else:
        p_dft = np.moveaxis(p_dft, (-2, -1), axes)
        return p_dft, np.fft.fft2(u, axes=axes) - p_dft


def rper(u, inverse_dft=True, axes=(-2, -1)):
    """Compute the periodic component of the real image u.

    Parameters
    ----------
    u : array_like
        Input image, assumed to be real-valued
    inverse_dft : boolean, optional
        Whether to calculate the final inverse DFT or to forego
        calcuation and to return the Fourier space representation
        of the decomposition
    axes : length-2 sequence of ints, optional
        The axes along which to perform the tranformation.

    Returns
    -------
    p : ndarray
        Periodic component.
    s : ndarray
        Smooth component.

    Notes
    -----

    This function returns the periodic-plus-smooth decomposition of
    the 2D array-like u. The image must be real.

    If inverse_dft is True, then the pair (p, s) is returned
    (p: periodic component; s: smooth component).

    If inverse_dft is False, then the pair

        (numpy.fft.rfft2(p, axes=axes), numpy.fft.rfft2(s, axes=axes))

    is returned.

    This function implements Algorithm 1.

    References
    ----------
    .. [1] Moisan, Lionel. "Periodic plus smooth image
           decomposition." Journal of Mathematical Imaging and
           Vision 39.2 (2011): 161-179.
           10.1007/s10851-010-0227-1. hal-00388020v2
    """
    u = np.asarray(u, dtype=np.float64)
    ut = np.moveaxis(u, axes, (-2, -1))

    m, n = ut.shape[-2:]

    arg = 2.*np.pi * np.fft.fftfreq(m, 1.)
    cos_m, sin_m = np.cos(arg), np.sin(arg)
    one_minus_exp_m = 1.0 - cos_m - 1j*sin_m

    arg = 2.*np.pi * np.fft.rfftfreq(n, 1.)
    cos_n, sin_n = np.cos(arg), np.sin(arg)
    one_minus_exp_n = 1.0 - cos_n - 1j*sin_n

    w1 = ut[..., -1] - ut[..., 0]
    # Use complex fft because irfft2 needs all modes in the first direction
    w1_dft = np.fft.fft(w1)
    v1_dft = w1_dft[..., None] * one_minus_exp_n[..., None, :]

    w2 = ut[..., -1, :] - ut[..., 0, :]
    w2_dft = np.fft.rfft(w2)
    v2_dft = one_minus_exp_m[..., None]*w2_dft[..., None, :]

    k_dft = 2.0 * (cos_m[:, None]+cos_n[None, :]-2.0)
    k_dft[..., 0, 0] = 1.0
    s_dft = (v1_dft+v2_dft) / k_dft
    s_dft[..., 0, 0] = 0.0

    if inverse_dft:
        s = np.fft.irfft2(s_dft, (m, n))
        s = np.moveaxis(s, (-2, -1), axes)
        return u - s, s
    else:
        u_dft = np.fft.rfft2(u, axes=axes)
        s_dft = np.moveaxis(s_dft, (-2, -1), axes)
        return u_dft-s_dft, s_dft


def rper2(u, inverse_dft=True, axes=(-2, -1)):
    """Compute the periodic component of the real images u.

    Parameters
    ----------
    u : array_like
        Input image, assumed to be real-valued
    inverse_dft : boolean, optional
        Whether to calculate the final inverse DFT or to forego
        calcuation and to return the Fourier space representation
        of the decomposition
    axes : length-2 sequence of ints, optional
        The axes along which to perform the tranformation.

    Returns
    -------
    p : ndarray
        Periodic component.
    s : ndarray
        Smooth component.

    Notes
    -----

    This function returns the periodic-plus-smooth decomposition of
    the array-like u along axes. The images must be real.

    If inverse_dft is True, then the pair (p, s) is returned
    (p: periodic component; s: smooth component).

    If inverse_dft is False, then the pair

        (numpy.fft.rfft2(p, axes=axes), numpy.fft.rfft2(s, axes=axes))

    is returned.

    This function implements Algorithm 2.

    References
    ----------
    .. [1] Moisan, Lionel. "Periodic plus smooth image
           decomposition." Journal of Mathematical Imaging and
           Vision 39.2 (2011): 161-179.
           10.1007/s10851-010-0227-1. hal-00388020v2
    """
    u = np.asarray(u, dtype=np.float64)
    ut = np.moveaxis(u, axes, (-2, -1))

    m, n = ut.shape[-2:]

    kernel = np.array([[0.0, 1.0, 0.0],
                       [1.0, -4.0, 1.0],
                       [0.0, 1.0, 0.0]],
                      dtype=np.float64, ndmin=ut.ndim)
    kp = ndi.convolve(ut, kernel, mode='wrap')

    du = ut[..., -1, :] - ut[..., 0, :]
    kp[..., 0, :] -= du
    kp[..., -1, :] += du

    du = ut[..., -1] - ut[..., 0]
    kp[..., 0] -= du
    kp[..., -1] += du

    kp_dft = np.fft.rfft2(kp)

    cos_m = np.cos(2. * np.pi * np.fft.fftfreq(m, 1.))
    cos_n = np.cos(2. * np.pi * np.fft.rfftfreq(n, 1.))
    k_dft = 2.0 * (cos_m[:, None]+cos_n[None, :]-2.0)
    k_dft[..., 0, 0] = 1.0
    p_dft = kp_dft / k_dft
    p_dft[..., 0, 0] = ut.sum(axis=(-2, -1))

    if inverse_dft:
        p = np.fft.irfft2(p_dft, (m, n))
        p = np.moveaxis(p, (-2, -1), axes)
        return p, u - p
    else:
        p_dft = np.moveaxis(p_dft, (-2, -1), axes)
        return p_dft, np.fft.rfft2(u, axes=axes) - p_dft


def ssd(a, b):
    """Sum of squared differences."""
    delta2 = b-a
    delta2 *= delta2
    return np.sum(delta2)


def energy(p, s):
    """Return the total energy of the periodic-plus-smooth decomposition.

    The periodic and smooth components p and s are 2D arrays of
    float64. They should have the same shape, although this is not
    required by this function.  2D arrays.

    The energy is defined in Moisan (2011), Theorem 1. The
    contribution of the periodic component is

        E_p(p) = sum sum [p(x)-p(y)]**2,
                  x   y

    where the first sum is carried over all boundary pixels x, and the
    second sum is carried over the indirect neighbors y of x. The
    contribution of the smooth component is

        E_s(s) = sum sum [s(x)-s(y)]**2,
                  x   y

    where the first sum is carried over all pixels x, and the second
    sum is carried over the direct neighbors y of x. The total energy
    is then defined as

        E(p, s) = E_p(p) + E_s(s).
    """
    p, s = np.broadcast_arrays(p, s)
    return 2*(ssd(p[:, 0], p[:, -1]) +
              ssd(p[0, :], p[-1, :]) +
              ssd(s[:-1, :], s[1:, :]) +
              ssd(s[:, :-1], s[:, 1:]))


class ImageLinearOperator(scipy.sparse.linalg.LinearOperator):
    """Linear operator that operate on images.

    This class defines a linear operator (in the SciPy sense) that
    operates on n-dimensional images, the shape of which is passed to
    the initializer

    >>> a = ImageLinearOperator((10, 5))
    >>> a.img_shape
    (10, 5)
    >>> a.shape
    (50, 50)

    SciPy linear operators operate on one-dimensional vectors: the
    methods _matvec and _adjoint implemented by each subclass must
    therefore first reshape the input array to a n-dimensional
    image. By convention, C-ordering will always be assumed.

        y = numpy.zeros_like(x)
        x2 = x.reshape(self.img_shape)
        y2 = y.reshape(self.img_shape)
        ......................
        # Operate on x2 and y2
        ......................
        return y

    Alternatively, developers may implement the method _apply that
    operates on n-dimensional images: the default implementation of
    _matvec calls this method on the input vector, suitably reshaped to
    a n-dimensional image.
    """
    def __init__(self, img_shape, dtype=np.float64):
        self.img_shape = img_shape
        n = np.product(self.img_shape)
        shape = (n, n)
        super(ImageLinearOperator, self).__init__(dtype, shape)

    def _matvec(self, x):
        y = np.zeros_like(x)
        x2 = x.reshape(self.img_shape)
        y2 = y.reshape(self.img_shape)
        self._apply(x2, y2)
        return y

    def _apply(x, y=None):
        """Apply this operator on the input image x.

        The shape of x must be self.img_shape. The returned array has
        same shape as x.

        If specified, the optional argument y must be an array of same
        shape as x. It is modified in-place, and a reference to y is
        returned.
        """
        raise NotImplementedError()


class OperatorQ1(ImageLinearOperator):
    """Implementation of Q1 as a ImageLinearOperator.

    Q1 is defined by Eq. (9) of Moisan (2011)

        F(s) = <u-s, Q1.(u-s)> + <s, Q2.s>,

    where F(s) is the function to be minimized with respect to the
    smooth component s. F is defined by Eq. (8)

        F(s) = E(u-s, s) + mean(s)**2,

    so that

        <v, Q1.v> = E(v, 0) and <v, Q2.v> = E(0, v) + mean(v)**2.

    Image p = u-s must be passed as a 1-dimensional vector. Internally,
    it is reshaped to a two-dimensional image (the shape of which is
    passed to the initializer), assuming C-ordering.
    """
    def __init__(self, img_shape, dtype=np.float64):
        super(OperatorQ1, self).__init__(img_shape, dtype)

    def _apply(self, x, y=None):
        if y is None:
            y = np.zeros_like(x)

        dx = 2*(x[:, 0]-x[:, -1])
        y[:, 0] = dx
        y[:, -1] = -dx

        dx = 2*(x[0, :]-x[-1, :])
        y[0, :] += dx
        y[-1, :] -= dx

        return y

    def _adjoint(self):
        return self


class OperatorQ(ImageLinearOperator):
    """Implementation of Q = Q1+Q2 as a ImageLinearOperator.

    Q1 and Q2 are defined by Eq. (9) of Moisan (2011)

        F(s) = <u-s, Q1.(u-s)> + <s, Q2.s>,

    where F(s) is the function to be minimized with respect to the
    smooth component s. F is defined by Eq. (8)

        F(s) = E(u-s, s) + mean(s)**2,

    so that

        <v, Q1.v> = E(v, 0) and <v, Q2.v> = E(0, v) + mean(v)**2.

    Image s must be passed as a 1-dimensional vector. Internally, it is
    reshaped to a two-dimensional image (the shape of which is passed
    to the initializer), assuming C-ordering.
    """
    def __init__(self, img_shape, dtype=np.float64):
        super(OperatorQ, self).__init__(img_shape, dtype)
        self.kernel = np.array([[0, -2, 0],
                                [-2, 8, -2],
                                [0, -2, 0]], dtype=dtype)

    def _apply(self, x, y=None):
        if y is None:
            y = np.zeros_like(x)

        ndi.convolve(x, self.kernel, output=y, mode='wrap')
        return y+x.mean()/self.shape[0]

    def _adjoint(self):
        return self
