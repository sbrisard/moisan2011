import itertools
import os

import numpy as np
import pytest

import moisan2011

from numpy.testing import assert_allclose, assert_array_equal
from PIL import Image


def img_shapes():
    sizes = (16, 17, 32, 33)
    return itertools.product(sizes, sizes)


@pytest.mark.parametrize('shape', list(img_shapes()))
def test_OperatorQ1(shape):
    np.random.seed(20180125)
    q = moisan2011.OperatorQ1(shape, dtype=np.int64)
    p1 = np.random.randint(-128, 128, size=(q.shape[0],), dtype=np.int64)
    p2 = p1.reshape(shape)
    expected = moisan2011.energy(p2, 0)
    actual = np.dot(q.matvec(p1), p1)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize('shape', list(img_shapes()))
def test_OperatorQ(shape):
    np.random.seed(20180125)
    q = moisan2011.OperatorQ(shape, dtype=np.int64)
    # Force non-zero mean
    u1 = (np.random.randint(-128, 128)
          + np.random.randint(-128, 128, size=q.shape[0:1], dtype=np.int64))
    u1[0] -= np.sum(u1) % u1.shape[0]
    u2 = u1.reshape(shape)
    expected = moisan2011.energy(u2, 0)+moisan2011.energy(0, u2)
    actual = np.dot(q.matvec(u1), u1)
    assert_array_equal(actual, expected)


def complex_operators():
    return [moisan2011._per, moisan2011.per, moisan2011.per2]


def real_operators():
    return [moisan2011.rper, moisan2011.rper2]


def operators():
    return itertools.chain(complex_operators(), real_operators())


def real_images():
    path = os.path.join(os.path.dirname(__file__), 'images', 'hut-648x364.png')
    u = np.asarray(Image.open(path), dtype=np.float64)
    m0, n0 = u.shape
    return [np.ascontiguousarray(u[:m, :n])
            for m, n in itertools.product([m0-1, m0], [n0-1, n0])]


def real_image_stacks():
    im_list = real_images()
    return [np.array([im]*3) for im in im_list]


def operators_and_real_images():
    return itertools.product(complex_operators(), real_images())


@pytest.mark.parametrize('per, u', operators_and_real_images())
def test_p_and_s_are_real(per, u):
    p, s = per(u, inverse_dft=True)
    assert np.max(np.abs(p.imag)) <= 1E-11
    assert np.max(np.abs(s.imag)) <= 1E-11


@pytest.mark.parametrize('per, u', operators_and_real_images())
def test_p_plus_s_is_u(per, u):
    eps = 1E-15
    p, s = per(u, inverse_dft=True)
    assert_allclose(p+s, u, rtol=eps, atol=eps)


@pytest.mark.parametrize('per, u', operators_and_real_images())
def test_dft_p_plus_dft_s_is_dft_u(per, u):
    eps = 1E-14
    if per in list(complex_operators()):
        dft_u = np.fft.fft2(u)
    else:
        dft_u = np.fft.rfft2(u)
    dft_p, dft_s = per(u, inverse_dft=False)
    assert_allclose(dft_p+dft_s, dft_u, rtol=eps, atol=eps)


@pytest.mark.parametrize('per, u, inverse_dft',
                         itertools.product(operators(), real_images(),
                                           [True, False]))
def test_mean_s_is_zero(per, u, inverse_dft):
    _, s = per(u, inverse_dft=inverse_dft)
    if inverse_dft:
        assert np.abs(np.mean(s)) <= 1E-13
    else:
        assert np.abs(s[0, 0]) <= 1E-15


@pytest.mark.parametrize('per, u', operators_and_real_images())
def test_residual(per, u):
    eps_rel = {moisan2011._per: 1E-14,
               moisan2011.per: 1E-14,
               moisan2011.per2: 1E-13,
               moisan2011.rper: 1E-11,
               moisan2011.rper2: 1E-10}[per]
    m, n = u.shape
    a = moisan2011.OperatorQ(u.shape, dtype=u.dtype)
    b = moisan2011.OperatorQ1(u.shape, dtype=u.dtype).matvec(u.reshape((m*n),))
    _, s = per(u, inverse_dft=True)
    x = s.real.reshape((m*n,))
    assert np.linalg.norm(b-a.matvec(x)) <= eps_rel*np.linalg.norm(b)


def consistency_params():
    op1 = (op for op in complex_operators() if op != moisan2011._per)
    it1 = itertools.product(op1, real_images(), [True, False])
    it2 = itertools.product(real_operators(), real_images(), [True])
    return itertools.chain(it1, it2)


@pytest.mark.parametrize('per, u, inverse_dft', consistency_params())
def test_consistency(per, u, inverse_dft):
    eps = {moisan2011.per: 1E-11,
           moisan2011.per2: 1E-9,
           moisan2011.rper: 1E-10,
           moisan2011.rper2: 1E-9}[per]
    p_exp, s_exp = moisan2011._per(u, inverse_dft=inverse_dft)
    p_act, s_act = per(u, inverse_dft=inverse_dft)
    assert_allclose(p_act, p_exp, eps, eps)
    assert_allclose(s_act, s_exp, eps, eps)


@pytest.mark.parametrize('per, u, axes, inverse_dft',
                         itertools.product((op for op in operators()
                                            if op != moisan2011._per),
                                           real_image_stacks(),
                                           [(-1, -2), (0, 1)],
                                           [True, False]))
def test_nd(per, u, axes, inverse_dft):
    eps = {moisan2011.per: 1E-11,
           moisan2011.per2: 1E-9,
           moisan2011.rper: 1E-10,
           moisan2011.rper2: 1E-9}[per]
    ut = np.moveaxis(u, (-2, -1), axes)
    ps, ss = per(ut, inverse_dft, axes)
    ps = np.moveaxis(ps, axes, (-2, -1))
    ss = np.moveaxis(ss, axes, (-2, -1))
    for p, s, u in zip(ps, ss, u):
        p_exp, s_exp = per(u, inverse_dft)
        assert_allclose(p_exp, p, eps, eps)
        assert_allclose(s_exp, s, eps, eps)
