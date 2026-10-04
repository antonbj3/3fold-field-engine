"""Contract tests for the four staged experimental capabilities."""
import numpy as np
import pytest

from field_engine.experimental.shape_massprop import shape_massprop, osteotomy
from field_engine.experimental.thickness_field import thickness_field
from field_engine.experimental.certify_refine import Filter, Frame, Thickness, INNE, UTE
from field_engine.experimental.cartilage_contact import cartilage_contact


def test_shape_massprop_mesh_edit_chain():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.],
                         [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    frame = dict(c=np.zeros(3), shaft=np.array([0., 0., 1.]),
                 a_v=np.array([0., 1., 0.]), w=np.array([0., 0., 0., 1.]))
    theta = np.array([0., 0., .1])
    edited = osteotomy(vertices, frame, theta)
    np.testing.assert_allclose(edited[3], [0., 0., 1.1])
    result = shape_massprop(dict(kind='mesh', V=vertices, F=faces, frame=frame), theta)
    np.testing.assert_allclose(result['V'], 1.1 / 6)
    np.testing.assert_allclose(result['dV'], [0., 0., 1 / 6], atol=1e-12)
    assert result['error_bound']['type'] == 'analytic_discrete_mesh'


def test_shape_massprop_csg_kink_surface():
    result = shape_massprop('finger', [0., 0., 0.], n=8)
    np.testing.assert_allclose(result['V'], 1.8e-4, rtol=1e-12)
    assert result['error_bound']['coverage'] == 1.0
    assert result['error_bound']['type'] == 'uncalibrated_surface_discretization'


def test_thickness_parallel_planes():
    pytest.importorskip("embreex")  # thickness_field casts rays through the optional embree backend
    v = np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    points = np.array([[.2, -.3, 0.], [-.3, .2, 0.]])
    result = thickness_field((v, f), (v + [0, 0, .2], f),
                             points=points, directions=np.tile([0., 0., 1.], (2, 1)))
    assert result['valid'].all()
    np.testing.assert_allclose(result['t'], .2, atol=1e-8)


def test_running_band_thickness_threshold():
    frame = Frame((0., 0., 0.))
    filt = Filter(Thickness(.2))
    assert filt.query((0., .19, frame)) == INNE
    assert filt.query((0., .21, frame)) == UTE
    assert filt.query((0., .2, frame)) == 'GRÄNS'
    assert filt.stats.queries == 3
    assert filt.stats.refined >= 1


def test_cartilage_contact_positive_pressure():
    force, radius, peak, pressure = cartilage_contact(1., .05, 1., .3, .02)
    assert force > 0 and radius > 0 and peak > 0
    assert pressure(1.1 * radius) == 0
    assert pressure(0) >= 0
