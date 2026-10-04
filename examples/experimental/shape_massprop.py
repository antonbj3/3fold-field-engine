"""Run with PYTHONPATH=src python examples/shape_massprop.py; mm and radians."""
import numpy as np
from field_engine.experimental.shape_massprop import shape_massprop

v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
frame = dict(c=np.zeros(3), shaft=np.array([0., 0., 1.]),
             a_v=np.array([0., 1., 0.]), w=np.array([0., 0., 0., 1.]))
out = shape_massprop(dict(kind='mesh', V=v, F=f, frame=frame), [0., 0., .1])
print('V_mm3', out['V'], 'dV_dlength_mm2', out['dV'][2], out['error_bound']['type'])
