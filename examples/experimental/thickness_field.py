"""Run with PYTHONPATH=src python examples/thickness_field.py; coordinates mm."""
import numpy as np
from field_engine.experimental.thickness_field import thickness_field

v = np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
f = np.array([[0, 1, 2], [0, 2, 3]])
out = thickness_field((v, f), (v + [0., 0., .2], f), points=[[0., 0., 0.]],
                      directions=[[0., 0., 1.]])
print('thickness_mm', out['t'][0], 'valid', out['valid'][0])
