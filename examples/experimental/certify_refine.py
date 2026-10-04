"""Run with PYTHONPATH=src python examples/certify_refine.py; coordinates mm."""
from field_engine.experimental.certify_refine import Filter, Frame, Thickness

filt = Filter(Thickness(.2))
for gap in (.19, .2, .21):
    print(gap, filt.query((0., gap, Frame((0., 0., 0.)))))
print('refined', filt.stats.refined, 'of', filt.stats.queries)
