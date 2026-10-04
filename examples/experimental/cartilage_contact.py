"""Run with PYTHONPATH=src python examples/cartilage_contact.py; SI units."""
from field_engine.experimental.cartilage_contact import cartilage_contact

force, radius, peak, pressure = cartilage_contact(1., .05, 1., .3, .02)
print('force_N', force, 'radius_m', radius, 'peak_Pa', peak,
      'center_Pa', pressure(0.))
