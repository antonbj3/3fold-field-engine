
import sys, json, math
from build123d import Cylinder

r_in = float(sys.argv[1]); h_in = float(sys.argv[2])
c = Cylinder(radius=r_in, height=h_in)
V = c.volume
SA = c.area

# recover (r,h) from V and SA alone (NOT from c.radius/c.height -- those would just echo the constructor
# args, a tautology). V = pi r^2 h ; SA = 2 pi r h + 2 pi r^2 = 2 pi r (h + r)
# => h = V/(pi r^2); substitute into SA: SA = 2 pi r (V/(pi r^2) + r) = 2V/r + 2 pi r^2
# solve f(r) = 2V/r + 2*pi*r^2 - SA = 0 via Newton's method from a coarse bracket start.
def f(r):
    return 2.0*V/r + 2.0*math.pi*r*r - SA
def fprime(r):
    return -2.0*V/(r*r) + 4.0*math.pi*r

r = r_in * 0.5  # deliberately WRONG initial guess (not the answer) to prove Newton converges to it, not echoes it
for _ in range(200):
    fr = f(r)
    dr = fr / fprime(r)
    r = r - dr
    if abs(dr) < 1e-14:
        break
h = V / (math.pi * r * r)

print(json.dumps({"r_in": r_in, "h_in": h_in, "occ_volume": V, "occ_area": SA,
                   "r_recovered": r, "h_recovered": h,
                   "r_abs_err": abs(r - r_in), "h_abs_err": abs(h - h_in)}))
