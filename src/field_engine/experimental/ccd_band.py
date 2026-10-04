"""Sound three-way CCD for linear PT/EE trajectories on closed [0, 1].

Zero thickness, represented binary64 endpoints. See PROOF.md for the arithmetic
and geometric contract. UNKNOWN must restrict a line search, never accept it.
The exact reserve uses SymPy only when the float filter cannot decide.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
import numpy as np

# Same IEEE constants as field_engine.experimental.certify_refine.fpband.f32ops
U = 2.0**-24
ETA = 2.0**-150
INFL = 1.0 + 2.0**-40
U64 = 2.0**-53
TINY64 = np.nextafter(0.0, 1.0)
SAFE, COLLISION, UNKNOWN = "SÄKER", "KOLLISION", "OSÄKER"


def _up(x):
    return np.nextafter(x, np.inf)


class Band:
    """Binary32 midpoint and outward binary64 absolute error radius."""
    def __init__(self, m, r):
        self.m, self.r = np.asarray(m, np.float32), np.asarray(r, np.float64)

    @classmethod
    def from64(cls, x, input_radius=0.0):
        x = np.asarray(x, np.float64)
        m = x.astype(np.float32)
        # Round the cast error outward too; do not presume exact subtraction.
        r = _up(_up(np.abs(x - m.astype(np.float64))) + input_radius)
        return cls(m, r)

    @staticmethod
    def rounding(m):
        # 2u >= u/(1-u), expressed using the computed rather than exact result.
        return _up(_up(2*U*np.abs(m.astype(np.float64))) + ETA)

    def __add__(self, b):
        m = self.m + b.m
        r = _up(_up(self.r + b.r) + self.rounding(m))
        return Band(m, r)

    def __neg__(self):
        return Band(-self.m, self.r)

    def __sub__(self, b):
        return self + (-b)

    def __mul__(self, b):
        m = self.m * b.m
        r = _up(np.abs(self.m.astype(np.float64))*b.r)
        r = _up(r + _up(self.r*np.abs(b.m.astype(np.float64))))
        r = _up(r + _up(self.r*b.r))
        return Band(m, _up(r + self.rounding(m)))

    def item(self, i):
        return Band(self.m[..., i], self.r[..., i])

    def lower(self):
        return np.nextafter(self.m.astype(np.float64) - self.r, -np.inf)

    def upper(self):
        return _up(self.m.astype(np.float64) + self.r)


def _difference(a, b):
    d = a - b  # binary64 subtraction; enclosing exact represented difference
    r = _up(_up(2*U64*np.abs(d)) + TINY64)
    return Band.from64(d, r)


def _dot(a, b):
    c = a*b
    return (c.item(0) + c.item(1)) + c.item(2)


def float_filter(x, kind, *, projection=False):
    """Batch of (N,2,4,3) endpoints; returns a mask of proved SAFE cases.

    V1 uses coordinate corners. V2 additionally tests a fixed direction at
    both time endpoints. Directions need not approximate true normals well.
    """
    x = np.asarray(x, np.float64)
    if x.ndim != 4 or x.shape[1:] != (2, 4, 3):
        raise ValueError("expected (N,2,4,3) endpoints")
    if kind not in ("PT", "EE"):
        raise ValueError("kind must be PT or EE")
    with np.errstate(all="ignore"):
        if kind == "PT":
            rel = [_difference(x[:,:,0], x[:,:,j]) for j in (1,2,3)]
            # Fourth corner of the enclosing barycentric square (u=v=1).
            corners = rel + [rel[1] + rel[2] - rel[0]]
        else:
            rel = [_difference(x[:,:,i], x[:,:,j]) for i in (0,1) for j in (2,3)]
            corners = rel
        lows = np.stack([c.lower() for c in corners], axis=2)
        highs = np.stack([c.upper() for c in corners], axis=2)
        finite = np.all(np.isfinite(lows) & np.isfinite(highs), axis=(1,2,3))
        safe = finite & np.any((np.min(lows, axis=(1,2)) > 0) |
                              (np.max(highs, axis=(1,2)) < 0), axis=1)
        if projection:
            if kind == "PT":
                normals = [np.cross(x[:,t,2]-x[:,t,1], x[:,t,3]-x[:,t,1]) for t in (0,1)]
            else:
                normals = [np.cross(x[:,t,1]-x[:,t,0], x[:,t,3]-x[:,t,2]) for t in (0,1)]
            # Any finite rounded direction is a legitimate exact direction.
            # Normalize by max abs for conditioning; no normal-error band is
            # needed because correctness does not require the true normal.
            for n in normals:
                scale = np.max(np.abs(n), axis=1)
                n = np.where((scale > 0)[:,None], n/scale[:,None], 0)
                nm = n.astype(np.float32)
                nb = Band(nm[:,None,:], np.zeros((len(x),1,3)))
                projs = [_dot(c, nb) for c in rel]
                pl = np.stack([c.lower() for c in projs], axis=2)
                ph = np.stack([c.upper() for c in projs], axis=2)
                ok = np.all(np.isfinite(pl) & np.isfinite(ph), axis=(1,2))
                safe |= ok & ((np.min(pl,axis=(1,2)) > 0) | (np.max(ph,axis=(1,2)) < 0))
        return safe


# Exact polynomial algebra, coefficients in ascending powers of t.
def _trim(a):
    a = list(a)
    while len(a)>1 and not a[-1]: a.pop()
    return tuple(a)

def _add(a,b):
    return _trim([(a[i] if i<len(a) else 0)+(b[i] if i<len(b) else 0)
                  for i in range(max(len(a),len(b)))])

def _neg(a): return tuple(-v for v in a)
def _sub(a,b): return _add(a,_neg(b))

def _mul(a,b):
    r=[0]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b): r[i+j]+=x*y
    return _trim(r)

def _vsub(a,b): return [_sub(x,y) for x,y in zip(a,b)]
def _cross(a,b):
    return [_sub(_mul(a[1],b[2]),_mul(a[2],b[1])),
            _sub(_mul(a[2],b[0]),_mul(a[0],b[2])),
            _sub(_mul(a[0],b[1]),_mul(a[1],b[0]))]
def _pdot(a,b):
    r=(0,)
    for x,y in zip(a,b): r=_add(r,_mul(x,y))
    return r
def _eval(p,t):
    r=0
    for c in reversed(p): r=r*t+c
    return r
def _sign(x): return (x>0)-(x<0)


def _vertices(x):
    return [[(Q(float(x[0,i,k])), Q(float(x[1,i,k]))-Q(float(x[0,i,k])))
             for k in range(3)] for i in range(4)]


def _on_segment(p,a,b,sign):
    if any(sign(z) for z in _cross(_vsub(p,a),_vsub(p,b))): return False
    return sign(_pdot(_vsub(p,a),_vsub(p,b))) <= 0


def _contact(v, kind, sign):
    if kind=="PT":
        p,a,b,c=v
        A,B,C=_vsub(b,a),_vsub(c,a),_vsub(p,a)
    else:
        a,b,c,d=v
        A,B,C=_vsub(b,a),_vsub(c,d),_vsub(c,a)
    D,Nu,Nv=_cross(A,B),_cross(C,B),_cross(A,C)
    for k in range(3):
        s=sign(D[k])
        if s:
            u,vv=sign(Nu[k])*s,sign(Nv[k])*s
            if kind=="PT":
                return u>=0 and vv>=0 and sign(_sub(_sub(D[k],Nu[k]),Nv[k]))*s>=0
            return u>=0 and vv>=0 and sign(_sub(D[k],Nu[k]))*s>=0 and sign(_sub(D[k],Nv[k]))*s>=0
    if kind=="PT":
        return any(_on_segment(p,e,f,sign) for e,f in ((a,b),(b,c),(c,a)))
    return any(_on_segment(p,e,f,sign) for p,e,f in ((a,c,d),(b,c,d),(c,a,b),(d,a,b)))


@dataclass(frozen=True)
class Decision:
    status: str
    method: str
    witness: object = None


def integer_filter(x, kind):
    """Exact dyadic Bernstein exclusion and rational-time contact witnesses.

    This may return None; no negative inference is made from three samples.
    """
    x=np.asarray(x,np.float64)
    if x.shape!=(2,4,3): raise ValueError("expected (2,4,3)")
    if kind not in ("PT","EE"): raise ValueError("kind must be PT or EE")
    if not np.all(np.isfinite(x)): return Decision(UNKNOWN,"nonfinite_input")
    ratios=[float(z).as_integer_ratio() for z in x.flat]
    power=max(d.bit_length()-1 for _,d in ratios)
    values=[n << (power-(d.bit_length()-1)) for n,d in ratios]
    ends=[[[values[12*t+3*i+k] for k in range(3)] for i in range(4)] for t in range(2)]
    v=[[(ends[0][i][k],ends[1][i][k]-ends[0][i][k]) for k in range(3)] for i in range(4)]
    if kind=="PT":
        p,a,b,c=v; A,B,C=_vsub(b,a),_vsub(c,a),_vsub(p,a)
    else:
        a,b,c,d=v; A,B,C=_vsub(b,a),_vsub(c,d),_vsub(c,a)
    det=_pdot(_cross(A,B),C)
    c=list(det)+[0]*(4-len(det));d0,d1,d2,d3=c
    # Three times degree-3 Bernstein coefficients, avoiding rational division.
    bern=(3*d0,3*d0+d1,3*d0+2*d1+d2,3*(d0+d1+d2+d3))
    if all(b>0 for b in bern) or all(b<0 for b in bern):
        return Decision(SAFE,"integer_bernstein")
    for ti,plane in ((0,d0),(1,8*d0+4*d1+2*d2+d3),(2,d0+d1+d2+d3)):
        if plane: continue
        points=(ends[0] if ti==0 else ends[1] if ti==2 else
                [[ends[0][i][k]+ends[1][i][k] for k in range(3)] for i in range(4)])
        const=[[(z,) for z in p] for p in points]
        if _contact(const,kind,lambda z:_sign(z[0])):
            return Decision(COLLISION,"integer_rational_witness",("0","1/2","1")[ti])
    return None


def exact_accelerated(x, kind):
    d=integer_filter(x,kind)
    return d if d is not None else exact_ccd(x,kind)


def exact_ccd(x, kind, *, refinement_limit=96):
    """Exact cubic root isolation; incomplete coplanar cases return UNKNOWN."""
    x=np.asarray(x,np.float64)
    if kind not in ("PT","EE"): raise ValueError("kind must be PT or EE")
    if x.shape!=(2,4,3): raise ValueError("expected (2,4,3)")
    if not np.all(np.isfinite(x)): return Decision(UNKNOWN,"nonfinite_input")
    v=_vertices(x)
    if kind=="PT":
        p,a,b,c=v; A,B,C=_vsub(b,a),_vsub(c,a),_vsub(p,a)
    else:
        a,b,c,d=v; A,B,C=_vsub(b,a),_vsub(c,d),_vsub(c,a)
    determinant=_pdot(_cross(A,B),C)
    if all(c==0 for c in determinant):
        for t in (Q(0),Q(1,2),Q(1)):
            if _contact(v,kind,lambda z:_sign(_eval(z,t))):
                return Decision(COLLISION,"rational_contact",str(t))
        # Static relative geometry includes rigid common translations.
        relative=[_vsub(q,v[0]) for q in v[1:]]
        if all(all(c==0 for c in z[1:]) for q in relative for z in q):
            return Decision(SAFE,"exact_static")
        return Decision(UNKNOWN,"coplanar_motion")
    try:
        import sympy as s
    except ImportError:
        return Decision(UNKNOWN,"exact_root_backend_unavailable")
    T=s.Symbol('t')
    def poly(z): return s.Poly.from_list(list(reversed(z)),T,domain=s.QQ)
    f=poly(determinant)
    intervals=f.intervals(eps=s.Rational(1,2**32),inf=0,sup=1)
    for (aa,bb),multiplicity in intervals:
        aa,bb=Q(aa),Q(bb)
        if aa==bb:
            sign=lambda z:_sign(_eval(z,aa))
        else:
            current=[aa,bb]
            cache={}
            def sign(z):
                z=_trim(z)
                if z in cache: return cache[z]
                if all(c==0 for c in z): return 0
                g=poly(z)
                if f.gcd(g).count_roots(s.Rational(current[0]),s.Rational(current[1]))>0:
                    cache[z]=0; return 0
                for _ in range(refinement_limit):
                    lo,hi=Q(0),Q(0)
                    a0,b0=current
                    for c0 in reversed(z):
                        products=(lo*a0,lo*b0,hi*a0,hi*b0)
                        lo,hi=min(products)+c0,max(products)+c0
                    if lo>0 or hi<0:
                        cache[z]=1 if lo>0 else -1; return cache[z]
                    a1,b1=f.refine_root(s.Rational(a0),s.Rational(b0),steps=2)
                    current[:]=[Q(a1),Q(b1)]
                raise ArithmeticError("root sign refinement budget exhausted")
        try:
            if _contact(v,kind,sign):
                return Decision(COLLISION,"exact_algebraic_root",[str(aa),str(bb)])
        except ArithmeticError:
            return Decision(UNKNOWN,"refinement_budget")
    return Decision(SAFE,"exact_root_exclusion")


def certify_batch(x, kind, *, projection=False, exact=True, accelerated=False):
    x=np.asarray(x,np.float64)
    safe=float_filter(x,kind,projection=projection)
    decisions=[Decision(SAFE,"float32_projection" if projection else "float32_corners")
               if yes else Decision(UNKNOWN,"float32_unresolved") for yes in safe]
    if exact:
        for i in np.flatnonzero(~safe):
            decisions[i]=(exact_accelerated if accelerated else exact_ccd)(x[i],kind)
    return decisions


def certify(x, kind, *, projection=True, exact=True, accelerated=True):
    return certify_batch(np.asarray(x,np.float64)[None],kind,projection=projection,exact=exact,accelerated=accelerated)[0]
