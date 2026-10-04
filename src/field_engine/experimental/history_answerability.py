"""Linear symmetric-box preflight. Float64 diagnostics, not outward certificates.

The admissible box and equality constraints are supplied by the consumer.
A tangent result never certifies a nonlinear query. Rank applies to exact
linear summaries and the declared query rows, not arbitrary encoders/bits.
"""
import time
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog
from scipy import sparse

LP_OPTIONS = {'primal_feasibility_tolerance':1e-10,
              'dual_feasibility_tolerance':1e-10,
              'ipm_optimality_tolerance':1e-12}


def rows(A, n):
    return np.asarray(A, float).reshape(-1, n)


def normalized(A):
    norm = np.max(abs(A), axis=1)
    keep = norm > 1e-15
    return A[keep] / norm[keep, None]


def independent(A):
    if not len(A): return A
    _, s, V = np.linalg.svd(A, full_matrices=False)
    return V[s > 1e-12*s[0]]


def preflight(q, S, h, R=None, check_dual=True):
    start = time.process_time()
    q = np.asarray(q, float); h = np.broadcast_to(h, q.shape).copy(); n = q.size
    if np.any(h < 0) or not np.isfinite(q).all(): raise ValueError('invalid domain/query')
    R = np.zeros((0, n)) if R is None else rows(R, n)
    S = rows(S, n)
    original_E = normalized(np.vstack([S, R]) * h)
    E = independent(original_E)
    d = q*h; scale = max(float(np.max(abs(d))), 1e-30); c = d/scale
    result = linprog(-c, A_eq=sparse.csr_matrix(E), b_eq=np.zeros(len(E)),
                     bounds=[(-1, 1)]*n, method='highs', options=LP_OPTIONS)
    if not result.success: raise RuntimeError(result.message)
    a = result.x*h
    lb = max(0., float(q@a))
    ER = independent(normalized(R*h))
    full = linprog(-c, A_eq=sparse.csr_matrix(ER) if len(ER) else None,
                   b_eq=np.zeros(len(ER)) if len(ER) else None,
                   bounds=[(-1, 1)]*n, method='highs', options=LP_OPTIONS)
    if not full.success: raise RuntimeError(full.message)
    support = max(0., float(-full.fun*scale))
    # Independent conventional L1 residual-decoder dual.
    # min sum v, -v <= c - E.T lambda <= v.
    dual = None
    if check_dual:
        ne = E.shape[0]
        C = sparse.hstack([sparse.csr_matrix(E.T), -sparse.eye(n)], format='csr')
        D = sparse.hstack([-sparse.csr_matrix(E.T), -sparse.eye(n)], format='csr')
        dr = linprog(np.r_[np.zeros(ne), np.ones(n)], A_ub=sparse.vstack([C,D]),
                     b_ub=np.r_[c,-c], bounds=[(None,None)]*ne+[(0,None)]*n,
                     method='highs-ipm', options={**LP_OPTIONS,'presolve':False})
        if not dr.success: raise RuntimeError(dr.message)
        # Recompute a valid L1 bound from the returned decoder rather than
        # trusting slack variables with solver feasibility tolerance.
        dual = float(np.sum(abs(c-E.T@dr.x[:ne]))*scale)
    residual = float(np.max(abs(original_E@result.x))) if len(original_E) else 0.
    return {'LB':lb,'half_answer_span':support,'full_answer_span':2*support,
            'LB_over_half_span':lb/support if support else None,
            'primal_scaled_residual':residual,'dual_L1_upper':dual,
            'primal_dual_gap':None if dual is None else dual-lb,
            'constraint_rows':len(original_E),'numerical_constraint_rank':len(E),
            'cpu_s':time.process_time()-start,'witness':a}


def minrank(Q, R=None):
    """Thresholded row-rank estimate, not a certificate of exact minimum rank.

    It can be LOWER than the exact minimum: on the Maxwell-ring single-cut
    family the exact all-time rank is 12 while this estimate gives 10 (sweep
    8-11). Never use it as the retained size of an exact deletion contract.
    """
    Q = np.atleast_2d(Q); n = Q.shape[1]
    T = np.eye(n) if R is None or not len(R) else null_space(rows(R,n))
    V = Q@T
    scale = max(float(np.max(abs(Q))),1e-30)
    singular = np.linalg.svd(V/scale,compute_uv=False)
    return {'rank':int(np.sum(singular>1e-9)), 'rank_kind':'FLOAT64_THRESHOLD_ESTIMATE',
            'may_underestimate_exact_rank':True,
            'rank_sweep':{str(t):int(np.sum(singular>t)) for t in [1e-7,1e-9,1e-11]},
            'singular_values_scaled':singular.tolist(), 'allowed_span_dim':T.shape[1]}


def two_point(ya, yb, eps, *, payload_equal):
    """IMPOSSIBLE only when the caller asserts both histories give one payload."""
    gap = float(np.max(abs(np.asarray(ya)-np.asarray(yb))))
    return {'separation':gap,'minimax_LB':gap/2,'epsilon':eps,
            'payload_equal':bool(payload_equal),
            'gate':'IMPOSSIBLE' if payload_equal and gap>2*eps else 'UNKNOWN'}


def diagnose(q, S, h, *, epsilon, query_contract, R=None):
    """Return scoped numerical diagnostics before retaining a linear summary.

    query_contract has no default: "linear" only for an exactly linear query,
    "tangent" for a Jacobian row, which deliberately never grants nonlinear
    answerability.
    The caller supplies a symmetric physical domain and its query pullback.
    No automatic deletion, admission, empirical calibration or outward proof.
    """
    if query_contract not in ("linear", "tangent"):
        raise ValueError("explicit linear or tangent query contract required")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    q = np.asarray(q, float)
    if q.ndim != 1 or not q.size or not np.isfinite(q).all():
        raise ValueError("finite nonempty query vector required")
    hh = np.broadcast_to(h, q.shape)
    if not np.isfinite(hh).all() or np.any(hh <= 0):
        raise ValueError("positive finite uncertainty radii required")
    if not np.isfinite(np.asarray(S)).all() or (R is not None and not np.isfinite(np.asarray(R)).all()):
        raise ValueError("finite constraint matrices required")
    report = preflight(q, S, hh, R)
    quality = report["primal_scaled_residual"] <= 1e-7 and abs(report["primal_dual_gap"]) <= 1e-7*max(1., report["LB"])
    if not quality:
        status = "UNKNOWN_NUMERICAL_QUALITY"
    elif query_contract == "tangent":
        status = "UNKNOWN_NONLINEAR_REMAINDER"
    elif report["LB"] > epsilon:
        status = "LINEAR_NUMERICAL_INSUFFICIENT"
    elif report["dual_L1_upper"] <= epsilon + 1e-12:
        status = "LINEAR_NUMERICAL_WITHIN_BUDGET"
    else:
        status = "UNKNOWN_NUMERICAL_MARGIN"
    report.update(status=status, epsilon=epsilon, query_contract=query_contract,
                  outward_certified=False, empirical_validity="UNKNOWN")
    return report
