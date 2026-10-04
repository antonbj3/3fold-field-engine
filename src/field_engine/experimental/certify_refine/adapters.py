from fractions import Fraction as Q

from .certify_refine import INNE, UTE
from .higham import (
    IDENTITY_ROWS,
    enclose_max,
    exact_vertical_determinant_zero,
    f32_divide_interval,
    f32_dot,
    f32_mul,
    f32_sub,
    f32_sub_vectors,
    f64_local_f32,
    f64_sub,
    from_f64,
)


def fq(value):
    return Q.from_float(float(value))


class SDF:
    threshold, mode = 0.0, "below"

    def cheap(self, query):
        node, point, frame = query
        point_values, point_errors = f64_local_f32(point, frame)
        if node["op"] == "halfspace":
            normal = [from_f64(value) for value in node["normal"]]
            normal_values = [item[0] for item in normal]
            normal_errors = [item[1] for item in normal]
            value, error = f32_dot(
                point_values,
                point_errors,
                normal_values,
                normal_errors,
            )
            offset, offset_error = from_f64(node["offset"])
            return f32_sub(value, error, offset, offset_error)
        if node["op"] == "box":
            center = [from_f64(value) for value in node["center"]]
            half_extents = [from_f64(value) for value in node["half_extents"]]
            values = []
            errors = []
            for i in range(3):
                delta, delta_error = f32_sub(
                    point_values[i],
                    point_errors[i],
                    center[i][0],
                    center[i][1],
                )
                value, error = f32_sub(
                    delta,
                    delta_error,
                    half_extents[i][0],
                    half_extents[i][1],
                )
                values.append(value)
                errors.append(error)
            return enclose_max(values, errors)
        raise ValueError(node["op"])

    def refine(self, query):
        node, point, frame = query
        delta = [fq(point[i]) - fq(frame.origin[i]) for i in range(3)]
        point_values = [
            sum(fq(frame.rows[j][i]) * delta[i] for i in range(3))
            for j in range(3)
        ]
        if node["op"] == "halfspace":
            return (
                sum(fq(node["normal"][i]) * point_values[i] for i in range(3))
                - fq(node["offset"])
            )
        if node["op"] == "box":
            return max(
                abs(point_values[i] - fq(node["center"][i]))
                - fq(node["half_extents"][i])
                for i in range(3)
            )
        raise ValueError(node["op"])

    def exact_decide(self, value):
        return INNE if value < 0 else UTE if value > 0 else "GRÄNS"


def cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def dot(left, right):
    return sum(left[i] * right[i] for i in range(3))


def parallel_side_nohit(query):
    triangle, _, _, frame = query
    if frame.rows != IDENTITY_ROWS:
        return False
    x_values = [float(point[0]) for point in triangle]
    y_values = [float(point[1]) for point in triangle]
    if len(set(x_values)) == 1 or len(set(y_values)) == 1:
        return True
    return exact_vertical_determinant_zero(triangle, frame.origin)


class RayTriangle:
    threshold, mode = 0.0, "all_above"

    def cheap(self, query):
        triangle, origin, tmax, frame = query
        a, a_error = f64_local_f32(triangle[0], frame)
        b, b_error = f64_local_f32(triangle[1], frame)
        c, c_error = f64_local_f32(triangle[2], frame)
        o, o_error = f64_local_f32(origin, frame)
        e1, e1_error = f32_sub_vectors(b, b_error, a, a_error)
        e2, e2_error = f32_sub_vectors(c, c_error, a, a_error)
        h = (-e2[1], e2[0], 0.0)
        h_error = (e2_error[1], e2_error[0], 0.0)
        det, det_error = f32_dot(e1, e1_error, h, h_error)
        if abs(det) <= det_error:
            if parallel_side_nohit(query):
                return (-1.0,) * 5, (0.0,) * 5
            return (0.0,), (1.0,)
        s, s_error = f32_sub_vectors(o, o_error, a, a_error)
        r, r_error = _cross_with_errors(s, s_error, e1, e1_error)
        u_numerator, u_numerator_error = f32_dot(s, s_error, h, h_error)
        u, u_error = f32_divide_interval(
            u_numerator,
            u_numerator_error,
            det,
            det_error,
        )
        v, v_error = f32_divide_interval(
            r[2],
            r_error[2],
            det,
            det_error,
        )
        t_numerator, t_numerator_error = f32_dot(e2, e2_error, r, r_error)
        t, t_error = f32_divide_interval(
            t_numerator,
            t_numerator_error,
            det,
            det_error,
        )
        one_minus_u, one_minus_u_error = f32_sub(1.0, 0.0, u, u_error)
        barycentric_sum, barycentric_sum_error = f32_sub(
            one_minus_u,
            one_minus_u_error,
            v,
            v_error,
        )
        tmax_value, tmax_error = from_f64(tmax)
        remaining, remaining_error = f32_sub(
            tmax_value,
            tmax_error,
            t,
            t_error,
        )
        return (
            u,
            v,
            barycentric_sum,
            t,
            remaining,
        ), (
            u_error,
            v_error,
            barycentric_sum_error,
            t_error,
            remaining_error,
        )

    def refine(self, query):
        triangle, origin, tmax, _ = query
        a, b, c = [tuple(map(fq, point)) for point in triangle]
        o = tuple(map(fq, origin))
        direction = (Q(0), Q(0), Q(1))
        e1 = tuple(b[i] - a[i] for i in range(3))
        e2 = tuple(c[i] - a[i] for i in range(3))
        h = cross(direction, e2)
        det = dot(e1, h)
        if det == 0:
            return False
        s = tuple(o[i] - a[i] for i in range(3))
        r = cross(s, e1)
        u, v, t = dot(s, h) / det, dot(direction, r) / det, dot(e2, r) / det
        return (
            u >= 0
            and v >= 0
            and u + v <= 1
            and t >= 0
            and t <= fq(tmax)
        )

    def exact_decide(self, value):
        return INNE if value else UTE


def _cross_with_errors(left, left_error, right, right_error):
    values = []
    errors = []
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        first, first_error = f32_mul(left[j], right[k])
        second, second_error = f32_mul(left[k], right[j])
        value, error = f32_sub(
            first,
            first_error + left_error[j] + right_error[k],
            second,
            second_error + left_error[k] + right_error[j],
        )
        values.append(value)
        errors.append(error)
    return tuple(values), tuple(errors)


class Thickness:
    mode = "below"

    def __init__(self, threshold=0.2):
        self.threshold = threshold

    def cheap(self, query):
        lower, upper, frame = query
        if frame.rows == IDENTITY_ROWS:
            lower_local, lower_local_error = f64_sub(lower, frame.origin[2])
            upper_local, upper_local_error = f64_sub(upper, frame.origin[2])
            lower_value, lower_error = from_f64(lower_local)
            upper_value, upper_error = from_f64(upper_local)
            return f32_sub(
                upper_value,
                upper_error + upper_local_error,
                lower_value,
                lower_error + lower_local_error,
            )
        lower_value, lower_error = f64_local_f32((0.0, 0.0, lower), frame)
        upper_value, upper_error = f64_local_f32((0.0, 0.0, upper), frame)
        return f32_sub(
            upper_value[2],
            upper_error[2],
            lower_value[2],
            lower_error[2],
        )

    def refine(self, query):
        lower, upper, _ = query
        return fq(upper) - fq(lower)

    def exact_decide(self, value):
        threshold = fq(self.threshold)
        return INNE if value < threshold else UTE if value > threshold else "GRÄNS"
