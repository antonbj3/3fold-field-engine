import ctypes
import math

U32 = 2.0 ** -24
U64 = 2.0 ** -53
_FLOAT32 = ctypes.c_float
IDENTITY_ROWS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def to_f32(value):
    return _FLOAT32(float(value)).value


def from_f64(value):
    value = float(value)
    return to_f32(value), U32 * abs(value)


def f32_add(x, ex, y, ey):
    if x == 0.0:
        return y, ey
    if y == 0.0:
        return x, ex
    value = to_f32(x + y)
    return value, ex + ey + U32 * (abs(x) + abs(value))


def f32_sub(x, ex, y, ey):
    return f32_add(x, ex, -y, ey)


def f32_mul(x, y):
    if x == 0.0 or y == 0.0:
        return 0.0, 0.0
    return to_f32(x * y), U32 * (abs(x) + abs(y))


def f64_add(x, ex, y, ey):
    if x == 0.0:
        return y, ey
    if y == 0.0:
        return x, ex
    value = float(x + y)
    return value, ex + ey + U64 * (abs(x) + abs(value))


def f64_sub(x, y):
    value = float(x - y)
    return value, U64 * (abs(x) + abs(value))


def f64_mul(x, y):
    if x == 0.0 or y == 0.0:
        return 0.0, 0.0
    value = float(x * y)
    return value, U64 * (abs(x) + abs(y))


def f32_dot(x, ex, y, ey):
    indices = [i for i in range(len(x)) if x[i] != 0.0 and y[i] != 0.0]
    if not indices:
        return 0.0, 0.0
    i = indices[0]
    value, error = f32_mul(x[i], y[i])
    error += ex[i] + ey[i]
    for i in indices[1:]:
        product, product_error = f32_mul(x[i], y[i])
        value, error = f32_add(value, error, product, product_error + ex[i] + ey[i])
    return value, error


def f64_local_f32(point, frame):
    if frame.rows == IDENTITY_ROWS:
        coordinates = []
        errors = []
        for i in range(3):
            delta, delta_error = f64_sub(float(point[i]), float(frame.origin[i]))
            value, error = from_f64(delta)
            coordinates.append(value)
            errors.append(error + delta_error)
        return tuple(coordinates), tuple(errors)
    coordinates = []
    errors = []
    for row in frame.rows:
        terms = []
        term_errors = []
        for i in range(3):
            if row[i] == 0.0:
                continue
            delta, delta_error = f64_sub(float(point[i]), float(frame.origin[i]))
            product, product_error = f64_mul(float(row[i]), delta)
            terms.append((product, product_error + delta_error))
        value, error = terms[0]
        for product, product_error in terms[1:]:
            value, error = f64_add(value, error, product, product_error)
        rounded, rounded_error = from_f64(value)
        coordinates.append(rounded)
        errors.append(rounded_error + error)
    return tuple(coordinates), tuple(errors)


def f32_sub_vectors(x, ex, y, ey):
    values = []
    errors = []
    for i in range(3):
        value, error = f32_sub(x[i], ex[i], y[i], ey[i])
        values.append(value)
        errors.append(error)
    return tuple(values), tuple(errors)


def f32_cross(x, ex, y, ey):
    values = []
    errors = []
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        left, left_error = f32_mul(x[j], y[k])
        right, right_error = f32_mul(x[k], y[j])
        value, error = f32_sub(left, left_error + ex[j] + ey[k], right, right_error + ex[k] + ey[j])
        values.append(value)
        errors.append(error)
    return tuple(values), tuple(errors)


def f32_divide_interval(numerator, numerator_error, denominator, denominator_error):
    denominator_low = abs(denominator) - denominator_error
    denominator_high = abs(denominator) + denominator_error
    if not math.isfinite(denominator_low) or denominator_low <= 0.0:
        return 0.0, math.inf
    numerator_values = (numerator - numerator_error, numerator + numerator_error)
    denominator_values = (denominator - denominator_error, denominator + denominator_error)
    bounds = [n / d for n in numerator_values for d in denominator_values if d != 0.0]
    if not bounds or not all(math.isfinite(value) for value in bounds):
        return 0.0, math.inf
    value = to_f32(numerator / denominator)
    error = max(U32 * abs(value), abs(value - min(bounds)), abs(value - max(bounds)))
    return value, error


def enclose_max(values, errors):
    lower = max(value - error for value, error in zip(values, errors))
    upper = max(value + error for value, error in zip(values, errors))
    return (lower + upper) * 0.5, (upper - lower) * 0.5


def dyadic_difference(left, right):
    left_numerator, left_denominator = float(left).as_integer_ratio()
    right_numerator, right_denominator = float(right).as_integer_ratio()
    denominator = max(left_denominator, right_denominator)
    return (
        left_numerator * (denominator // left_denominator)
        - right_numerator * (denominator // right_denominator),
        denominator,
    )


def exact_vertical_determinant_zero(triangle, origin):
    axes = []
    for axis in (0, 1):
        differences = [
            dyadic_difference(point[axis], origin[axis])
            for point in triangle
        ]
        axes.append([numerator for numerator, _ in differences])
    e1x, e2x = axes[0][0], axes[0][1]
    e1y, e2y = axes[1][0], axes[1][1]
    return e1x * e2y - e1y * e2x == 0
