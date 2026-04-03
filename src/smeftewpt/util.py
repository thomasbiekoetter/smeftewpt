import cmath


def is_equal(a, b, eps=1e-4):
    if abs(a - b) < eps:
        y = True
    else:
        y = False
    return y


def logsave(x, y, IReps=1e-10):
    return cmath.log((x / (y + IReps)) + IReps).real
