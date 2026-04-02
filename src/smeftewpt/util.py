def is_equal(a, b, eps=1e-4):
    if abs(a - b) < eps:
        y = True
    else:
        y = False
    return y
