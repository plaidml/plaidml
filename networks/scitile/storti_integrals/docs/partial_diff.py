def partial(F, wrt, delta):
    F_neg = -F
    dims = edsl.TensorDims(3)
    x, y, z = edsl.TensorIndexes(3)
    F.bind_dims(*dims)
    O = edsl.TensorOutput(*dims)
    if wrt == 'x':
        O[x, y, z] = F[x + 1, y, z] + F_neg[x - 1, y, z]
    elif wrt == 'y':
        O[x, y, z] = F[x, y + 1, z] + F_neg[x, y - 1, z]
    elif wrt == 'z':
        O[x, y, z] = F[x, y, z + 1] + F_neg[x, y, z - 1]
    return O / (2.0 * delta)