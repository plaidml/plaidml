def partial(F, wrt, delta):
    F_neg = -F
    dims = edsl.TensorDims(3)
    x, y, z = edsl.TensorIndexes(3)
    F.bind_dims(*dims)
    OC = edsl.Contraction().outShape(*dims)
    if wrt == 'x':
        O = OC.outAccess(x, y, z).assign(F[x + 1, y, z] + F_neg[x - 1, y, z]).build()
    elif wrt == 'y':
        O = OC.outAccess(x, y, z).assign([x, y + 1, z] + F_neg[x, y - 1, z]).build()
    elif wrt == 'z':
        O = OC.outAccess(x, y, z).assign([x, y, z + 1] + F_neg[x, y, z - 1]).build()
    return O / (2.0 * delta)
