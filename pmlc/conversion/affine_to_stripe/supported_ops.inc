// Copyright 2019 Intel Corporation

// List of Affine ops with Stripe conversion support.

#ifndef AFFINE_OP
#error Undefined AFFINE_OP macro.
#endif

#ifndef AFFINE_LAST_OP
#define AFFINE_LAST_OP(OP) AFFINE_OP(OP)
#endif

AFFINE_OP(AffineForOp)
AFFINE_OP(AffineTerminatorOp)
AFFINE_LAST_OP(ConstantIndexOp)

#undef AFFINE_OP
#undef AFFINE_LAST_OP
