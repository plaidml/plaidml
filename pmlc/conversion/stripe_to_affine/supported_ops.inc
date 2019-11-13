// Copyright 2019 Intel Corporation

// List of Stripe ops with affine conversion support.

#ifndef STRIPE_OP
#error Undefined STRIPE_OP macro.
#endif

#ifndef STRIPE_LAST_OP
#define STRIPE_LAST_OP(OP) STRIPE_OP(OP)
#endif

STRIPE_OP(AffinePolyOp)
STRIPE_OP(LoadOp)
STRIPE_OP(ParallelForOp)
STRIPE_OP(RefineOp)
STRIPE_LAST_OP(TerminateOp)

#undef STRIPE_OP
#undef STRIPE_LAST_OP
