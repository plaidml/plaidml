// Copyright 2020 Intel Corporation

#include "pmlc/rt/register.h"

namespace pmlc::rt {

extern void registerBuiltins();
extern void registerPrng();
extern void registerBoundsCheck();
extern void registerXsmm();

void registerRuntime() {
  registerBuiltins();
  registerPrng();
  registerBoundsCheck();
  registerXsmm();
}

} // namespace pmlc::rt
