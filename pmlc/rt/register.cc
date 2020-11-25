// Copyright 2020 Intel Corporation

#include "pmlc/rt/register.h"

namespace pmlc::rt {

extern void registerBuiltins();
extern void registerPrng();
extern void registerXsmm();

void registerRuntime() {
  registerBuiltins();
  registerPrng();
  registerXsmm();
}

} // namespace pmlc::rt
