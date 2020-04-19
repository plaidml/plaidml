// Copyright 2020 Intel Corporation

#include "llvm/Support/raw_ostream.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/rt/memref.h"

extern "C" void plaidml_rt_trace(const char *msg) {
  llvm::outs() << msg << "\n";
  llvm::outs().flush();
}

namespace {
struct Registration {
  Registration() {
    pmlc::compiler::registerSymbol("plaidml_rt_trace", plaidml_rt_trace);
  }
};
static Registration reg;
} // namespace
