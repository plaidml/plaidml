// Copyright 2020 Intel Corporation

#include "llvm/Support/raw_ostream.h"

extern "C" void plaidml_rt_trace(const char *msg) {
  llvm::outs() << msg << "\n";
  llvm::outs().flush();
}
