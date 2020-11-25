// Copyright 2020 Intel Corporation

#include "pmlc/compiler/program.h"
#include "pmlc/target/intel_gen/pipeline.h"
#include "pmlc/target/intel_gen_ocl_spirv/pipeline.h"
#include "pmlc/target/x86/pipeline.h"

namespace pmlc::compiler {

void registerTargets() {
  pmlc::target::intel_gen::registerTarget();
  pmlc::target::intel_gen_ocl_spirv::registerTarget();
  pmlc::target::x86::registerTarget();
}

} // namespace pmlc::compiler
