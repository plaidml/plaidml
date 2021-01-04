// Copyright 2020 Intel Corporation

#include "pmlc/compiler/program.h"
#include "pmlc/target/intel_gen/pipeline.h"
#include "pmlc/target/intel_gen_ocl_spirv/pipeline.h"
#ifdef PML_ENABLE_LEVEL_ZERO
#include "pmlc/target/intel_level_zero/pipeline.h"
#endif
#include "pmlc/target/x86/pipeline.h"

namespace pmlc::compiler {

void registerTargets() {
  pmlc::target::intel_gen::registerTarget();
  pmlc::target::intel_gen_ocl_spirv::registerTarget();
#ifdef PML_ENABLE_LEVEL_ZERO
  pmlc::target::intel_level_zero::registerTarget();
#endif
  pmlc::target::x86::registerTarget();
}

} // namespace pmlc::compiler
