#pragma once

#include "pmlc/conversion/SCFToGPU/SCFToGPUPass.h"

namespace pmlc::conversion::scf_to_gpu {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/SCFToGPU/Passes.h.inc"

} // namespace pmlc::conversion::scf_to_gpu
