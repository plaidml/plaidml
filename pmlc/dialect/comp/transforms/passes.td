// Copyright 2020 Intel Corporation

#ifndef __PMLC_DIALECT_COMP_PASSES__
#define __PMLC_DIALECT_COMP_PASSES__

include "mlir/Pass/PassBase.td"

def ExecEnvCoalescing : FunctionPass<"pmlc-execenv-coalescing"> {
  let summary = "Combines same execution environments.";
  let constructor = "pmlc::dialect::comp::createExecEnvCoalescingPass()";
}

#endif // __PMLC_DIALECT_COMP_PASSES__
