#ifndef __PMLC_DIALECT_LINALGX_PASSES__
#define __PMLC_DIALECT_LINALGX_PASSES__

include "mlir/Pass/PassBase.td"

def RegulateDepthwise : Pass<"linalgx-regulate-depthwise", "mlir::func::FuncOp"> {
  let summary = "Regulate depthwise convolutions";
  let constructor = "pmlc::dialect::linalgx::createRegulateDepthwisePass()";
}

#endif // __PMLC_DIALECT_LINALGX_PASSES__
