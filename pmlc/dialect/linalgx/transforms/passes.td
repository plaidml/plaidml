#ifndef __PMLC_DIALECT_LINALGX_PASSES__
#define __PMLC_DIALECT_LINALGX_PASSES__

include "mlir/Pass/PassBase.td"

def RegulateConvolution : FunctionPass<"linalgx-regulate-convolution"> {
  let summary = "Regulate convolutions";
  let constructor = "pmlc::dialect::linalgx::createRegulateConvolutionPass()";
}

#endif // __PMLC_DIALECT_LINALGX_PASSES__
