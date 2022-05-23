#ifndef __PMLC_DIALECT_LINALGX_PASSES__
#define __PMLC_DIALECT_LINALGX_PASSES__

include "mlir/Pass/PassBase.td"

def RegulateDepthwise : FunctionPass<"linalgx-regulate-depthwise"> {
  let summary = "Regulate depthwise convolutions";
  let constructor = "pmlc::dialect::linalgx::createRegulateDepthwisePass()";
}

def NameConvVariables : FunctionPass<"linalgx-name-conv-variables"> {
  let summary = "Name variables of convolution as n, c, h, w and so on";
  let constructor = "pmlc::dialect::linalgx::createNameConvVariablesPass()";
}
#endif // __PMLC_DIALECT_LINALGX_PASSES__
