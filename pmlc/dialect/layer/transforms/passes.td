#ifndef __PMLC_DIALECT_LAYER_PASSES__
#define __PMLC_DIALECT_LAYER_PASSES__

include "mlir/Pass/PassBase.td"

def InlineLayers : FunctionPass<"layer-inline-layers"> {
  let summary = "Inline layers";
  let constructor = "pmlc::dialect::layer::createInlineLayersPass()";
}

#endif // __PMLC_DIALECT_LAYER_PASSES__
