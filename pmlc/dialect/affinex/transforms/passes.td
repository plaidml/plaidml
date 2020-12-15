#ifndef __PMLC_DIALECT_AFFINEX_PASSES__
#define __PMLC_DIALECT_AFFINEX_PASSES__

include "mlir/Pass/PassBase.td"

def AffinexMemRefDataFlowOpt : FunctionPass<"affinex-memref-dataflow-opt"> {
  let summary = "Performs store to load forwarding.";
  let constructor = "pmlc::dialect::affinex::createAffinexMemRefDataFlowOpt()";
}

#endif // __PMLC_DIALECT_AFFINEX_PASSES__
