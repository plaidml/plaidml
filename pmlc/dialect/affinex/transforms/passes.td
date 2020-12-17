#ifndef __PMLC_DIALECT_AFFINEX_PASSES__
#define __PMLC_DIALECT_AFFINEX_PASSES__

include "mlir/Pass/PassBase.td"

def AffinexMemRefDataFlowOpt : FunctionPass<"affinex-memref-dataflow-opt"> {
  let summary = "Performs store to load forwarding.";
  let constructor = "pmlc::dialect::affinex::createAffinexMemRefDataFlowOpt()";
}

def AffinexDeadMemRefElimination : FunctionPass<"affinex-dead-memref-elimination"> {
  let summary = "Eliminate unused MemRefs.";
  let constructor = "pmlc::dialect::affinex::createAffinexDeadMemRefElimination()";
}

#endif // __PMLC_DIALECT_AFFINEX_PASSES__
