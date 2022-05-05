#ifndef __PMLC_DIALECT_AFFINEX_PASSES__
#define __PMLC_DIALECT_AFFINEX_PASSES__

include "mlir/Pass/PassBase.td"

def AffinexLoopUnroll : Pass<"affinex-loop-unroll", "mlir::FuncOp"> {
  let summary = "Fully unrolls AffineFor loops which meet the operation limit.";
  let constructor = "pmlc::dialect::affinex::createAffinexLoopUnroll()";
  let options = [
  Option<"operationLimit", "operation-limit", "uint64_t",
          /*default=*/"256",
          "Maximum number of operations allowed in an unrolled AffineFor loop">,
  ];
}

def AffinexMemRefDataFlowOpt : Pass<"affinex-memref-dataflow-opt", "mlir::FuncOp"> {
  let summary = "Performs store to load forwarding.";
  let constructor = "pmlc::dialect::affinex::createAffinexMemRefDataFlowOpt()";
}

def AffinexDeadMemRefElimination : Pass<"affinex-dead-memref-elimination", "mlir::FuncOp"> {
  let summary = "Eliminate unused MemRefs.";
  let constructor = "pmlc::dialect::affinex::createAffinexDeadMemRefElimination()";
}

#endif // __PMLC_DIALECT_AFFINEX_PASSES__
