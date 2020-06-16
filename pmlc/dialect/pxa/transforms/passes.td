#ifndef __PMLC_DIALECT_PXA_PASSES__
#define __PMLC_DIALECT_PXA_PASSES__

include "mlir/Pass/PassBase.td"

def AutoTileExample : FunctionPass<"pxa-autotile-example"> {
  let summary = "Tile all dimensions by 10";
  let constructor = "pmlc::dialect::pxa::createAutoTileExamplePass()";
}

def Fusion : FunctionPass<"pxa-fusion"> {
  let summary = "Fuse blocks that are compatible";
  let constructor = "pmlc::dialect::pxa::createFusionPass()";
}

def MemRefDataFlowOpt : FunctionPass<"pxa-dataflow-opt"> {
  let summary = "Perform reduce/load forwarding for memrefs";
  let constructor = "pmlc::dialect::pxa::createMemRefDataFlowOptPass()";
}

def TestStrideInfo : Pass<"pxa-stride-info"> {
  let summary = "Report stride data for all loads/stores for unit tests";
  let constructor = "pmlc::dialect::pxa::createTestStrideInfoPass()";
}

#endif // __PMLC_DIALECT_PXA_PASSES__
