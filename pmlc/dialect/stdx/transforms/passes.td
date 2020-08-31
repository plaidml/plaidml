#ifndef __PMLC_DIALECT_STDX_PASSES__
#define __PMLC_DIALECT_STDX_PASSES__

include "mlir/Pass/PassBase.td"

def BoundsCheck : FunctionPass<"stdx-check-bounds"> {
  let summary = "Check bounds for Load and Store Ops";
  let constructor = "pmlc::dialect::stdx::createBoundsCheckPass()";
}

def I1StorageToI32 : FunctionPass<"stdx-i1-storage-to-i32"> {
  let summary = "Convert i1 storage type to i32";
  let constructor = "pmlc::dialect::stdx::createI1StorageToI32Pass()";
}

def SubgroupBlockOps : FunctionPass<"stdx-subgroup-block-ops"> {
  let summary = "Transform to intel_subgroup_block_reads and writes";
  let constructor = "pmlc::dialect::stdx::createSubgroupBlockOpsPass()";
  let dependentDialects = ["StdXDialect"];
}

def SubgroupBroadcast : FunctionPass<"stdx-subgroup-broadcast"> {
  let summary = "Perform unvectorization for subgroups broadcast usage";
  let constructor = "pmlc::dialect::stdx::createSubgroupBroadcastPass()";
  let dependentDialects = ["StdXDialect"];
}

#endif // __PMLC_DIALECT_STDX_PASSES__
