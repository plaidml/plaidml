// Copyright 2019, Intel Corporation

#ifndef __PML_TILE_INTERFACES__
#define __PML_TILE_INTERFACES__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def ContractionOpInterface : OpInterface<"ContractionOp"> {
  let methods = [
    StaticInterfaceMethod<"TODO", "Operation*", "create", (ins
      "OpBuilder*":$builder,
      "Location":$loc,
      "Value*":$size,
      "ArrayRef<Value*>":$srcs,
      "Value*":$sink
    )>,
    InterfaceMethod<"TODO", "std::vector<Value*>", "getSourceIndexMaps">,
    InterfaceMethod<"TODO", "Value*", "getSinkIndexMap">,
    InterfaceMethod<"TODO", "Value*", "getSizeMap">,
    InterfaceMethod<"TODO", "unsigned", "getNumOperands">,
    InterfaceMethod<"TODO", "AggregationKind", "getAggregationKind">,
    InterfaceMethod<"TODO", "CombinationKind", "getCombinationKind">,
    InterfaceMethod<"TODO", "Value*", "getInitializer">,
  ];
}

def SpecialOpInterface : OpInterface<"SpecialOp">;

#endif // __PML_TILE_INTERFACES__
