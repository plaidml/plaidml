#ifndef __PML_ELTWISE_ENUMS__
#define __PML_ELTWISE_ENUMS__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def AggregationKind : I64EnumAttr<
    "AggregationKind",
    "Operation to aggregate with",
    [
      I64EnumAttrCase<"assign", 0>,
      I64EnumAttrCase<"add", 1>,
      I64EnumAttrCase<"max", 2>,
      I64EnumAttrCase<"min", 3>,
      I64EnumAttrCase<"mul", 4>,
    ]> {
  let cppNamespace = "pmlc::dialect::eltwise";
  let returnType = "AggregationKind";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getValue().getZExtValue())";
}

def CombinationKind : I64EnumAttr<
    "CombinationKind",
    "Operation to combine with",
    [
      I64EnumAttrCase<"none", 0>,
      I64EnumAttrCase<"add", 1>,
      I64EnumAttrCase<"cond", 2>,
      I64EnumAttrCase<"eq", 3>,
      I64EnumAttrCase<"mul", 4>,
    ]> {
  let cppNamespace = "pmlc::dialect::eltwise";
  let returnType = "CombinationKind";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getValue().getZExtValue())";
}

#endif // __PML_ELTWISE_ENUMS__
