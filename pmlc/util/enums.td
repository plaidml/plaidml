#ifndef __PML_UTIL_ENUMS__
#define __PML_UTIL_ENUMS__

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
  let cppNamespace = "pmlc::util";
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
  let cppNamespace = "pmlc::util";
  let returnType = "CombinationKind";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getValue().getZExtValue())";
}

def DataType : I64EnumAttr<
    "DataType",
    "Core data types",
    [
      I64EnumAttrCase<"invalid", 0>,
      I64EnumAttrCase<"i1",      1>,
      I64EnumAttrCase<"i8",      2>,
      I64EnumAttrCase<"u8",      3>,
      I64EnumAttrCase<"i16",     4>,
      I64EnumAttrCase<"u16",     5>,
      I64EnumAttrCase<"i32",     6>,
      I64EnumAttrCase<"u32",     7>,
      I64EnumAttrCase<"i64",     8>,
      I64EnumAttrCase<"u64",     9>,
      I64EnumAttrCase<"bf16",   10>,
      I64EnumAttrCase<"f16",    11>,
      I64EnumAttrCase<"f32",    12>,
      I64EnumAttrCase<"f64",    13>,
    ]> {
  let cppNamespace = "pmlc::util";
  let returnType = "DataType";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getValue().getZExtValue())";
}

#endif // __PML_UTIL_ENUMS__
