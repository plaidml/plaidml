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
    "Elemental data types",
    [
      I64EnumAttrCase<"invalid", 0>,
      I64EnumAttrCase<"six", 1>,
      I64EnumAttrCase<"uix", 2>,
      I64EnumAttrCase<"fx", 3>,
      I64EnumAttrCase<"i1", 4>,
      I64EnumAttrCase<"si8", 5>,
      I64EnumAttrCase<"ui8", 6>,
      I64EnumAttrCase<"si16", 7>,
      I64EnumAttrCase<"ui16", 8>,
      I64EnumAttrCase<"si32", 9>,
      I64EnumAttrCase<"ui32", 10>,
      I64EnumAttrCase<"si64", 11>,
      I64EnumAttrCase<"ui64", 12>,
      I64EnumAttrCase<"bf16", 13>,
      I64EnumAttrCase<"f16", 14>,
      I64EnumAttrCase<"f32", 15>,
      I64EnumAttrCase<"f64", 16>,
    ]> {
  let cppNamespace = "pmlc::util";
  let returnType = "DataType";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getValue().getZExtValue())";
}

def InterpolationMode : I64EnumAttr<
    "InterpolationMode",
    "Modes that controls interpolation",
    [
     I64EnumAttrCase<"nearest", 0>,
     I64EnumAttrCase<"linear", 1>,
     I64EnumAttrCase<"cubic", 2>,
    ]> {
  let cppNamespace = "pmlc::util";
  let returnType = "InterpolationMode";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getValue().getZExtValue())";
}

def NearestMode : I64EnumAttr<
    "NearestMode",
    "Specifies round mode when InterpolationMode == nearest ",
    [
     I64EnumAttrCase<"round_prefer_floor", 0>,
     I64EnumAttrCase<"round_prefer_ceil", 1>,
     I64EnumAttrCase<"floor", 2>,
     I64EnumAttrCase<"ceil", 3>,
     I64EnumAttrCase<"simple", 4>,
    ]> {
  let cppNamespace = "pmlc::util";
  let returnType = "NearestMode";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getValue().getZExtValue())";
}

def ScatterMode : I64EnumAttr<
    "ScatterMode",
    "Scatter update mode",
    [
      I64EnumAttrCase<"normal", 0>,
      I64EnumAttrCase<"update_slice", 1>,
      I64EnumAttrCase<"update_elt", 2>,
      I64EnumAttrCase<"update_nd", 3>,
    ]> {
  let cppNamespace = "pmlc::util";
  let returnType = "ScatterMode";
  let convertFromStorage = "static_cast<" # returnType # ">($_self.getValue().getZExtValue())";
}

#endif // __PML_UTIL_ENUMS__
