// Copyright 2019, Intel Corporation

include "mlir/IR/OpBase.td"

def HirDialect : Dialect {
  let name = "pml_hir";
  let cppNamespace = "pmlc::dialect::hir";
}

class HirOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<HirDialect, mnemonic, traits> {}

def CallOp : HirOp<"call"> {
  let summary = "Call operation";
}

def ContractionOp : HirOp<"contraction"> {
  let summary = "Contraction operation";
}

// def DimOp : HirOp<"dim"> {
//   let summary = "Symbolic tensor dimension";
// }

def PlaceholderOp : HirOp<"decl"> {
  let summary = "Placeholder operation";
  let arguments = (ins AnyAttr:$value);
  let results = (outs ScalarType:$result);
}

