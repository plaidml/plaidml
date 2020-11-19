// Copyright 2020 Intel Corporation

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def LayerDialect : Dialect {
  let name = "layer";
  let cppNamespace = "pmlc::dialect::layer";
}

class Layer_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<LayerDialect, mnemonic, traits> {
}


def LayerOp : Layer_Op<"layer", [NoSideEffect, IsolatedFromAbove]> {
  let summary = "layer operation";
  let arguments = (ins
    Variadic<AnyType>:$operands,
    StrAttr:$op,
    DictionaryAttr:$attrs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "mlir::OpBuilder &builder, "
    "mlir::OperationState& result, "
    "mlir::StringRef op, "
    "mlir::ArrayRef<mlir::Value> operands, "
    "mlir::ArrayRef<mlir::Value> innerResults, "
    "mlir::DictionaryAttr attrs"
  >];

  let extraClassDeclaration = [{
    mlir::Block *getBody();
  }];

  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
}

def LayerReturnOp : Layer_Op<"layer.return", [
  NoSideEffect,
  ReturnLike,
  Terminator,
  HasParent<"LayerOp">
]> {
  let summary = "Return results from a layer";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}
