// Copyright 2020 Intel Corporation

#ifndef __PML_STDX_OPS__
#define __PML_STDX_OPS__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

include "mlir/Interfaces/SideEffectInterfaces.td"


def StdX_Dialect : Dialect {
  let name = "stdx";
  let cppNamespace = "pmlc::dialect::stdx";
}

class StdX_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<StdX_Dialect, mnemonic, traits> {
  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
  let verifier = [{ return verify$cppClass(*this); }];
}

def FPToUIOp :
    StdX_Op<"fptoui", [NoSideEffect]> {
  let summary = "cast from floating-point to unsigned integer type";
  let arguments = (ins AnyType:$result);
  let results = (outs AnyType);
  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value source, Type destType",
    [{ buildFPToUIOp(builder, result, source, destType); }]
  >];
}

class StdX_Plain_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<StdX_Dialect, mnemonic, traits> {
}

def ReshapeOp : StdX_Plain_Op<"reshape"> {
  let summary = "tensor reshape operation";
  let arguments = (ins AnyMemRef:$tensor);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    `(` $tensor `)` attr-dict `:` functional-type($tensor, results)
  }];
}

def ASinOp : StdX_Plain_Op<"asin", [NoSideEffect]> {
  let summary = "elementwise arcsine";
  let arguments = (ins AnyType:$tensor);
  let results = (outs AnyType);
  let assemblyFormat = [{
    `(` $tensor `)` attr-dict `:` functional-type(operands, results)
  }];
}

#endif // __PML_STDX_OPS__
