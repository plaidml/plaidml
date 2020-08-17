// Copyright 2020 Intel Corporation

#ifndef __PML_STDX_OPS__
#define __PML_STDX_OPS__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"


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

def FPToUIOp : StdX_Op<"fptoui", [NoSideEffect]> {
  let summary = "cast from floating-point to unsigned integer type";
  let arguments = (ins AnyType:$result);
  let results = (outs AnyType);
  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value source, Type destType",
    [{ buildFPToUIOp(builder, result, source, destType); }]
  >];
}

def UIToFPOp : StdX_Op<"uitofp", [NoSideEffect]> {
  let summary = "cast from unsigned integer to floating-point type";
  let arguments = (ins AnyType:$result);
  let results = (outs AnyType);
  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState &result, Value source, Type destType",
    [{ buildUIToFPOp(builder, result, source, destType); }]
  >];
}

def ReshapeOp : StdX_Op<"reshape",
  [DeclareOpInterfaceMethods<ViewLikeOpInterface>]> {
  let summary = "tensor reshape operation";
  let arguments = (ins AnyMemRef:$tensor);
  let results = (outs AnyMemRef:$result);
  let assemblyFormat = [{
    `(` $tensor `)` attr-dict `:` functional-type($tensor, results)
  }];
  let verifier = ?;
}

class StdX_UnaryFuncOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<StdX_Dialect, mnemonic, traits> {
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType);
  let assemblyFormat = [{
    `(` $value `)` attr-dict `:` functional-type(operands, results)
  }];
  let verifier = ?;
}

def ACosOp : StdX_UnaryFuncOp<"acos"> {
  let summary = "arccosine";
}

def ASinOp : StdX_UnaryFuncOp<"asin"> {
  let summary = "arcsine";
}

def ATanOp : StdX_UnaryFuncOp<"atan"> {
  let summary = "arctangent";
}

def CosHOp : StdX_UnaryFuncOp<"cosh"> {
  let summary = "hyperbolic cosine";
}

def ErfOp : StdX_UnaryFuncOp<"erf"> {
  let summary = "Gauss error function";
}

def FloorOp : StdX_UnaryFuncOp<"floor"> {
  let summary = "integer floor";
}

def PowOp : StdX_Op<"pow"> {
  let summary = "raise to power";
  let arguments = (ins AnyType:$base, AnyType:$exponent);
  let results = (outs AnyType);
  let assemblyFormat = [{
    `(` $base `,` $exponent `)` attr-dict `:` functional-type(operands, results)
  }];
  let verifier = ?;
}

def RoundOp : StdX_UnaryFuncOp<"round"> {
  let summary = "round to nearest integer";
}

def SinHOp : StdX_UnaryFuncOp<"sinh"> {
  let summary = "hyperbolic sine";
}

def TanOp : StdX_UnaryFuncOp<"tan"> {
  let summary = "tangent";
}

#endif // __PML_STDX_OPS__
