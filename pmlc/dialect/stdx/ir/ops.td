// Copyright 2020 Intel Corporation

#ifndef __PML_STDX_OPS__
#define __PML_STDX_OPS__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

include "mlir/Interfaces/SideEffects.td"


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

def FPToSIOp :
    StdX_Op<"fptosi", [NoSideEffect]> {
  let summary = "cast from floating-point to signed integer type";
  let arguments = (ins AnyType:$result);
  let results = (outs AnyType);
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value source, Type destType", [{
       buildFPToSIOp(builder, result, source, destType);
  }]>];
}

def FPToUIOp :
    StdX_Op<"fptoui", [NoSideEffect]> {
  let summary = "cast from floating-point to unsigned integer type";
  let arguments = (ins AnyType:$result);
  let results = (outs AnyType);
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value source, Type destType", [{
       buildFPToUIOp(builder, result, source, destType);
  }]>];
}

def ReshapeOp : StdX_Op<"reshape"> {
  let summary = "tensor reshape operation";
  let arguments = (ins
    AnyMemRef:$tensor,
    Variadic<Index>:$dims);
  let results = (outs AnyMemRef:$result);
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value tensor, ArrayRef<Value> dims, MemRefType destType", [{
        buildReshapeOp(builder, result, tensor, dims, destType);
  }]>];
}

#endif // __PML_STDX_OPS__
