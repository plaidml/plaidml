// Copyright 2020 Intel Corporation

#ifndef __PML_XSMM_OPS__
#define __PML_XSMM_OPS__

include "mlir/Interfaces/SideEffectInterfaces.td"

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

#ifndef AFFINE_OPS_BASE
include "mlir/Dialect/Affine/IR/AffineOpsBase.td"
#endif

def XSMM_Dialect : Dialect {
  let name = "xsmm";
  let cppNamespace = "pmlc::dialect::xsmm";
}

class XSMM_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<XSMM_Dialect, mnemonic, traits> {
  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
}

def XSMM_GemmDispatchOp : XSMM_Op<"gemm.dispatch", [NoSideEffect]> {
  let summary = "query LIBXSMM's code registry based on GEMM shape";
  let arguments = (ins I64ArrayAttr:$tile, I64ArrayAttr:$tileld);
  let results = (outs I64:$ptr);
  let printer = ?;
  let parser = ?;
  let assemblyFormat = "$tile `,` $tileld attr-dict";
}

def XSMM_GemmInvokeOp : XSMM_Op<"gemm.invoke"> {
  let summary = "invoke a previously queried LIBXSMM kernel";
  let arguments = (ins I64:$ptr,
    AnyMemRef:$c, AffineMapAttr:$cAccessMap,
    AnyMemRef:$a, AffineMapAttr:$aAccessMap,
    AnyMemRef:$b, AffineMapAttr:$bAccessMap,
    Variadic<Index>:$mapOperands);
  let results = (outs AnyMemRef:$c_out);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}

#endif // __PML_XSMM_OPS__
