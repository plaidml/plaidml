// Copyright 2020 Intel Corporation

#ifndef __PML_XSMM__OPS__
#define __PML_XSMM__OPS__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

include "mlir/Dialect/AffineOps/AffineOpsBase.td"

def XSMM_Dialect : Dialect {
  let name = "xsmm";
  let cppNamespace = "pmlc::dialect::xsmm";
}

class XSMM_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<XSMM_Dialect, mnemonic, traits> {
  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
  let verifier = [{ return verify$cppClass(*this); }];
}

def XSMM_GemmOp : XSMM_Op<"gemm"> {
  let arguments = (ins
    AnyMemRef:$c,
    AffineMapAttr:$cMap,
    AnyMemRef:$a,
    AffineMapAttr:$aMap,
    AnyMemRef:$b,
    AffineMapAttr:$bMap,
    I64ArrayAttr:$tile,
    Variadic<Index>:$mapOperands);

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}

#endif // __PML_XSMM__OPS__
