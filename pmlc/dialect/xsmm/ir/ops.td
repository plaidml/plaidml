// Copyright 2020 Intel Corporation

#ifndef __PML_XSMM__OPS__
#define __PML_XSMM__OPS__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

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

def XSMM_SMMDispatchOp : XSMM_Op<"smmdispatch"> {
  let arguments = (ins AnyMemRef:$memref, Variadic<Index>:$indices);
  let results = (outs AnyType:$res);
}

//def XSMM_DMMDispatchOp : XSMM_Op<"dmmdispatch"> {
//  let arguments = (ins AnyMemRef:$memref, Variadic<Index>:$indices);
//  let results = (outs AnyType:$res);
//}

#endif // __PML_XSMM__OPS__
