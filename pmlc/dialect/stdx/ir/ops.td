// Copyright 2020 Intel Corporation

#ifndef __PML_STDX_OPS__
#define __PML_STDX_OPS__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

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

def AtomicRMWOp : StdX_Op<"atomic_rmw"> {
  let arguments = (ins AnyMemRef:$memref, Variadic<Index>:$indices);
  let results = (outs AnyType:$res);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    Block *getBody() { return &body().front(); }
    Value getInductionVar() { return getBody()->getArgument(0); }
    OpBuilder getBodyBuilder() {
      return OpBuilder(getBody(), std::prev(getBody()->end()));
    }
    MemRefType getMemRefType() {
      return memref().getType().cast<MemRefType>();
    }
  }];
}

def AtomicRMWYieldOp :
    StdX_Op<"atomic_rmw.yield", [HasParent<"AtomicRMWOp">, Terminator]> {
  let summary = "terminator for atomic_rmw operation";
  let arguments = (ins AnyType:$result);
}

#endif // __PML_STDX_OPS__
