// Copyright 2019, Intel Corporation

#ifndef __PML_PXA_OPS__
#define __PML_PXA_OPS__

#ifndef AFFINE_OPS_BASE
include "mlir/Dialect/AffineOps/AffineOpsBase.td"
#endif

#ifndef MLIR_LOOPLIKEINTERFACE
include "mlir/Transforms/LoopLikeInterface.td"
#endif

#ifndef __PML_UTIL_ENUMS__
include "pmlc/util/enums.td"
#endif

def PXA_Dialect : Dialect {
  let name = "pxa";
  let cppNamespace = "pmlc::dialect::pxa";
}

class PXA_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<PXA_Dialect, mnemonic, traits>;

class PXA_OpWithPP<string mnemonic, list<OpTrait> traits = []> :
    PXA_Op<mnemonic, traits> {
  let printer = [{ print$cppClass(p, *this); }];
  let parser = [{ return parse$cppClass(parser, result); }];
}

def AnyStdScalar : AnyTypeOf<[AnyFloat, AnySignlessInteger]> {}

def AffineReduceOp : PXA_OpWithPP<"reduce"> {
  let summary = "affine reduction operation";
  let arguments = (ins
    AggregationKind:$agg,
    AnyStdScalar:$val,
    AnyMemRef:$out,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
    AffineMap getAffineMap() { return map(); }
    operand_range getMapOperands() { return idxs(); }
  }];
}

#endif // __PML_PXA_OPS__
