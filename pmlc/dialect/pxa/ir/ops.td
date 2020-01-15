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
  let printer = [{ print$cppClass(&p, *this); }];
  let parser = [{ return parse$cppClass(&parser, result); }];
}

def AnyStdScalar : AnyTypeOf<[AnyFloat, AnyInteger]> {}

def AffineReduceOp : PXA_Op<"reduce"> {
  let summary = "Reduce into memory";
  let arguments = (ins
    AggregationKind:$agg,
    AnyStdScalar:$val,
    AnyMemRef:$out,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );
  let results = (outs);
}

def ImplicitAffineTerminator
    : SingleBlockImplicitTerminator<"AffineTerminatorOp">;

def AffineParallelForOp : PXA_Op<"parallel_for", [ImplicitAffineTerminator]> {
  let summary = "multi-index parallel for operation";
  let arguments = (ins I64ArrayAttr:$ranges, Variadic<Index>:$dynamic_ranges);
  let regions = (region SizedRegion<1>:$inner);
}

#endif // __PML_PXA_OPS__
