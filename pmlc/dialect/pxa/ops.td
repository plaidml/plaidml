// Copyright 2019, Intel Corporation

#ifndef __PML_PXA_OPS__
#define __PML_PXA_OPS__

include "mlir/Dialect/AffineOps/AffineOpsBase.td"
include "mlir/Transforms/LoopLikeInterface.td"

#ifndef __PML_UTIL_ENUMS__
include "pmlc/util/enums.td"
#endif

def PxaDialect : Dialect {
  let name = "pxa";
  let cppNamespace = "pmlc::dialect::pxa";
}

class PxaOp<string mnemonic, list<OpTrait> traits = []> :
    Op<PxaDialect, mnemonic, traits>;

def AnyScalar : AnyTypeOf<[AnyFloat, AnyInteger]> {}

def ReduceOp : PxaOp<"reduce"> {
  let summary = "Reduce into memory";
  let arguments = (ins AggregationKind:$agg, AnyScalar:$val, AnyMemRef:$out, AffineMapAttr:$map, Variadic<Index>:$idxs);
  let results = (outs);
}

def ImplicitAffineTerminator
    : SingleBlockImplicitTerminator<"AffineTerminatorOp">;

def AffineParallelForOp : PxaOp<"parallel_for",
    [ImplicitAffineTerminator,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "multi-index parallel for operation";
  let arguments = (ins I64ArrayAttr:$ranges, Variadic<Index>:$runtime_ranges);
  let regions = (region SizedRegion<1>:$inner);
}

def CastOp : PxaOp<"cast"> {
  let summary = "cast operation";
  let arguments = (ins AnyScalar:$input, BoolAttr:$in_signed, BoolAttr:$out_signed);
  let results = (outs AnyScalar:$result);
}

#endif // __PML_PXA_OPS__
