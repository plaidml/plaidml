// Copyright 2019, Intel Corporation

#ifndef __PML_PXA_OPS__
#define __PML_PXA_OPS__

#ifndef AFFINE_OPS_BASE
include "mlir/Dialect/Affine/IR/AffineOpsBase.td"
#endif

#ifndef __PML_UTIL_ENUMS__
include "pmlc/util/enums.td"
#endif

include "mlir/Interfaces/SideEffects.td"

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

// TODO: Add NoSideEffect
def AffineReduceOp : PXA_OpWithPP<"reduce", []> {
  let summary = "affine reduction operation";
  let arguments = (ins
    AggregationKind:$agg,
    AnyStdScalar:$val,
    AnyMemRef:$mem,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );
  let results = (outs AnyMemRef:$result);

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, AggregationKind agg, Value val, Value mem, AffineMap map, ValueRange idxs", [{
      build(builder, result, mem.getType(), agg, val, mem, map, idxs);
    }]
  >];

  let extraClassDeclaration = [{
    Value getMemRef() { return mem(); }
    MemRefType getMemRefType() { return mem().getType().cast<MemRefType>(); }
    AffineMap getAffineMap() { return map(); }
    operand_range getMapOperands() { return idxs(); }
  }];
}

#endif // __PML_PXA_OPS__
