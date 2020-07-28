// Copyright 2019, Intel Corporation

#ifndef __PML_PXA_OPS__
#define __PML_PXA_OPS__

#ifndef AFFINE_OPS_BASE
include "mlir/Dialect/Affine/IR/AffineOpsBase.td"
#endif

#ifndef __PML_UTIL_ENUMS__
include "pmlc/util/enums.td"
#endif

include "mlir/Interfaces/SideEffectInterfaces.td"

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

def AffineReduceOp : PXA_OpWithPP<"reduce", []> {
  let summary = "affine reduction operation";
  let arguments = (ins
    AggregationKind:$agg,
    AnyStdScalar:$val,
    Arg<AnyMemRef, "the memref to reduce", [MemRead, MemWrite]>:$mem,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );
  let results = (outs AnyMemRef:$result);

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let builders = [OpBuilder<
    "OpBuilder& builder, OperationState& result, AggregationKind agg,"
    "Value val, Value mem, AffineMap map, ValueRange idxs", [{
      build(builder, result, mem.getType(), agg, val, mem, map, idxs);
    }]
  >];

  let extraClassDeclaration = [{
    Value getMemRef() { return mem(); }
    unsigned getMemRefOperandIndex() { return 1; }
    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }
    MemRefType getMemRefType() { return mem().getType().cast<MemRefType>(); }
    AffineMap getAffineMap() { return map(); }
    operand_range getMapOperands() { return idxs(); }
    static StringRef getMapAttrName() { return "map"; }
    Value getValueToStore() { return val(); }
  }];
}

def AffineVectorReduceOp : PXA_OpWithPP<"vector_reduce", []> {
  let summary = "affine vector reduction operation";
  let arguments = (ins
    AggregationKind:$agg,
    AnyVector:$vector,
    Arg<AnyMemRef, "the memref to reduce", [MemRead, MemWrite]>:$mem,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );
  let results = (outs AnyMemRef:$result);

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let builders = [OpBuilder<
    "OpBuilder& builder, OperationState& result, AggregationKind agg,"
    "Value vector, Value mem, AffineMap map, ValueRange idxs", [{
      build(builder, result, mem.getType(), agg, vector, mem, map, idxs);
    }]
  >];

  let extraClassDeclaration = [{
    Value getMemRef() { return mem(); }
    unsigned getMemRefOperandIndex() { return 1; }
    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }
    MemRefType getMemRefType() { return mem().getType().cast<MemRefType>(); }
    AffineMap getAffineMap() { return map(); }
    operand_range getMapOperands() { return idxs(); }
    static StringRef getMapAttrName() { return "map"; }
    Value getValueToStore() { return vector(); }
    mlir::VectorType getVectorType() {
      return vector().getType().cast<mlir::VectorType>();
    }
  }];
}

#endif // __PML_PXA_OPS__
