// Copyright 2019, Intel Corporation

#ifndef __PML_PXA_OPS__
#define __PML_PXA_OPS__

include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
include "mlir/Dialect/Affine/IR/AffineOpsBase.td"
include "pmlc/util/enums.td"

include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
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

def AffineReduceOp : PXA_OpWithPP<"reduce"> {
  let summary = "affine reduction operation";
  let arguments = (ins
    AtomicRMWKindAttr:$agg,
    AnyStdScalar:$val,
    Arg<AnyMemRef, "the memref to reduce", [MemRead, MemWrite]>:$mem,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );
  let results = (outs AnyMemRef:$result);

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let builders = [OpBuilder<
    "OpBuilder& builder, OperationState& result, AtomicRMWKind agg,"
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

def AffineGemmOp : PXA_OpWithPP<"gemm"> {
  let arguments = (ins
    Arg<AnyMemRef, "the C memref to store to", [MemWrite]>:$c,
    AffineMapAttr:$cAccessMap,
    AffineMapAttr:$cTileMap,
    Arg<AnyMemRef, "the A memref to load from", [MemRead]>:$a,
    AffineMapAttr:$aAccessMap,
    AffineMapAttr:$aTileMap,
    Arg<AnyMemRef, "the B memref to load from", [MemRead]>:$b,
    AffineMapAttr:$bAccessMap,
    AffineMapAttr:$bTileMap,
    I64ArrayAttr:$tile,
    Variadic<Index>:$mapOperands);
  let results = (outs AnyMemRef:$out);

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    operand_range getOperandsForA();
    operand_range getOperandsForB();
    operand_range getOperandsForC();
  }];
}

def AffineVectorReduceOp : PXA_OpWithPP<"vector_reduce", []> {
  let summary = "affine vector reduction operation";
  let arguments = (ins
    AtomicRMWKindAttr:$agg,
    AnyVector:$vector,
    Arg<AnyMemRef, "the memref to reduce", [MemRead, MemWrite]>:$mem,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );
  let results = (outs AnyMemRef:$result);

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let builders = [OpBuilder<
    "OpBuilder& builder, OperationState& result, AtomicRMWKind agg,"
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

class AffineLoadOpBase<string mnemonic, list<OpTrait> traits = []> :
    PXA_OpWithPP<mnemonic, !listconcat(traits,
        [DeclareOpInterfaceMethods<AffineReadOpInterface>])> {
  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
      [MemRead]>:$memref,
      Variadic<Index>:$indices);

  code extraClassDeclarationBase = [{
    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 0; }

    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

    /// Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
      return getAttr(getMapAttrName()).cast<AffineMapAttr>();
    }

    static StringRef getMapAttrName() { return "map"; }
  }];
}

def AffineLoadOp : AffineLoadOpBase<"load"> {
  let summary = "affine load operation";
  let description = [{
    The "affine.load" op reads an element from a memref, where the index
    for each memref dimension is an affine expression of loop induction
    variables and symbols. The output of 'affine.load' is a new value with the
    same type as the elements of the memref. An affine expression of loop IVs
    and symbols must be specified for each dimension of the memref. The keyword
    'symbol' can be used to indicate SSA identifiers which are symbolic.

    Example 1:

    ```mlir
    %1 = affine.load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>
    ```

    Example 2: Uses 'symbol' keyword for symbols '%n' and '%m'.

    ```mlir
    %1 = affine.load %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>
    ```
  }];

  let results = (outs AnyType:$result);

  let builders = [
    /// Builds an affine load op with the specified map and operands.
    OpBuilder<"OpBuilder &builder, OperationState &result, AffineMap map, "
                      "ValueRange operands">,
    /// Builds an affine load op with an identity map and operands.
    OpBuilder<"OpBuilder &builder, OperationState &result, Value memref, "
                      "ValueRange indices = {}">,
    /// Builds an affine load op with the specified map and its operands.
    OpBuilder<"OpBuilder &builder, OperationState &result, Value memref, "
                      "AffineMap map, ValueRange mapOperands">
  ];

  let extraClassDeclaration = extraClassDeclarationBase;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

#endif // __PML_PXA_OPS__
