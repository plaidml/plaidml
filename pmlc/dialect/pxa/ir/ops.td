// Copyright 2019, Intel Corporation

#ifndef __PML_PXA_OPS__
#define __PML_PXA_OPS__

include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "mlir/Dialect/StandardOps/IR/StandardOpsBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "pmlc/util/enums.td"
include "interfaces.td"

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

class PxaReduceOpBase<string mnemonic, list<OpTrait> traits = []> :
    PXA_OpWithPP<mnemonic, !listconcat(traits, [
        PxaGenericOpInterface,
        DeclareOpInterfaceMethods<PxaReduceOpInterface>,
        DeclareOpInterfaceMethods<AffineMapAccessInterface>,
        MemRefsNormalizable])> {

  code extraClassDeclarationBase = [{
    mlir::ValueRange inputs() { return {}; }
    mlir::ValueRange outputs() { return {memref()}; }

    mlir::OpOperandVector getInputOperands() { return {}; }
    mlir::OpOperandVector getOutputOperands() {
      mlir::OpOperandVector result;
      result.push_back(&getOperation()->getOpOperand(1));
      return result;
    }

    mlir::SmallVector<int64_t> getTiedVectorShape(mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
      assert(opOperand->getOperandNumber() == 1);
      mlir::SmallVector<int64_t> result(getMemRefType().getRank(), 1);
      if (mlir::VectorType vecType = getVectorType()) {
        mlir::ArrayRef<int64_t> shape = vecType.getShape();
        for (size_t i = 0; i < shape.size(); i++)
          result[i + shape.size() - result.size()] = shape[i];
      }
      return result;
    }

    mlir::AffineValueMap getTiedAffineValueMap(mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
      assert(opOperand->getOperandNumber() == 1);
      return mlir::AffineValueMap(map(), idxs());
    }

    mlir::Value getMemRef() { return memref(); }
    mlir::MemRefType getMemRefType() {
      return memref().getType().cast<mlir::MemRefType>();
    }

    mlir::VectorType getVectorType() {
      return val().getType().dyn_cast<mlir::VectorType>();
    }

    mlir::AffineMap getAffineMap() { return map(); }
    operand_range getMapOperands() { return idxs(); }
    static mlir::StringRef getMapAttrName() { return "map"; }
    mlir::AffineMapAttr getAffineMapAttr() { return mapAttr(); }
    mlir::AtomicRMWKind getAgg() { return agg(); }
  }];
}

def PxaReduceOp : PxaReduceOpBase<"reduce"> {
  let summary = "affine reduction operation";
  let arguments = (ins
    AtomicRMWKindAttr:$agg,
    AnyType:$val,
    AnyMemRef:$memref,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );
  let results = (outs AnyMemRef:$result);

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins
      "mlir::AtomicRMWKind":$agg,
      "mlir::Value":$val,
      "mlir::Value":$memref,
      "mlir::AffineMap":$map,
      "mlir::ValueRange":$idxs
    ), [{
      build($_builder, $_state, memref.getType(), agg, val, memref, map, idxs);
    }]>
  ];

  let extraClassDeclaration = extraClassDeclarationBase # [{
    mlir::Value getValueToStore() { return val(); }
  }];
}

def PxaVectorReduceOp : PxaReduceOpBase<"vector_reduce"> {
  let summary = "affine vector reduction operation";
  let arguments = (ins
    AtomicRMWKindAttr:$agg,
    AnyVector:$val,
    AnyMemRef:$memref,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );
  let results = (outs AnyMemRef:$result);

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins
      "mlir::AtomicRMWKind":$agg,
      "mlir::Value":$val,
      "mlir::Value":$memref,
      "mlir::AffineMap":$map,
      "mlir::ValueRange":$idxs
    ), [{
      build($_builder, $_state, memref.getType(), agg, val, memref, map, idxs);
    }]>
  ];

  let extraClassDeclaration = extraClassDeclarationBase # [{
    mlir::Value getValueToStore() { return val(); }
  }];
}

class PxaLoadOpBase<string mnemonic, list<OpTrait> traits = []> :
    PXA_OpWithPP<mnemonic, !listconcat(traits, [
        PxaGenericOpInterface,
        DeclareOpInterfaceMethods<PxaReadOpInterface>,
        DeclareOpInterfaceMethods<AffineMapAccessInterface>,
        MemRefsNormalizable])> {
  let arguments = (ins
    AnyMemRef:$memref,
    AffineMapAttr:$map,
    Variadic<Index>:$idxs
  );

  code extraClassDeclarationBase = [{
    mlir::ValueRange inputs() { return {memref()}; }
    mlir::ValueRange outputs() { return {}; }

    mlir::OpOperandVector getInputOperands() {
      mlir::OpOperandVector result;
      result.push_back(&getOperation()->getOpOperand(0));
      return result;
    }
    mlir::OpOperandVector getOutputOperands() { return {}; }

    mlir::SmallVector<int64_t> getTiedVectorShape(mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
      assert(opOperand->getOperandNumber() == 0);
      mlir::SmallVector<int64_t> result(getMemRefType().getRank(), 1);
      if (mlir::VectorType vecType = getVectorType()) {
        mlir::ArrayRef<int64_t> shape = vecType.getShape();
        for (size_t i = 0; i < shape.size(); i++)
          result[i + shape.size() - result.size()] = shape[i];
      }
      return result;
    }

    mlir::AffineValueMap getTiedAffineValueMap(mlir::OpOperand *opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
      assert(opOperand->getOperandNumber() == 0);
      return mlir::AffineValueMap(map(), idxs());
    }

    mlir::Value getMemRef() { return memref(); }
    mlir::MemRefType getMemRefType() {
      return memref().getType().cast<mlir::MemRefType>();
    }

    mlir::VectorType getVectorType() {
      return result().getType().dyn_cast<mlir::VectorType>();
    }

    /// Returns the affine map used to index the memref for this operation.
    mlir::AffineMapAttr getAffineMapAttr() { return mapAttr(); }

    static mlir::StringRef getMapAttrName() { return "map"; }
  }];
}

def PxaLoadOp : PxaLoadOpBase<"load", [NoSideEffect]> {
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
    OpBuilder<(ins
      "mlir::AffineMap":$map,
      "mlir::ValueRange":$operands
    )>,
    /// Builds an affine load op with an identity map and operands.
    OpBuilder<(ins
      "mlir::Value":$memref,
      CArg<"mlir::ValueRange", "{}">:$indices
    )>,
    /// Builds an affine load op with the specified map and its operands.
    OpBuilder<(ins
      "mlir::Value":$memref,
      "mlir::AffineMap":$map,
      "mlir::ValueRange":$mapOperands
    )>
  ];

  let extraClassDeclaration = extraClassDeclarationBase;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def PxaVectorLoadOp : PxaLoadOpBase<"vector_load"> {
  let summary = "affine vector load operation";
  let description = [{
    The "affine.vector_load" is the vector counterpart of
    [affine.load](#affineload-operation). It reads a slice from a
    [MemRef](../LangRef.md#memref-type), supplied as its first operand,
    into a [vector](../LangRef.md#vector-type) of the same base elemental type.
    The index for each memref dimension is an affine expression of loop induction
    variables and symbols. These indices determine the start position of the read
    within the memref. The shape of the return vector type determines the shape of
    the slice read from the memref. This slice is contiguous along the respective
    dimensions of the shape. Strided vector loads will be supported in the future.
    An affine expression of loop IVs and symbols must be specified for each
    dimension of the memref. The keyword 'symbol' can be used to indicate SSA
    identifiers which are symbolic.

    Example 1: 8-wide f32 vector load.

    ```mlir
    %1 = affine.vector_load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>, vector<8xf32>
    ```

    Example 2: 4-wide f32 vector load. Uses 'symbol' keyword for symbols '%n' and '%m'.

    ```mlir
    %1 = affine.vector_load %0[%i0 + symbol(%n), %i1 + symbol(%m)] : memref<100x100xf32>, vector<4xf32>
    ```

    Example 3: 2-dim f32 vector load.

    ```mlir
    %1 = affine.vector_load %0[%i0, %i1] : memref<100x100xf32>, vector<2x8xf32>
    ```

    TODOs:
    * Add support for strided vector loads.
    * Consider adding a permutation map to permute the slice that is read from memory
    (see [vector.transfer_read](../Vector/#vectortransfer_read-vectortransferreadop)).
  }];

  let results = (outs AnyVector:$result);

  let builders = [
    OpBuilder<(ins
      "mlir::VectorType":$type,
      "mlir::Value":$memref,
      "mlir::AffineMap":$map,
      "mlir::ValueRange":$mapOperands
    )>
  ];

  let extraClassDeclaration = extraClassDeclarationBase;

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def PxaStoreOp : PXA_OpWithPP<"store", [MemRefsNormalizable]> {
  let summary = "PXA non-affine store operation";

  let arguments = (ins
      AtomicRMWKindAttr:$agg,
      AnyTypeOf<[AnySignlessInteger, AnyFloat, AnyVector]>:$value,
      MemRefOf<[AnySignlessInteger, AnyFloat]>:$memref,
      Variadic<Index>:$indices);
  let results = (outs MemRefOf<[AnySignlessInteger, AnyFloat]>:$result);

  let assemblyFormat = [{
    $agg $value `,` $memref `[` $indices `]` attr-dict `:` `(` type($value) `,`
    type($memref) `)` `->` type($result)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins
      "mlir::AtomicRMWKind":$agg,
      "mlir::Value":$val,
      "mlir::Value":$mem,
      "mlir::ValueRange":$idxs
    ), [{
      build($_builder, $_state, mem.getType(), agg, val, mem, idxs);
    }]>
  ];
}

def PxaGenericOp : PXA_OpWithPP<"generic", [
    PxaGenericOpInterface,
    AttrSizedOperandSegments]> {
  let summary = "PXA generic operation";
  let arguments = (ins
    Variadic<AnyMemRef>:$inputs,
    Variadic<AnyMemRef>:$outputs,
    Variadic<Index>:$inputIndices,
    Variadic<Index>:$outputIndices,
    AffineMapArrayAttr:$inputAccessMaps,
    AffineMapArrayAttr:$inputTileMaps,
    AffineMapArrayAttr:$outputAccessMaps,
    AffineMapArrayAttr:$outputTileMaps,
    StrAttr:$kernel,
    I64ArrayAttr:$tile,
    OptionalAttr<I64ArrayAttr>:$batches,
    I64ArrayAttr:$reductions
  );
  let results = (outs Variadic<AnyMemRef>:$results);

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    void getAffineValueMaps(
      mlir::ArrayAttr accessMaps,
      mlir::OperandRange mapOperands,
      mlir::SmallVectorImpl<mlir::AffineValueMap>& result);

    mlir::OpOperandVector getInputOperands();
    mlir::OpOperandVector getOutputOperands();

    mlir::SmallVector<int64_t> getTiedVectorShape(mlir::OpOperand *opOperand);

    mlir::SmallVector<mlir::AffineValueMap> getAffineValueMaps();
    mlir::AffineValueMap getTiedAffineValueMap(mlir::OpOperand *opOperand);

    static mlir::StringRef getInputAccessMapsAttrName() {
      return "inputAccessMaps";
    }
    static mlir::StringRef getInputTileMapsAttrName() {
      return "inputTileMaps";
    }
    static mlir::StringRef getOutputAccessMapsAttrName() {
      return "outputAccessMaps";
    }
    static mlir::StringRef getOutputTileMapsAttrName() {
      return "outputTileMaps";
    }
    static mlir::StringRef getKernelAttrName() { return "kernel"; }
    static mlir::StringRef getTileAttrName() { return "tile"; }
    static mlir::StringRef getBatchesAttrName() { return "batches"; }
    static mlir::StringRef getReductionsAttrName() { return "reductions"; }
  }];
}

// TODO: replace PrngOp with an PxaGenericOp
def PrngOp : PXA_Op<"prng"> {
  let summary = "pseudorandom number generator operation";
  let arguments = (ins AnyMemRef:$state, AnyMemRef:$tensor, AnyMemRef:$new_state);
  let results = (outs AnyMemRef:$result_tensor, AnyMemRef:$result_state);
}

#endif // __PML_PXA_OPS__
