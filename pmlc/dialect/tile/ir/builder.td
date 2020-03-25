// Copyright 2020 Intel Corporation

#ifndef __PML_TILE_BASE__
include "pmlc/dialect/tile/ir/base.td"
#endif

def PlaceholderOp : TileOp<"var"> {
  let summary = "placeholder operation";
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
}

def TupleOp : TileOp<"tuple"> {
  let arguments = (ins Variadic<AnyType>:$elts);
  let results = (outs AnyTuple:$result);
}

def NoneOp : TileOp<"none"> {
  let results = (outs NoneType:$result);
}

def StringOp : TileOp<"str"> {
  let arguments = (ins StrAttr:$value);
  let results = (outs StrType:$result);
  let extraClassDeclaration = [{
    StringAttr getValue() { return getAttrOfType<StringAttr>("value"); }
  }];
}

def DimOp : TileOpWithPPV<"dim", [NoSideEffect]>, HasFolder {
  let summary = "dimension index operation";
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$tensor, IndexAttr:$dim);
  let results = (outs Index:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, Value tensor, int64_t value", [{
      auto indexType = builder->getIndexType();
      result.addOperands(tensor);
      result.addAttribute("dim", builder->getIntegerAttr(indexType, value));
      result.addTypes(indexType);
    }]
  >];

  let extraClassDeclaration = [{
    IntegerAttr resolve();
  }];
}

def PolyIndexOp : TileOpWithPPV<"idx", [NoSideEffect]> {
  let summary = "polynomial index operation";
  let arguments = (ins I64Attr:$id, OptionalAttr<StrAttr>:$name);
  let results = (outs Index:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, int64_t id, StringRef name", [{
      result.addTypes(builder->getIndexType());
      result.addAttribute("id", builder->getI64IntegerAttr(id));
      if (name.size()) {
        result.addAttribute("name", builder->getStringAttr(name));
      }
    }]
  >];
}

class PolyOp<string mnemonic, list<OpTrait> traits = []> :
    TileOp<mnemonic, !listconcat(traits, [NoSideEffect])>,
    HasFolder {
  let results = (outs Index:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ArrayRef<Value> operands", [{
      result.addOperands(operands);
      result.addTypes(builder->getIndexType());
    }]
  >];
}

class PolyUnaryOp<string mnemonic, list<OpTrait> traits = []> :
    PolyOp<mnemonic, traits> {
  let arguments = (ins Index:$in);
}

class PolyBinaryOp<string mnemonic, list<OpTrait> traits = []> :
    PolyOp<mnemonic, traits> {
  let arguments = (ins Index:$lhs, Index:$rhs);
  let printer = [{ printPolyBinaryOp(&p, *this); }];
  let parser = [{ return parsePolyBinaryOp(&parser, result); }];
}

def PolyAddOp : PolyBinaryOp<"poly_add", [Commutative]> {
  let summary = "Polynomial addition operation";
}

def PolyDivOp : PolyBinaryOp<"poly_div"> {
  let summary = "Polynomial division operation";
}

def PolyMulOp : PolyBinaryOp<"poly_mul", [Commutative]> {
  let summary = "Polynomial multiplication operation";
}

def PolyNegOp : PolyUnaryOp<"poly_neg"> {
  let summary = "Polynomial negative operation";
  let arguments = (ins Index:$input);
}

def PolySubOp : PolyBinaryOp<"poly_sub"> {
  let summary = "Polynomial subtraction operation";
}

def PolyMaxOp : PolyBinaryOp<"poly_max", [Commutative]> {
  let summary = "Polynomial max operation";
}

def PolyMinOp : PolyBinaryOp<"poly_min", [Commutative]> {
  let summary = "Polynomial min operation";
}

def AffineTensorMapOp : TileOpWithPPV<"tmap", [NoSideEffect]> {
  let summary = "affine tensor map operation";
  let arguments = (ins EltwiseAny:$tensor, Variadic<Index>:$dims);
  let results = (outs AffineTensorMap:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, Value tensor, ArrayRef<Value> dims", [{
      result.addOperands(tensor);
      result.addOperands(dims);
      result.addTypes(builder->getType<AffineTensorMapType>());
    }]
  >];
}

def AffineMapOp : TileOpWithPPV<"map", [NoSideEffect]> {
  let summary = "affine map operation";
  let arguments = (ins Variadic<Index>:$dims);
  let results = (outs AffineMap:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ArrayRef<Value> dims", [{
      result.addOperands(dims);
      result.addTypes(builder->getType<AffineMapType>());
    }]
  >];
}

def AffineConstraintsOp : TileOpWithPPV<"cons", [NoSideEffect]> {
  let summary = "affine constraint operation";
  let arguments = (ins Variadic<Index>:$pairs);
  let results = (outs AffineConstraints:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result", [{
      result.setOperandListToResizable();
      result.addTypes(builder->getType<AffineConstraintsType>());
    }]
  >];
}

def SymbolicContractionOp : TileOpWithPPV<"sym_contract", [NoSideEffect]>,
    HasCanonicalizer {
  let arguments = (ins
    EltwiseAny:$init,
    AffineConstraints:$cons,
    AffineMap:$size,
    AffineMap:$sink,
    Variadic<AffineTensorMap>:$srcs,
    AggregationKind:$agg,
    CombinationKind:$combo,
    OptionalAttr<UnitAttr>:$no_reduce,
    OptionalAttr<StrAttr>:$name
  );
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
}
