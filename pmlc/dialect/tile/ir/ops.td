// Copyright 2020 Intel Corporation

#ifndef __PML_TILE_BASE__
include "pmlc/dialect/tile/ir/base.td"
#endif

include "pmlc/dialect/tile/ir/builder.td"

def ConstantOp : TileOpWithPPV<"constant", [ConstantLike, NoSideEffect]>, HasFolder {
  let summary = "constant operation";
  let arguments = (ins IndexAttr:$value);
  let results = (outs Index:$result);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState& result, int64_t value", [{
      auto indexType = builder.getIndexType();
      result.addAttribute("value", builder.getIntegerAttr(indexType, value));
      result.addTypes(indexType);
    }]
  >];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
  }];
}

def ContractionOp : TileOpWithPPV<"contract", [NoSideEffect]> {
  let arguments = (ins
    EltwiseAny:$init,
    Variadic<AnyType>:$operands,
    AggregationKind:$agg,
    CombinationKind:$combo,
    AffineMapAttr:$sink,
    AffineMapArrayAttr:$srcs,
    OptionalAttr<IntegerSetAttr>:$cons,
    OptionalAttr<AffineMapAttr>:$shape,
    OptionalAttr<UnitAttr>:$no_reduce,
    OptionalAttr<AffineMapAttr>:$lowerBounds,
    OptionalAttr<AffineMapAttr>:$upperBounds,
    OptionalAttr<StrAttr>:$name
  );
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);

  let builders = [OpBuilder<
    "OpBuilder &builder, "
    "OperationState& result, "
    "Type resultType, "
    "Value init, "
    "ArrayRef<Value> tensors, "
    "AggregationKind agg, "
    "CombinationKind combo, "
    "AffineMap sink, "
    "ArrayRef<AffineMap> srcs, "
    "IntegerSet cons, "
    "bool no_reduce, "
    "StringRef name"
  >];

  let extraClassDeclaration = [{
    static unsigned getNumTensors(CombinationKind combo);
    static StringRef getSinkAttrName() { return "sink"; }
    static StringRef getSourcesAttrName() { return "srcs"; }
    static StringRef getConstraintsAttrName() { return "cons"; }
    static StringRef getLowerBoundsAttrName() { return "lowerBounds"; }
    static StringRef getUpperBoundsAttrName() { return "upperBounds"; }

    unsigned getNumTensors();
    unsigned getNumSymbols();
    Value getTensor(unsigned i);
    Value getSymbol(unsigned i);
    AffineMap getSourceMap(unsigned i);

    void setLowerBounds(ArrayRef<int64_t> bounds);
    void setUpperBounds(ArrayRef<int64_t> bounds);
    void setSink(AffineMap sink);
    void setSources(ArrayRef<AffineMap> srcs);
    void setConstraints(IntegerSet cons);
  }];
}

def IndexOp : TileOp<"index", [NoSideEffect]>, HasCanonicalizer {
  let summary = "tensor index lookup operation";
  let arguments = (ins APIntAttr:$axis, Variadic<EltwiseIndex>:$dims);
  let results = (outs EltwiseIndex:$result);

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value> operands);
  }];
}

class SpecialOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, !listconcat(traits, [
      SpecialOpInterface,
      GenericBuilderInterface
  ])> {

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value> operands);
  }];
}

def GatherOp : SpecialOp<"gather">, HasCanonicalizer {
  let summary = "special gather operation";
  let arguments = (ins
    RankedTensorOf<[AnyScalar]>:$tensor,
    RankedTensorOf<[AnyInteger]>:$dims);
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
}

def PrngOp : SpecialOp<"prng", []>, HasCanonicalizer {
  let summary = "pseudorandom number generator";
  let arguments = (ins
    RankedTensorOf<[UI32]>:$state,
    Variadic<EltwiseIndex>:$dims);
  let results = (outs
    RankedTensorOf<[AnyFloat]>:$result,
    RankedTensorOf<[UI32]>:$new_state);
}

def ReshapeOp : SpecialOp<"reshape">, HasCanonicalizer, HasFolder {
  let summary = "tensor reshape operation";
  let arguments = (ins
    RankedTensorOf<[AnyScalar]>:$tensor,
    Variadic<EltwiseIndex>:$dims);
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
}

def ScatterOp : SpecialOp<"scatter">, HasCanonicalizer {
  let summary = "special scatter operation";
  let arguments = (ins
    RankedTensorOf<[AnyScalar]>:$tensor,
    RankedTensorOf<[Index]>:$dims,
    RankedTensorOf<[AnyScalar]>:$other);
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
}

def ShapeOp : SpecialOp<"shape">, HasCanonicalizer {
  let summary = "tensor shape operation";
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$tensor);
  let results = (outs RankedTensorOf<[SI32]>:$result);
}

def TraceOp : Op<TileDialect, "trace", []>, HasCanonicalizer {
  let summary = "tracepoint operation";
  let arguments = (ins AnyType:$in, StrAttr:$msg);
  let results = (outs AnyType:$out);

  let builders = [OpBuilder<
    "OpBuilder &builder, OperationState& result, Value tensor, StringRef msg", [{
      result.addTypes(tensor.getType());
      result.addOperands(tensor);
      result.addAttribute("msg", builder.getStringAttr(msg));
    }]
  >];
}
