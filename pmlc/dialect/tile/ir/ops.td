// Copyright 2020 Intel Corporation

#ifndef __PML_TILE_BASE__
include "pmlc/dialect/tile/ir/base.td"
#endif

#ifndef __PMLC_UTIL_INTERFACES__
include "pmlc/util/interfaces.td"
#endif

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

def ContractionOp : TileOpWithPPV<"contract", [
  NoSideEffect,
  MaterializeOperandsOpInterface
]> {
  let arguments = (ins
    EltwiseAny:$init,
    Variadic<AnyType>:$operands,
    AggregationKind:$agg,
    CombinationKind:$combo,
    AffineMapAttr:$sink,
    AffineMapArrayAttr:$srcs,
    OptionalAttr<IntegerSetAttr>:$cons,
    OptionalAttr<AffineMapAttr>:$shape,
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

    RankedTensorType getResultType() { return result().getType().cast<RankedTensorType>(); }

    LogicalResult materializeOperands(OpBuilder& builder);
  }];
}

class SpecialOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, traits>;

class MaterializeOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
  Op<TileDialect, mnemonic, !listconcat(traits, [
    MaterializeOperandsOpInterface
  ])> {
  let extraClassDeclaration = [{
    LogicalResult materializeOperands(OpBuilder& builder);
  }];
}

def GatherOp : MaterializeOp<"gather"> {
  let summary = "special gather operation";
  let arguments = (ins
    RankedTensorOf<[AnyScalar]>:$tensor,
    RankedTensorOf<[AnyInteger]>:$dims);
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
}

def IndexOp : SpecialOp<"index"> {
  let summary = "tensor index lookup operation";
  let arguments = (ins APIntAttr:$axis, Variadic<EltwiseIndex>:$dims);
  let results = (outs EltwiseIndex:$result);
}

def PrngOp : SpecialOp<"prng"> {
  let summary = "pseudorandom number generator";
  let arguments = (ins RankedTensorOf<[UI32]>:$state);
  let results = (outs
    RankedTensorOf<[AnyFloat]>:$result,
    RankedTensorOf<[UI32]>:$new_state);
}

def ReshapeOp : MaterializeOp<"reshape">,
  HasFolder {
  let summary = "tensor reshape operation";
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$tensor);
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
  let assemblyFormat = [{
    $tensor attr-dict `:` functional-type(operands, results)
  }];
}

def ScatterOp : MaterializeOp<"scatter"> {
  let summary = "special scatter operation";
  let arguments = (ins
    RankedTensorOf<[AnyScalar]>:$tensor,
    RankedTensorOf<[AnyInteger]>:$dims);
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
}

def ShapeOp : MaterializeOp<"shape"> {
  let summary = "tensor shape operation";
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$tensor);
  let results = (outs RankedTensorOf<[SI32]>:$result);
}

def TraceOp : MaterializeOp<"trace", []> {
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
