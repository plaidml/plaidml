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
    "mlir::OpBuilder &builder, mlir::OperationState& result, int64_t value", [{
      auto indexType = builder.getIndexType();
      result.addAttribute("value", builder.getIntegerAttr(indexType, value));
      result.addTypes(indexType);
    }]
  >];

  let extraClassDeclaration = [{
    mlir::Attribute getValue() { return getAttr("value"); }
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
    "mlir::OpBuilder &builder, "
    "mlir::OperationState& result, "
    "mlir::Type resultType, "
    "mlir::Value init, "
    "mlir::ArrayRef<mlir::Value> tensors, "
    "util::AggregationKind agg, "
    "util::CombinationKind combo, "
    "mlir::AffineMap sink, "
    "mlir::ArrayRef<mlir::AffineMap> srcs, "
    "mlir::IntegerSet cons, "
    "mlir::StringRef name"
  >];

  let extraClassDeclaration = [{
    static unsigned getNumTensors(CombinationKind combo);
    static mlir::StringRef getSinkAttrName() { return "sink"; }
    static mlir::StringRef getSourcesAttrName() { return "srcs"; }
    static mlir::StringRef getConstraintsAttrName() { return "cons"; }
    static mlir::StringRef getLowerBoundsAttrName() { return "lowerBounds"; }
    static mlir::StringRef getUpperBoundsAttrName() { return "upperBounds"; }

    unsigned getNumTensors();
    unsigned getNumSymbols();
    mlir::Value getTensor(unsigned i);
    mlir::Value getSymbol(unsigned i);
    mlir::AffineMap getSourceMap(unsigned i);

    void setLowerBounds(mlir::ArrayRef<int64_t> bounds);
    void setUpperBounds(mlir::ArrayRef<int64_t> bounds);
    void setSink(mlir::AffineMap sink);
    void setSources(mlir::ArrayRef<mlir::AffineMap> srcs);
    void setConstraints(mlir::IntegerSet cons);

    mlir::RankedTensorType getResultType() { return result().getType().cast<mlir::RankedTensorType>(); }

    mlir::LogicalResult materializeOperands(mlir::OpBuilder& builder);
  }];
}

class SpecialOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, traits>;

class MaterializeOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
  Op<TileDialect, mnemonic, !listconcat(traits, [
    MaterializeOperandsOpInterface
  ])> {
  let extraClassDeclaration = [{
    mlir::LogicalResult materializeOperands(mlir::OpBuilder& builder);
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

def PragmaOp : MaterializeOp<"pragma", [
      AllTypesMatch<["tensor", "result"]>
    ]> {
  let summary = "pragma operation";
  let arguments = (ins AnyType:$tensor, StrAttr:$op, DictionaryAttr:$attrs);
  let results = (outs AnyType:$result);

  let builders = [OpBuilder<
    "mlir::OpBuilder &builder, mlir::OperationState& result, mlir::Value tensor, mlir::StringRef op,"
      "mlir::DictionaryAttr attrs", [{
      result.addTypes(tensor.getType());
      result.addOperands(tensor);
      result.addAttribute("op", builder.getStringAttr(op));
      result.addAttribute("attrs", attrs);
    }]
  >];

  let assemblyFormat = [{
    $tensor $op $attrs attr-dict `:` type($tensor)
  }];
}
