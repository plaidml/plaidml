// Copyright 2019, Intel Corporation

#ifndef AFFINE_OPS_BASE
include "mlir/Dialect/AffineOps/AffineOpsBase.td"
#endif

include "pmlc/util/enums.td"
include "pmlc/util/interfaces.td"
include "pmlc/dialect/eltwise/interfaces.td"
include "pmlc/dialect/eltwise/predicates.td"
include "pmlc/dialect/tile/interfaces.td"

// TODO: this should be defined upstream
def IntegerSetAttr : Attr<
    CPred<"$_self.isa<IntegerSetAttr>()">, "IntegerSet attribute"> {
  let storageType = [{ IntegerSetAttr }];
  let returnType = [{ IntegerSet }];
  let constBuilderCall = "IntegerSetAttr::get($0)";
}

def TileDialect : Dialect {
  let name = "tile";
  let cppNamespace = "pmlc::dialect::tile";
}

def AffineMap : Type<CPred<"$_self.isa<AffineMapType>()">, "affine-map">;
def AffineConstraints : Type<CPred<"$_self.isa<AffineConstraintsType>()">, "affine-constraints">;
def AffineTensorMap : Type<CPred<"$_self.isa<AffineTensorMapType>()">, "affine-tensor-map">;
def StrType : Type<CPred<"$_self.isa<StringType>()">, "string">;

class TileOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, traits> {}

class TileOpWithPPV<string mnemonic, list<OpTrait> traits = []> :
    TileOp<mnemonic, traits> {
  let printer = [{ print$cppClass(&p, *this); }];
  let parser = [{ return parse$cppClass(&parser, result); }];
  let verifier = [{ return verify$cppClass(*this); }];
}

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
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$tensor, APIntAttr:$dim);
  let results = (outs IndexLike:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, Value* tensor, int64_t value", [{
      result.addOperands(tensor);
      result.addAttribute("dim", builder->getI64IntegerAttr(value));
      result.addTypes(builder->getIndexType());
    }]
  >];

  let extraClassDeclaration = [{
    IntegerAttr resolve();
  }];
}

def AffineConstantOp : TileOpWithPPV<"affine_const", [NoSideEffect]>, HasFolder {
  let summary = "affine constant";
  let arguments = (ins I64Attr:$value);
  let results = (outs IndexLike:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, int64_t value", [{
      result.addAttribute("value", builder->getI64IntegerAttr(value));
      result.addTypes(builder->getIndexType());
    }]
  >];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
  }];
}

def AffineIndexOp : TileOpWithPPV<"idx"> {
  let summary = "affine index";
  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs IndexLike:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, StringRef name", [{
      result.addTypes(builder->getIndexType());
      if (name.size()) {
        result.addAttribute("name", builder->getStringAttr(name));
      }
    }]
  >];
}

class AffineOp<string mnemonic, list<OpTrait> traits = []> :
    TileOp<mnemonic, !listconcat(traits, [NoSideEffect])>,
    HasFolder {
  let results = (outs IndexLike:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addTypes(builder->getIndexType());
    }]
  >];
}

class AffineUnaryOp<string mnemonic, list<OpTrait> traits = []> : AffineOp<mnemonic, traits> {
  let arguments = (ins IndexLike:$in);
}

class AffineBinaryOp<string mnemonic, list<OpTrait> traits = []> : AffineOp<mnemonic, traits> {
  let arguments = (ins IndexLike:$lhs, IndexLike:$rhs);
  let printer = [{ printAffineBinaryOp(&p, *this); }];
  let parser = [{ return parseAffineBinaryOp(&parser, result); }];
}

def AffineAddOp : AffineBinaryOp<"affine_add", [Commutative]> {
  let summary = "Affine addition operation";
}

def AffineDivOp : AffineBinaryOp<"affine_div"> {
  let summary = "Affine division operation";
}

def AffineMulOp : AffineBinaryOp<"affine_mul", [Commutative]> {
  let summary = "Affine multiplication operation";
}

def AffineNegOp : AffineUnaryOp<"affine_neg"> {
  let summary = "Affine negative operation";
  let arguments = (ins IndexLike:$input);
}

def AffineSubOp : AffineBinaryOp<"affine_sub"> {
  let summary = "Affine subtraction operation";
}

def AffineMaxOp : AffineBinaryOp<"affine_max", [Commutative]> {
  let summary = "Affine max operation";
}

def AffineMinOp : AffineBinaryOp<"affine_min", [Commutative]> {
  let summary = "Affine min operation";
}

def AffineTensorMapOp : TileOpWithPPV<"tmap", [NoSideEffect]> {
  let summary = "affine tensor map operation";
  let arguments = (ins EltwiseAny:$tensor, Variadic<Index>:$dims);
  let results = (outs AffineTensorMap:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, Value* tensor, ArrayRef<Value*> dims", [{
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
    "Builder* builder, OperationState& result, ArrayRef<Value*> dims", [{
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

def SymbolicContractionOp : TileOpWithPPV<"sym_cion", [NoSideEffect]>, HasCanonicalizer {
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

def ContractionOp : TileOpWithPPV<"cion", [NoSideEffect]> {
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
    OptionalAttr<StrAttr>:$name
  );
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);

  let builders = [OpBuilder<
    "Builder* builder, "
    "OperationState& result, "
    "Type resultType, "
    "Value* init, "
    "ArrayRef<Value*> tensors, "
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
    unsigned getNumTensors();
    unsigned getNumSymbols();
    Value* getTensor(unsigned i);
    Value* getSymbol(unsigned i);
  }];
}

def IndexOp : TileOp<"index", [NoSideEffect]>, HasCanonicalizer {
  let summary = "tensor index lookup operation";
  let arguments = (ins EltwiseAny:$tensor, APIntAttr:$dim);
  let results = (outs EltwiseIndex:$result);

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands);
  }];
}

class SpecialOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, !listconcat(traits, [SpecialOpInterface, GenericBuilderInterface])> {

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands);

    static Operation* create(OpBuilder* builder, Location loc, Type type, ArrayRef<Value*> operands) {
      OperationState state(loc, getOperationName());
      state.addOperands(operands);
      state.addAttribute("type", TypeAttr::get(type));
      state.addTypes(getResultType(operands));
      return builder->createOperation(state);
    }
  }];
}

def GatherOp : SpecialOp<"gather">, HasCanonicalizer {
  let summary = "special gather operation";
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$tensor, VectorOf<[ScalarIndex]>:$dims);
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
}

def PrngOp : SpecialOp<"prng", []>, HasCanonicalizer {
  let summary = "pseudorandom number generator";
  let arguments = (ins RankedTensorOf<[ScalarUINT32]>:$state, Variadic<EltwiseIndex>:$dims);
  let results = (outs RankedTensorOf<[AnyScalarFloat]>:$result, RankedTensorOf<[ScalarUINT32]>:$new_state);
}

def ReshapeOp : SpecialOp<"reshape">, HasCanonicalizer {
  let summary = "tensor reshape operation";
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$tensor, Variadic<EltwiseIndex>:$dims);
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
}

def ScatterOp : SpecialOp<"scatter">, HasCanonicalizer {
  let summary = "special scatter operation";
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$tensor, VectorOf<[ScalarIndex]>:$dims, RankedTensorOf<[AnyScalar]>:$other);
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
}

def ShapeOp : SpecialOp<"shape">, HasCanonicalizer {
  let summary = "tensor shape operation";
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$tensor);
  let results = (outs RankedTensorOf<[ScalarIndex]>:$result);
}
