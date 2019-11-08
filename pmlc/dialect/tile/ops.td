// Copyright 2019, Intel Corporation

include "pmlc/util/enums.td"
include "pmlc/util/interfaces.td"
include "pmlc/dialect/eltwise/interfaces.td"
include "pmlc/dialect/eltwise/predicates.td"
include "pmlc/dialect/tile/interfaces.td"

def TileDialect : Dialect {
  let name = "tile";
  let cppNamespace = "pmlc::dialect::tile";
}

def AffineIndexMap : Type<CPred<"$_self.isa<AffineIndexMapType>()">, "affine-index-map">;
def AffineSizeMap : Type<CPred<"$_self.isa<AffineSizeMapType>()">, "affine-size-map">;
def StrType : Type<CPred<"$_self.isa<StringType>()">, "string">;

class TileOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, traits> {}

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

def DimOp : TileOp<"dim">, HasFolder {
  let summary = "dimension index operation";
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$tensor, APIntAttr:$dim);
  let results = (outs ScalarIndex:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, Value* tensor, int64_t value", [{
      auto type = ScalarType::get(builder->getContext(), DataType::INTX);
      result.addOperands(tensor);
      result.addAttribute("dim", builder->getI64IntegerAttr(value));
      result.addTypes(type);
    }]
  >];
}

def AffineConstantOp : TileOp<"affine_const">, HasFolder {
  let summary = "affine constant";
  let arguments = (ins I64Attr:$value);
  let results = (outs ScalarIndex:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, int64_t value", [{
      result.addAttribute("value", builder->getI64IntegerAttr(value));
      result.addTypes(ScalarType::get(builder->getContext(), DataType::INTX));
    }]
  >];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
  }];
}

def AffineIndexOp : TileOp<"idx"> {
  let summary = "affine index";
  let results = (outs ScalarIndex:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result", [{
      result.addTypes(ScalarType::get(builder->getContext(), DataType::INTX));
    }]
  >];
}

class AffineOp<string mnemonic, list<OpTrait> traits = []> :
    TileOp<mnemonic, !listconcat(traits, [NoSideEffect])>,
    HasFolder {
  let results = (outs ScalarIndex:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addTypes(ScalarType::get(builder->getContext(), DataType::INTX));
    }]
  >];
}

class AffineUnaryOp<string mnemonic, list<OpTrait> traits = []> : AffineOp<mnemonic, traits> {
  let arguments = (ins ScalarIndex:$in);
}

class AffineBinaryOp<string mnemonic, list<OpTrait> traits = []> : AffineOp<mnemonic, traits> {
  let arguments = (ins ScalarIndex:$lhs, ScalarIndex:$rhs);
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
  let arguments = (ins ScalarIndex:$input);
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

def AffineDomainOp : TileOp<"domain">, HasCanonicalizer {
  let summary = "affine domain";
  let arguments = (ins OptionalAttr<BoolAttr>:$no_reduce);
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
  let regions = (region SizedRegion<1>:$body);
}

def AffineSourceIndexMapOp : TileOp<"src_idx_map", [HasParent<"AffineDomainOp">]> {
  let summary = "affine source index map";
  let arguments = (ins EltwiseAny:$tensor, Variadic<ScalarIndex>:$dims);
  let results = (outs AffineIndexMap:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, Value* tensor, ArrayRef<Value*> dims", [{
      result.addOperands(tensor);
      result.addOperands(dims);
      result.addTypes(builder->getType<AffineIndexMapType>());
    }]
  >];
}

def AffineSinkIndexMapOp : TileOp<"sink_idx_map", [HasParent<"AffineDomainOp">]> {
  let summary = "affine sink index map";
  let arguments = (ins Variadic<ScalarIndex>:$dims);
  let results = (outs AffineIndexMap:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ArrayRef<Value*> dims", [{
      result.addOperands(dims);
      result.addTypes(builder->getType<AffineIndexMapType>());
    }]
  >];
}

def AffineSizeMapOp : TileOp<"size_map", [HasParent<"AffineDomainOp">]> {
  let summary = "affine size map";
  let arguments = (ins Variadic<ScalarIndex>:$sizes);
  let results = (outs AffineSizeMap:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ArrayRef<Value*> sizes", [{
      result.addOperands(sizes);
      result.addTypes(builder->getType<AffineSizeMapType>());
    }]
  >];
}

def ConstraintOp : TileOp<"constraint", [Terminator]> {
  let summary = "constraint operation";
  let arguments = (ins ScalarIndex:$lhs, ScalarIndex:$rhs);
  let regions = (region SizedRegion<1>:$body);
}

class ContractionOp<string mnemonic, list<OpTrait> traits = []> :
    TileOp<mnemonic, !listconcat(traits, [Terminator, ContractionOpInterface])>;

class UnaryConOp<string mnemonic, string agg, string combo> : ContractionOp<mnemonic> {
  let arguments = (ins
    AffineSizeMap:$size,
    AffineIndexMap:$src,
    AffineIndexMap:$sink,
    Variadic<RankedTensorOf<[AnyScalar]>>:$init
  );

  let extraClassDeclaration = [{
    static Operation* create(OpBuilder* builder, Location loc, Value* size, ArrayRef<Value*> srcs, Value* sink) {
      OperationState state(loc, getOperationName());
      state.addOperands(size);
      state.addOperands(srcs);
      state.addOperands(sink);
      state.setOperandListToResizable();
      return builder->createOperation(state);
    }
    Value* getSizeMap() { return size(); }
    std::vector<Value*> getSourceIndexMaps() { return {src()}; }
    Value* getSinkIndexMap() { return sink(); }
    unsigned getNumOperands() { return 1; }
    AggregationKind getAggregationKind() { return AggregationKind::}] # agg # [{; }
    CombinationKind getCombinationKind() { return CombinationKind::}] # combo # [{; }
    Value* getInitializer() {
      if (getOperation()->getNumOperands() < 4) {
        return nullptr;
      }
      return getOperand(3);
    }
  }];
}

class BinaryConOp<string mnemonic, string agg, string combo> : ContractionOp<mnemonic> {
  let arguments = (ins
    AffineSizeMap:$size,
    AffineIndexMap:$src1,
    AffineIndexMap:$src2,
    AffineIndexMap:$sink,
    Variadic<RankedTensorOf<[AnyScalar]>>:$init
  );

  let extraClassDeclaration = [{
    static Operation* create(OpBuilder* builder, Location loc, Value* size, ArrayRef<Value*> srcs, Value* sink) {
      OperationState state(loc, getOperationName());
      state.addOperands(size);
      state.addOperands(srcs);
      state.addOperands(sink);
      state.setOperandListToResizable();
      return builder->createOperation(state);
    }
    Value* getSizeMap() { return size(); }
    std::vector<Value*> getSourceIndexMaps() { return {src1(), src2()}; }
    Value* getSinkIndexMap() { return sink(); }
    unsigned getNumOperands() { return 2; }
    AggregationKind getAggregationKind() { return AggregationKind::}] # agg # [{; }
    CombinationKind getCombinationKind() { return CombinationKind::}] # combo # [{; }
    Value* getInitializer() {
      if (getOperation()->getNumOperands() < 5) {
        return nullptr;
      }
      return getOperand(4);
    }
  }];
}

class TernaryConOp<string mnemonic, string agg, string combo> : ContractionOp<mnemonic> {
  let arguments = (ins
    AffineSizeMap:$size,
    AffineIndexMap:$src1,
    AffineIndexMap:$src2,
    AffineIndexMap:$src3,
    AffineIndexMap:$sink,
    Variadic<RankedTensorOf<[AnyScalar]>>:$init
  );

  let extraClassDeclaration = [{
    static Operation* create(OpBuilder* builder, Location loc, Value* size, ArrayRef<Value*> srcs, Value* sink) {
      OperationState state(loc, getOperationName());
      state.addOperands(size);
      state.addOperands(srcs);
      state.addOperands(sink);
      state.setOperandListToResizable();
      return builder->createOperation(state);
    }
    Value* getSizeMap() { return size(); }
    std::vector<Value*> getSourceIndexMaps() { return {src1(), src2(), src3()}; }
    Value* getSinkIndexMap() { return sink(); }
    unsigned getNumOperands() { return 3; }
    AggregationKind getAggregationKind() { return AggregationKind::}] # agg # [{; }
    CombinationKind getCombinationKind() { return CombinationKind::}] # combo # [{; }
    Value* getInitializer() {
      if (getOperation()->getNumOperands() < 6) {
        return nullptr;
      }
      return getOperand(5);
    }
  }];
}

// Assign
def ConAssignOp : UnaryConOp<"=(x)", "assign", "none"> {
  let summary = "=(x) contraction";
}

def ConAssignAddOp : BinaryConOp<"=(x+y)", "assign", "add"> {
  let summary = "=(x + y) contraction";
}

def ConAssignCondOp : TernaryConOp<"=(x==y?z)", "assign", "cond"> {
  let summary = "=(x == y ? z) contraction";
}

def ConAssignEqOp : BinaryConOp<"=(x==y)", "assign", "eq"> {
  let summary = "=(x == y) contraction";
}

def ConAssignMulOp : BinaryConOp<"=(x*y)", "assign", "mul"> {
  let summary = "=(x * y) contraction";
}

// Max
def ConMaxOp : UnaryConOp<">(x)", "max", "none"> {
  let summary = ">(x) contraction";
}

def ConMaxAddOp : BinaryConOp<">(x+y)", "max", "add"> {
  let summary = ">(x + y) contraction";
}

def ConMaxCondOp : TernaryConOp<">(x==y?z)", "max", "cond"> {
  let summary = ">(x == y ? z) contraction";
}

def ConMaxEqOp : BinaryConOp<">(x==y)", "max", "eq"> {
  let summary = ">(x == y) contraction";
}

def ConMaxMulOp : BinaryConOp<">(x*y)", "max", "mul"> {
  let summary = ">(x * y) contraction";
}

// Min
def ConMinOp : UnaryConOp<"<(x)", "min", "none"> {
  let summary = "<(x) contraction";
}

def ConMinAddOp : BinaryConOp<"<(x+y)", "min", "add"> {
  let summary = "<(x + y) contraction";
}

def ConMinCondOp : TernaryConOp<"<(x==y?z)", "min", "cond"> {
  let summary = "<(x == y ? z) contraction";
}

def ConMinEqOp : BinaryConOp<"<(x==y)", "min", "eq"> {
  let summary = "<(x == y) contraction";
}

def ConMinMulOp : BinaryConOp<"<(x*y)", "min", "mul"> {
  let summary = "<(x * y) contraction";
}

// Prod
def ConProdOp : UnaryConOp<"*(x)", "mul", "none"> {
  let summary = "*(x) contraction";
}

def ConProdAddOp : BinaryConOp<"*(x+y)", "mul", "add"> {
  let summary = "*(x + y) contraction";
}

def ConProdCondOp : TernaryConOp<"*(x==y?z)", "mul", "cond"> {
  let summary = "*(x == y ? z) contraction";
}

def ConProdEqOp : BinaryConOp<"*(x==y)", "mul", "eq"> {
  let summary = "*(x == y) contraction";
}

def ConProdMulOp : BinaryConOp<"*(x*y)", "mul", "mul"> {
  let summary = "*(x * y) contraction";
}

// Sum
def ConSumOp : UnaryConOp<"+(x)", "add", "none"> {
  let summary = "+(x) contraction";
}

def ConSumAddOp : BinaryConOp<"+(x+y)", "add", "add"> {
  let summary = "+(x + y) contraction";
}

def ConSumCondOp : TernaryConOp<"+(x==y?z)", "add", "cond"> {
  let summary = "+(x == y ? z) contraction";
}

def ConSumEqOp : BinaryConOp<"+(x==y)", "add", "eq"> {
  let summary = "+(x == y) contraction";
}

def ConSumMulOp : BinaryConOp<"+(x*y)", "add", "mul"> {
  let summary = "+(x * y) contraction";
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

def PrngOp : SpecialOp<"prng">, HasCanonicalizer {
  let summary = "pseudorandom number generator";
  let arguments = (ins RankedTensorOf<[AnyScalar]>:$state, Variadic<ScalarIndex>:$dims);
  let results = (outs RankedTensorOf<[AnyScalar]>:$result);
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
  let results = (outs VectorOf<[ScalarIndex]>:$result);
}
