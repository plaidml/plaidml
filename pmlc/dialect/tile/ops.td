// Copyright 2019, Intel Corporation

include "pmlc/dialect/eltwise/enums.td"
include "pmlc/dialect/eltwise/predicates.td"

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
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, RankedTensorType type", [{
      result->addTypes(type);
    }]
  >];
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
}

def DimOp : TileOp<"dim">, HasFolder {
  let summary = "dimension index operation";
  let arguments = (ins Eltwise_AnyTensor:$tensor, APIntAttr:$dim);
  let results = (outs Index:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, Value* tensor, int64_t value", [{
      auto type = builder->getIndexType();
      result->addOperands(tensor);
      result->addAttribute("dim", builder->getIntegerAttr(type, value));
      result->addTypes(type);
    }]
  >];
}

def AffineConstantOp : TileOp<"const_dim">, HasFolder {
  let summary = "affine constant";
  let arguments = (ins I64Attr:$value);
  let results = (outs Index:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, int64_t value", [{
      result->addAttribute("value", builder->getI64IntegerAttr(value));
      result->addTypes(builder->getIndexType());
    }]
  >];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
  }];
}

def AffineIndexOp : TileOp<"idx"> {
  let summary = "affine index";
  let results = (outs Index:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result", [{
      result->addTypes(builder->getIndexType());
    }]
  >];
}

class AffineOp<string mnemonic> : TileOp<mnemonic, [NoSideEffect]>, HasFolder {
  let results = (outs Index:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addTypes(builder->getIndexType());
    }]
  >];
}

class AffineUnaryOp<string mnemonic> : AffineOp<mnemonic> {
  let arguments = (ins Index:$in);
}

class AffineBinaryOp<string mnemonic> : AffineOp<mnemonic> {
  let arguments = (ins Index:$lhs, Index:$rhs);
}

def AffineAddOp : AffineBinaryOp<"affine_add"> {
  let summary = "Affine addition operation";
}

def AffineDivOp : AffineBinaryOp<"affine_div"> {
  let summary = "Affine division operation";
}

def AffineMulOp : AffineBinaryOp<"affine_mul"> {
  let summary = "Affine multiplication operation";
}

def AffineNegOp : AffineUnaryOp<"affine_neg"> {
  let summary = "Affine negative operation";
  let arguments = (ins Index:$input);
}

def AffineSubOp : AffineBinaryOp<"affine_sub"> {
  let summary = "Affine subtraction operation";
}

def AffineDomainOp : TileOp<"domain">, HasCanonicalizer {
  let summary = "affine domain";
  let results = (outs Eltwise_AnyTensor:$result);
  let regions = (region SizedRegion<1>:$body);
}

def AffineSourceIndexMapOp : TileOp<"src_idx_map", [HasParent<"AffineDomainOp">]> {
  let summary = "affine source index map";
  let arguments = (ins Eltwise_AnyTensor:$tensor, Variadic<Index>:$dims);
  let results = (outs AffineIndexMap:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, Value* tensor, ArrayRef<Value*> dims", [{
      result->addOperands(tensor);
      result->addOperands(dims);
      result->addTypes(builder->getType<AffineIndexMapType>());
    }]
  >];
}

def AffineSinkIndexMapOp : TileOp<"sink_idx_map", [HasParent<"AffineDomainOp">]> {
  let summary = "affine sink index map";
  let arguments = (ins Variadic<Index>:$dims);
  let results = (outs AffineIndexMap:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ArrayRef<Value*> dims", [{
      result->addOperands(dims);
      result->addTypes(builder->getType<AffineIndexMapType>());
    }]
  >];
}

def AffineSizeMapOp : TileOp<"size_map", [HasParent<"AffineDomainOp">]> {
  let summary = "affine size map";
  let arguments = (ins Variadic<Index>:$sizes);
  let results = (outs AffineSizeMap:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ArrayRef<Value*> sizes", [{
      result->addOperands(sizes);
      result->addTypes(builder->getType<AffineSizeMapType>());
    }]
  >];
}

def ContractionOpInterface : OpInterface<"ContractionOp"> {
  let methods = [
    InterfaceMethod<"std::vector<Value*>", "getSourceIndexMaps">,
    InterfaceMethod<"Value*", "getSinkIndexMap">,
    InterfaceMethod<"Value*", "getSizeMap">,
    InterfaceMethod<"unsigned", "getNumOperands">,
    InterfaceMethod<"AggregationKind", "getAggregationKind">,
    InterfaceMethod<"CombinationKind", "getCombinationKind">,
  ];
}

class ContractionOp<string mnemonic, list<OpTrait> traits = []> :
    TileOp<mnemonic, !listconcat(traits, [
      Terminator,
      HasParent<"AffineDomainOp">,
      ContractionOpInterface
    ])>;

class UnaryConOp<string mnemonic, string agg, string combo> : ContractionOp<mnemonic> {
  let arguments = (ins
    AffineSizeMap:$size,
    AffineIndexMap:$src,
    AffineIndexMap:$sink);

  let extraClassDeclaration = [{
    Value* getSizeMap() { return size(); }
    std::vector<Value*> getSourceIndexMaps() { return {src()}; }
    Value* getSinkIndexMap() { return sink(); }
    unsigned getNumOperands() { return 1; }
    AggregationKind getAggregationKind() { return AggregationKind::}] # agg # [{; }
    CombinationKind getCombinationKind() { return CombinationKind::}] # combo # [{; }
  }];
}

class BinaryConOp<string mnemonic, string agg, string combo> : ContractionOp<mnemonic> {
  let arguments = (ins
    AffineSizeMap:$size,
    AffineIndexMap:$src1,
    AffineIndexMap:$src2,
    AffineIndexMap:$sink);

  let extraClassDeclaration = [{
    Value* getSizeMap() { return size(); }
    std::vector<Value*> getSourceIndexMaps() { return {src1(), src2()}; }
    Value* getSinkIndexMap() { return sink(); }
    unsigned getNumOperands() { return 2; }
    AggregationKind getAggregationKind() { return AggregationKind::}] # agg # [{; }
    CombinationKind getCombinationKind() { return CombinationKind::}] # combo # [{; }
  }];
}

class TernaryConOp<string mnemonic, string agg, string combo> : ContractionOp<mnemonic> {
  let arguments = (ins
    AffineSizeMap:$size,
    AffineIndexMap:$src1,
    AffineIndexMap:$src2,
    AffineIndexMap:$src3,
    AffineIndexMap:$sink);

  let extraClassDeclaration = [{
    Value* getSizeMap() { return size(); }
    std::vector<Value*> getSourceIndexMaps() { return {src1(), src2(), src3()}; }
    Value* getSinkIndexMap() { return sink(); }
    unsigned getNumOperands() { return 3; }
    AggregationKind getAggregationKind() { return AggregationKind::}] # agg # [{; }
    CombinationKind getCombinationKind() { return CombinationKind::}] # combo # [{; }
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
