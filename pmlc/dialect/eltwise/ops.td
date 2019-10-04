// Copyright 2019, Intel Corporation

include "pmlc/dialect/eltwise/predicates.td"

def EltwiseDialect : Dialect {
  let name = "eltwise";
  let cppNamespace = "pmlc::dialect::eltwise";
}

class EltwiseOp<string mnemonic, list<OpTrait> traits = []> :
    Op<EltwiseDialect, mnemonic, traits>;
  
def ScalarConstantOp : EltwiseOp<"sconst", [NoSideEffect]>, HasFolder {
  let summary = "constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, double value", [{
      result->addAttribute("value", builder->getF32FloatAttr(value));
      result->addTypes(builder->getTensorType({}, type));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, int64_t value", [{
      result->addAttribute("value", builder->getI64IntegerAttr(value));
      result->addTypes(builder->getTensorType({}, type));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, Attribute value", [{
      result->addAttribute("value", value);
      result->addTypes(builder->getTensorType({}, type));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      llvm_unreachable("unhandled transcoding");
    }]
  >];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
    static constexpr int operands() { return 0; }
  }];
}

class UnaryOp<string mnemonic, list<OpTrait> traits = []> :
    EltwiseOp<mnemonic, traits>,
    HasCanonicalizer {
  let arguments = (ins Eltwise_AnyTensor:$in, TypeAttr:$type);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addAttribute("type", TypeAttr::get(type));
      result->addTypes(getResultType(operands));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, Value* operand", [{
      build(builder, result, type, llvm::SmallVector<Value*, 1>({operand}));
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 1; }
    static Type getResultType(ArrayRef<Value*> operands) {
      return ComputeResultType(operands);
    }
  }];
}

class BinaryOp<string mnemonic, list<OpTrait> traits = []> :
    EltwiseOp<mnemonic, traits>,
    HasCanonicalizer { 
  let arguments = (ins Eltwise_AnyTensor:$lhs, Eltwise_AnyTensor:$rhs, TypeAttr:$type);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addAttribute("type", TypeAttr::get(type));
      result->addTypes(getResultType(operands));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, Value* lhs, Value* rhs", [{
      build(builder, result, type, {lhs, rhs});
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 2; }
    static Type getResultType(ArrayRef<Value*> operands) {
      return ComputeResultType(operands);
    }
  }];
}

class CompareOp<string mnemonic, list<OpTrait> traits = []> :
    EltwiseOp<mnemonic, traits>,
    HasCanonicalizer {
  let arguments = (ins Eltwise_AnyTensor:$lhs, Eltwise_AnyTensor:$rhs, TypeAttr:$type);
  let results = (outs Eltwise_BoolTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addAttribute("type", TypeAttr::get(type));
      result->addTypes(getResultType(operands));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, Value* lhs, Value* rhs", [{
      build(builder, result, type, {lhs, rhs});
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 2; }
    static Type getResultType(ArrayRef<Value*> operands) {
      return ComputeResultType(operands, DataType::BOOLEAN);
    }
  }];
}

class UnaryIntOp<string mnemonic, list<OpTrait> traits = []> :
    UnaryOp<mnemonic, traits>,
    Arguments<(ins Eltwise_IntegerTensor:$in, TypeAttr:$type)>,
    Results<(outs Eltwise_IntegerTensor:$result)>;

class UnaryFloatOp<string mnemonic, list<OpTrait> traits = []> :
    UnaryOp<mnemonic, traits>,
    Arguments<(ins Eltwise_FloatTensor:$in, TypeAttr:$type)>,
    Results<(outs Eltwise_FloatTensor:$result)>;

class BinaryIntOp<string mnemonic, list<OpTrait> traits = []> :
    BinaryOp<mnemonic, traits>,
    Arguments<(ins Eltwise_IntegerTensor:$in, TypeAttr:$type)>,
    Results<(outs Eltwise_IntegerTensor:$result)>;

class CastOp<string mnemonic> : EltwiseOp<mnemonic, [NoSideEffect]>, HasCanonicalizer {
  let arguments = (ins Eltwise_AnyTensor:$tensor, APIntAttr:$bitwidth);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands"
  >, OpBuilder<
    "Builder* builder, OperationState* result, Value* tensor, IntegerAttr bitwidth"
  >];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 1; }
    static Type getResultType(Value* tensor, IntegerAttr bitwidth);
  }];
}

def AsFloatOp : CastOp<"as_float"> {
  let summary = "cast into float operation";
  let results = (outs Eltwise_FloatTensor:$result);
}

def AsIntOp : CastOp<"as_int"> {
  let summary = "cast into integer operation";
  let results = (outs Eltwise_IntegerTensor:$result);
}

def AsUIntOp : CastOp<"as_uint"> {
  let summary = "cast into unsigned operation";
  let results = (outs Eltwise_UnsignedTensor:$result);
}

def AsBoolOp : EltwiseOp<"as_bool", [NoSideEffect]> {
  let summary = "cast into boolean operation";
  let arguments = (ins Eltwise_AnyTensor:$in, TypeAttr:$type);
  let results = (outs Eltwise_BoolTensor:$result);
  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addAttribute("type", TypeAttr::get(getResultType(operands)));
      result->addTypes(getResultType(operands));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, Value* val", [{
      build(builder, result, type, std::vector<Value*>({val}));
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 1; }
    static Type getResultType(ArrayRef<Value*> operands) {
      auto tensorType = GetTensorType(operands[0]->getType());
      auto elementType = ScalarType::get(operands[0]->getContext(), DataType::BOOLEAN);
      return RankedTensorType::get(tensorType.getShape(), elementType);
    }
  }];
}

def AbsOp : UnaryOp<"abs", [NoSideEffect]>;
def ACosOp : UnaryFloatOp<"acos", [NoSideEffect]>;
def AddOp : BinaryOp<"add", [Commutative, NoSideEffect]>;
def AndOp : BinaryIntOp<"and", [Commutative, NoSideEffect]>;
def ASinOp : UnaryFloatOp<"asin", [NoSideEffect]>;
def AssignOp : UnaryOp<"assign", [NoSideEffect]>;
def ATanOp : UnaryFloatOp<"atan", [NoSideEffect]>;
def CeilOp : UnaryOp<"ceil", [NoSideEffect]>;
def CmpEqOp : CompareOp<"cmp_eq", [Commutative, NoSideEffect]>;
def CmpGeOp : CompareOp<"cmp_ge", [NoSideEffect]>;
def CmpGtOp : CompareOp<"cmp_gt", [NoSideEffect]>;
def CmpLeOp : CompareOp<"cmp_le", [NoSideEffect]>;
def CmpLtOp : CompareOp<"cmp_lt", [NoSideEffect]>;
def CmpNeOp : CompareOp<"cmp_ne", [Commutative, NoSideEffect]>;
def CosHOp : UnaryFloatOp<"cosh", [NoSideEffect]>;
def CosOp : UnaryFloatOp<"cos", [NoSideEffect]>;
def DivOp : BinaryOp<"div", [NoSideEffect]>;
def ExpOp : UnaryFloatOp<"exp", [NoSideEffect]>;
def FloorOp : UnaryOp<"floor", [NoSideEffect]>;
// Prevent CSE from dropping 'ident' by avoiding the NoSideEffect trait
def IdentOp : UnaryOp<"ident">;
def LogOp : UnaryFloatOp<"log", [NoSideEffect]>;
def MaxOp : BinaryOp<"max", [Commutative, NoSideEffect]>;
def MinOp : BinaryOp<"min", [Commutative, NoSideEffect]>;
def ModOp : BinaryOp<"mod", [NoSideEffect]>;
def MulOp : BinaryOp<"mul", [Commutative, NoSideEffect]>, HasFolder;
def NegOp : UnaryOp<"neg", [NoSideEffect]>;
def NotOp : UnaryIntOp<"not", [NoSideEffect]>;
def OrOp : BinaryIntOp<"or", [Commutative, NoSideEffect]>;
def PowOp : BinaryOp<"pow", [NoSideEffect]>;
def ReluOp : UnaryOp<"relu", [NoSideEffect]>;
def RoundOp : UnaryOp<"round", [NoSideEffect]>;
def ShlOp : BinaryIntOp<"shl", [NoSideEffect]>;
def ShrOp : BinaryIntOp<"shr", [NoSideEffect]>;
def SignOp : UnaryOp<"sign", [NoSideEffect]>;
def SinHOp : UnaryFloatOp<"sinh", [NoSideEffect]>;
def SinOp : UnaryFloatOp<"sin", [NoSideEffect]>;
def SqrtOp : UnaryFloatOp<"sqrt", [NoSideEffect]>;
def SubOp : BinaryOp<"sub", [NoSideEffect]>;
def TanHOp : UnaryFloatOp<"tanh", [NoSideEffect]>;
def TanOp : UnaryFloatOp<"tan", [NoSideEffect]>;
def XorOp : BinaryIntOp<"xor", [Commutative, NoSideEffect]>;

def GatherOp : EltwiseOp<"gather", [NoSideEffect]>, HasCanonicalizer {
  let summary = "special gather operation";
  let arguments = (ins Eltwise_AnyTensor:$tensor, Eltwise_IntegerTensor:$dims);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 2; }
    static Type getResultType(ArrayRef<Value*> operands);
  }];
}

def IndexOp : EltwiseOp<"index", [NoSideEffect]>, HasCanonicalizer {
  let summary = "tensor index lookup operation";
  let arguments = (ins Eltwise_AnyTensor:$tensor, Eltwise_Integer:$dim);
  let results = (outs Eltwise_IntegerTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 2; }
    static Type getResultType(ArrayRef<Value*> operands);
  }];
}

def PrngOp : EltwiseOp<"prng">, HasCanonicalizer {
  let summary = "pseudorandom number generator";
  let arguments = (ins Eltwise_AnyTensor:$state, Variadic<Eltwise_Integer>:$dims);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 3; }
    static Type getResultType(ArrayRef<Value*> operands);
  }];
}

def ReshapeOp : EltwiseOp<"reshape", [NoSideEffect]>, HasCanonicalizer {
  let summary = "tensor reshape operation";
  let arguments = (ins Eltwise_AnyTensor:$tensor, Variadic<Eltwise_Integer>:$dims);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 2; }
    static Type getResultType(ArrayRef<Value*> operands);
  }];
}

// TODO
// def ScatterOp : EltwiseOp<"scatter", [NoSideEffect]>, HasCanonicalizer {
//   let summary = "special scatter operation";
//   let arguments = (ins Eltwise_AnyTensor:$tensor, Eltwise_IntegerTensor:$dims, Eltwise_AnyTensor:$other);
//   let results = (outs Eltwise_AnyTensor:$result);

//   let builders = [OpBuilder<
//     "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
//       result->addOperands(operands);
//       result->addTypes(getResultType(operands));
//     }]
//   >];

//   let extraClassDeclaration = [{
//     static constexpr int operands() { return 2; }
//     static Type getResultType(ArrayRef<Value*> operands);
//   }];
// }

def SelectOp : EltwiseOp<"select", [NoSideEffect]>, HasCanonicalizer {
  let summary = "conditional selection";
  let arguments = (ins
    Eltwise_BoolTensor:$cond,
    Eltwise_AnyTensor:$tcase,
    Eltwise_AnyTensor:$fcase,
    TypeAttr:$type);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addAttribute("type", TypeAttr::get(type));
      result->addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 3; }
    static Type getResultType(ArrayRef<Value*> operands) {
      return ComputeResultType(operands.drop_front());
    }
  }];
}

def ShapeOp : EltwiseOp<"shape", [NoSideEffect]>, HasCanonicalizer {
  let summary = "tensor shape operation";
  let arguments = (ins Eltwise_AnyTensor:$tensor);
  let results = (outs Eltwise_IntegerTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 1; }
    static Type getResultType(ArrayRef<Value*> operands);
  }];
}
