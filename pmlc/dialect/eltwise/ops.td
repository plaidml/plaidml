// Copyright 2019, Intel Corporation

include "pmlc/dialect/eltwise/enums.td"
include "pmlc/dialect/eltwise/predicates.td"

def EltwiseDialect : Dialect {
  let name = "eltwise";
  let cppNamespace = "pmlc::dialect::eltwise";
}

def EltwiseBuilderInterface : OpInterface<"EltwiseBuilder"> {
  let methods = [
    StaticInterfaceMethod<"TODO", "Type", "getResultType", (ins "ArrayRef<Value*>":$operands)>,
    StaticInterfaceMethod<"TODO", "Operation*", "create", (ins
      "OpBuilder*":$builder,
      "Location":$loc,
      "ScalarType":$type,
      "ArrayRef<Value*>":$operands
    )>,
  ];
}

class EltwiseBuilderImpl {
  code EltwiseBuilderImpl_create = [{
    static Operation* create(OpBuilder* builder, Location loc, ScalarType type, ArrayRef<Value*> operands) {
      OperationState state(loc, getOperationName());
      state.addOperands(operands);
      state.addAttribute("type", TypeAttr::get(type));
      state.addTypes(getResultType(operands));
      return builder->createOperation(state);
    }
  }];
}

def EltwiseOpInterface : OpInterface<"EltwiseOp">;

class EltwiseOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<EltwiseDialect, mnemonic, !listconcat(traits, [EltwiseOpInterface, EltwiseBuilderInterface])>,
    EltwiseBuilderImpl;

def SpecialOpInterface : OpInterface<"SpecialOp">;
  
class SpecialOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<EltwiseDialect, mnemonic, !listconcat(traits, [SpecialOpInterface, EltwiseBuilderInterface])>,
    EltwiseBuilderImpl;

def ScalarConstantOp :
    Op<EltwiseDialect, "sconst", [NoSideEffect]>,
    HasFolder {
  let summary = "constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, double value", [{
      result.addAttribute("value", builder->getF32FloatAttr(value));
      result.addTypes(builder->getTensorType({}, type));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, int64_t value", [{
      result.addAttribute("value", builder->getI64IntegerAttr(value));
      result.addTypes(builder->getTensorType({}, type));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, Attribute value", [{
      result.addAttribute("value", value);
      result.addTypes(builder->getTensorType({}, type));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands", [{
      llvm_unreachable("unhandled transcoding");
    }]
  >];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
  }];
}

class UnaryOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EltwiseOp<mnemonic, traits>,
    HasCanonicalizer {
  let arguments = (ins Eltwise_AnyTensor:$in, TypeAttr:$type);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addAttribute("type", TypeAttr::get(type));
      result.addTypes(getResultType(operands));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, Value* operand", [{
      build(builder, result, type, llvm::SmallVector<Value*, 1>({operand}));
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands) {
      return ComputeResultType(operands);
    }
  }] # EltwiseBuilderImpl_create;
}

class BinaryOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EltwiseOp<mnemonic, traits>,
    HasCanonicalizer { 
  let arguments = (ins Eltwise_AnyTensor:$lhs, Eltwise_AnyTensor:$rhs, TypeAttr:$type);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addAttribute("type", TypeAttr::get(type));
      result.addTypes(getResultType(operands));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, Value* lhs, Value* rhs", [{
      build(builder, result, type, {lhs, rhs});
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands) {
      return ComputeResultType(operands);
    }
  }] # EltwiseBuilderImpl_create;
}

class CompareOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EltwiseOp<mnemonic, traits>,
    HasCanonicalizer {
  let arguments = (ins Eltwise_AnyTensor:$lhs, Eltwise_AnyTensor:$rhs, TypeAttr:$type);
  let results = (outs Eltwise_BoolTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addAttribute("type", TypeAttr::get(type));
      result.addTypes(getResultType(operands));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, Value* lhs, Value* rhs", [{
      build(builder, result, type, {lhs, rhs});
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands) {
      return ComputeResultType(operands, DataType::BOOLEAN);
    }
  }] # EltwiseBuilderImpl_create;
}

class UnaryIntOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    UnaryOp<mnemonic, traits>,
    Arguments<(ins Eltwise_IntegerTensor:$in, TypeAttr:$type)>,
    Results<(outs Eltwise_IntegerTensor:$result)>;

class UnaryFloatOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    UnaryOp<mnemonic, traits>,
    Arguments<(ins Eltwise_FloatTensor:$in, TypeAttr:$type)>,
    Results<(outs Eltwise_FloatTensor:$result)>;

class BinaryIntOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    BinaryOp<mnemonic, traits>,
    Arguments<(ins Eltwise_IntegerTensor:$lhs, Eltwise_IntegerTensor:$rhs, TypeAttr:$type)>,
    Results<(outs Eltwise_IntegerTensor:$result)>;

def CastOpInterface : OpInterface<"CastOp">;

class CastOp<string mnemonic> :
    EltwiseOp<mnemonic, [CastOpInterface, NoSideEffect]>,
    HasCanonicalizer {
  let arguments = (ins Eltwise_AnyTensor:$tensor);

  let builders = [OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands"
  >, OpBuilder<
    "Builder* builder, OperationState result, Value* tensor, IntegerAttr bitwidth"
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands);

    static Operation* create(OpBuilder* builder, Location loc, ScalarType type, ArrayRef<Value*> operands) {
      OperationState state(loc, getOperationName());
      if (operands.size() != 2) {
        throw std::runtime_error("CastOp requires 2 operands");
      }
      auto tensor = operands[0];
      state.addOperands(tensor);
      state.addTypes(getResultType(operands));
      return builder->createOperation(state);
    }
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

def AsBoolOp : EltwiseOp<"as_bool"> {
  let summary = "cast into boolean operation";
  let arguments = (ins Eltwise_AnyTensor:$in);
  let results = (outs Eltwise_BoolTensor:$result);
  let builders = [OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addTypes(getResultType(operands));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, Value* val", [{
      build(builder, result, type, std::vector<Value*>({val}));
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands) {
      auto tensorType = GetTensorType(operands[0]->getType());
      auto elementType = ScalarType::get(tensorType.getContext(), DataType::BOOLEAN);
      return RankedTensorType::get(tensorType.getShape(), elementType);
    }
  }] # EltwiseBuilderImpl_create;
}

def AbsOp : UnaryOp<"abs">;
def ACosOp : UnaryFloatOp<"acos">;
def AddOp : BinaryOp<"add", [Commutative, NoSideEffect]>;
def AndOp : BinaryIntOp<"and", [Commutative, NoSideEffect]>;
def ASinOp : UnaryFloatOp<"asin">;
def AssignOp : UnaryOp<"assign">;
def ATanOp : UnaryFloatOp<"atan">;
def CeilOp : UnaryOp<"ceil">;
def CmpEqOp : CompareOp<"cmp_eq", [Commutative, NoSideEffect]>;
def CmpGeOp : CompareOp<"cmp_ge">;
def CmpGtOp : CompareOp<"cmp_gt">;
def CmpLeOp : CompareOp<"cmp_le">;
def CmpLtOp : CompareOp<"cmp_lt">;
def CmpNeOp : CompareOp<"cmp_ne", [Commutative, NoSideEffect]>;
def CosHOp : UnaryFloatOp<"cosh">;
def CosOp : UnaryFloatOp<"cos">;
def DivOp : BinaryOp<"div">;
def ExpOp : UnaryFloatOp<"exp">;
def FloorOp : UnaryOp<"floor">;
// Prevent CSE from dropping 'ident' by avoiding the NoSideEffect trait
def IdentOp : UnaryOp<"ident", []>;
def LogOp : UnaryFloatOp<"log">;
def MaxOp : BinaryOp<"max", [Commutative, NoSideEffect]>;
def MinOp : BinaryOp<"min", [Commutative, NoSideEffect]>;
def ModOp : BinaryOp<"mod">;
def MulOp : BinaryOp<"mul", [Commutative, NoSideEffect]>, HasFolder;
def NegOp : UnaryOp<"neg">;
def NotOp : UnaryIntOp<"not">;
def OrOp : BinaryIntOp<"or", [Commutative, NoSideEffect]>;
def PowOp : BinaryOp<"pow">;
def ReluOp : UnaryOp<"relu">;
def RoundOp : UnaryOp<"round">;
def ShlOp : BinaryIntOp<"shl">;
def ShrOp : BinaryIntOp<"shr">;
def SignOp : UnaryOp<"sign">;
def SinHOp : UnaryFloatOp<"sinh">;
def SinOp : UnaryFloatOp<"sin">;
def SqrtOp : UnaryFloatOp<"sqrt">;
def SubOp : BinaryOp<"sub">;
def TanHOp : UnaryFloatOp<"tanh">;
def TanOp : UnaryFloatOp<"tan">;
def XorOp : BinaryIntOp<"xor", [Commutative, NoSideEffect]>;

def GatherOp : SpecialOp<"gather">, HasCanonicalizer {
  let summary = "special gather operation";
  let arguments = (ins Eltwise_AnyTensor:$tensor, Eltwise_IntegerTensor:$dims);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands);
  }] # EltwiseBuilderImpl_create;
}

def IndexOp : SpecialOp<"index">, HasCanonicalizer {
  let summary = "tensor index lookup operation";
  let arguments = (ins Eltwise_AnyTensor:$tensor, Eltwise_Integer:$dim);
  let results = (outs Eltwise_IntegerTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands);
  }] # EltwiseBuilderImpl_create;
}

def PrngOp : SpecialOp<"prng">, HasCanonicalizer {
  let summary = "pseudorandom number generator";
  let arguments = (ins Eltwise_AnyTensor:$state, Variadic<Eltwise_Integer>:$dims);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands);
  }] # EltwiseBuilderImpl_create;
}

def ReshapeOp : SpecialOp<"reshape">, HasCanonicalizer {
  let summary = "tensor reshape operation";
  let arguments = (ins Eltwise_AnyTensor:$tensor, Variadic<Eltwise_Integer>:$dims);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands);
  }] # EltwiseBuilderImpl_create;
}

// TODO
// def ScatterOp : SpecialOp<"scatter">, HasCanonicalizer {
//   let summary = "special scatter operation";
//   let arguments = (ins Eltwise_AnyTensor:$tensor, Eltwise_IntegerTensor:$dims, Eltwise_AnyTensor:$other);
//   let results = (outs Eltwise_AnyTensor:$result);

//   let builders = [OpBuilder<
//     "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands", [{
//       result.addOperands(operands);
//       result.addTypes(getResultType(operands));
//     }]
//   >];

//   let extraClassDeclaration = [{
//     static Type getResultType(ArrayRef<Value*> operands);
//   }];
// }

def SelectOp : EltwiseOp<"select">, HasCanonicalizer {
  let summary = "conditional selection";
  let arguments = (ins
    Eltwise_BoolTensor:$cond,
    Eltwise_AnyTensor:$tcase,
    Eltwise_AnyTensor:$fcase,
    TypeAttr:$type);
  let results = (outs Eltwise_AnyTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addAttribute("type", TypeAttr::get(type));
      result.addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands) {
      return ComputeResultType(operands.drop_front());
    }
  }] # EltwiseBuilderImpl_create;
}

def ShapeOp : SpecialOp<"shape">, HasCanonicalizer {
  let summary = "tensor shape operation";
  let arguments = (ins Eltwise_AnyTensor:$tensor);
  let results = (outs Eltwise_IntegerTensor:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands);
  }] # EltwiseBuilderImpl_create;
}
