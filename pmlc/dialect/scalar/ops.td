// Copyright 2019, Intel Corporation

include "pmlc/dialect/scalar/predicates.td"

def ScalarDialect : Dialect {
  let name = "pml_scalar";
  let cppNamespace = "pmlc::dialect::scalar";
}

def ScalarConstantOp :
    Op<ScalarDialect, "constant", [NoSideEffect]>,
    HasFolder {
  let summary = "constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs ScalarType);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, Attribute value", [{
      result->addAttribute("value", value);
      result->addTypes(type);
  }]>, OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      llvm_unreachable("unhandled transcoding");
  }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
    static constexpr int operands() { return 0; }
  }];
}

class UnaryOp<string mnemonic, list<OpTrait> traits = []> : 
    Op<ScalarDialect, mnemonic, !listconcat(traits, [NoSideEffect])> {
  let arguments = (ins ScalarType:$in, TypeAttr:$type);
  let results = (outs ScalarType:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addAttribute("type", TypeAttr::get(type));
      result->addTypes(type);
  }]>, OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, Value* operand", [{
      build(builder, result, type, llvm::SmallVector<Value*, 1>({operand}));
  }]>];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 1; }
  }];
}

class BinaryOp<string mnemonic, list<OpTrait> traits = []> :
    Op<ScalarDialect, mnemonic, !listconcat(traits, [NoSideEffect])> { 
  let arguments = (ins ScalarType:$lhs, ScalarType:$rhs, TypeAttr:$type);
  let results = (outs ScalarType:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addAttribute("type", TypeAttr::get(type));
      result->addTypes(type);
  }]>, OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, Value* lhs, Value* rhs", [{
      build(builder, result, type, {lhs, rhs});
  }]>];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 2; }
  }];
}

class CompareOp<string mnemonic, list<OpTrait> traits = []> :
    Op<ScalarDialect, mnemonic, !listconcat(traits, [NoSideEffect])> {
  let arguments = (ins ScalarType:$lhs, ScalarType:$rhs, TypeAttr:$type);
  let results = (outs BooleanType:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, ArrayRef<Value*> operands", [{
      result->addOperands(operands);
      result->addAttribute("type", TypeAttr::get(type));
      result->addTypes(ScalarType::get(result->getContext(), DataType::BOOLEAN));
  }]>, OpBuilder<
    "Builder* builder, OperationState* result, ScalarType type, Value* lhs, Value* rhs", [{
      build(builder, result, type, {lhs, rhs});
  }]>];

  let extraClassDeclaration = [{
    static constexpr int operands() { return 2; }
  }];
}

class UnaryIntOp<string mnemonic, list<OpTrait> traits = []> :
    UnaryOp<mnemonic, traits>,
    Arguments<(ins ScalarIntegerLike:$in, TypeAttr:$type)>,
    Results<(outs ScalarIntegerLike)>;

class UnaryFloatOp<string mnemonic, list<OpTrait> traits = []> :
    UnaryOp<mnemonic, traits>,
    Arguments<(ins ScalarFloatLike:$in, TypeAttr:$type)>,
    Results<(outs ScalarFloatLike)>;

class BinaryIntOp<string mnemonic, list<OpTrait> traits = []> :
    BinaryOp<mnemonic, traits>,
    Arguments<(ins ScalarIntegerLike:$in, TypeAttr:$type)>,
    Results<(outs ScalarIntegerLike)>;

def AbsOp : UnaryOp<"abs">;
def ACosOp : UnaryFloatOp<"acos">;
def AddOp : BinaryOp<"add", [Commutative]>, HasFolder;
def AndOp : BinaryIntOp<"and", [Commutative]>;
def ASinOp : UnaryFloatOp<"asin">;
def ATanOp : UnaryFloatOp<"atan">;
def CmpEqOp : CompareOp<"cmp_eq", [Commutative]>;
def CmpGeOp : CompareOp<"cmp_ge">;
def CmpGtOp : CompareOp<"cmp_gt">;
def CmpLeOp : CompareOp<"cmp_le">;
def CmpLtOp : CompareOp<"cmp_lt">;
def CmpNeOp : CompareOp<"cmp_ne", [Commutative]>;
def CosHOp : UnaryFloatOp<"cosh">;
def CosOp : UnaryFloatOp<"cos">;
def DivOp : BinaryOp<"div">, HasFolder;
def ExpOp : UnaryFloatOp<"exp">;
def LogOp : UnaryFloatOp<"log">;
def ModOp : BinaryOp<"mod">;
def MulOp : BinaryOp<"mul", [Commutative]>, HasFolder;
def NegOp : UnaryOp<"neg">, HasFolder;
def NotOp : UnaryIntOp<"not">;
def OrOp : BinaryIntOp<"or", [Commutative]>;
def ReluOp : UnaryOp<"relu">;
def ShlOp : BinaryIntOp<"shl">;
def ShrOp : BinaryIntOp<"shr">;
def SignOp : UnaryOp<"sign">;
def SinHOp : UnaryFloatOp<"sinh">;
def SinOp : UnaryFloatOp<"sin">;
def SqrtOp : UnaryFloatOp<"sqrt">;
def SubOp : BinaryOp<"sub">, HasFolder;
def TanHOp : UnaryFloatOp<"tanh">;
def TanOp : UnaryFloatOp<"tan">;
def XorOp : BinaryIntOp<"xor", [Commutative]>;

// Cast
// Select
