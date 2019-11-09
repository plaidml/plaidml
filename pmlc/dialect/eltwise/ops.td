// Copyright 2019, Intel Corporation
#ifndef __PML_ELTWISE_OPS__
#define __PML_ELTWISE_OPS__

#ifndef __PML_UTIL_INTERFACES__
include "pmlc/util/interfaces.td"
#endif

#ifndef __PML_ELTWISE_INTERFACES__
include "pmlc/dialect/eltwise/interfaces.td"
#endif

#ifndef __PML_ELTWISE_PREDICATES__
include "pmlc/dialect/eltwise/predicates.td"
#endif

def EltwiseDialect : Dialect {
  let name = "eltwise";
  let cppNamespace = "pmlc::dialect::eltwise";
}

class EltwiseBuilderImpl {
  code EltwiseBuilderImpl_create = [{
    static Operation* create(OpBuilder* builder, Location loc, Type type, ArrayRef<Value*> operands) {
      OperationState state(loc, getOperationName());
      state.addOperands(operands);
      state.addAttribute("type", TypeAttr::get(type));
      state.addTypes(getResultType(operands));
      return builder->createOperation(state);
    }
  }];
}

class EltwiseOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<EltwiseDialect, mnemonic, !listconcat(traits, [EltwiseOpInterface, GenericBuilderInterface])>,
    EltwiseBuilderImpl;

def ScalarConstantOp :
    Op<EltwiseDialect, "sconst", [NoSideEffect]>,
    HasFolder {
  let summary = "constant";
  let arguments = (ins AnyAttr:$value);
  let results = (outs EltwiseAny:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, Type type, double value", [{
      result.addAttribute("value", builder->getF64FloatAttr(value));
      result.addTypes(RankedTensorType::get({}, type));
    }]
  >, OpBuilder<
    "Builder* builder, OperationState& result, Type type, int64_t value", [{
      result.addAttribute("value", builder->getI64IntegerAttr(value));
      result.addTypes(RankedTensorType::get({}, type));
    }]
  >];

  let extraClassDeclaration = [{
    Attribute getValue() { return getAttr("value"); }
    IntegerAttr getIntAttr() { return getAttrOfType<IntegerAttr>("value"); }
    FloatAttr getFloatAttr() { return getAttrOfType<FloatAttr>("value"); }
  }];
}

class UnaryOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EltwiseOp<mnemonic, traits>,
    HasCanonicalizer {
  let arguments = (ins EltwiseAny:$in, TypeAttr:$type);
  let results = (outs EltwiseAny:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addAttribute("type", TypeAttr::get(type));
      result.addTypes(getResultType(operands));
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
  let arguments = (ins EltwiseAny:$lhs, EltwiseAny:$rhs, TypeAttr:$type);
  let results = (outs EltwiseAny:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addAttribute("type", TypeAttr::get(type));
      result.addTypes(getResultType(operands));
    }]
  >];

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands) {
      return ComputeResultType(operands);
    }
  }] # EltwiseBuilderImpl_create;
}

def CastOp : EltwiseOp<"cast">, HasCanonicalizer {
  let summary = "cast operation";
  let arguments = (ins EltwiseAny:$tensor);
  let results = (outs EltwiseAny:$result);

  let extraClassDeclaration = [{
    static Type getResultType(ArrayRef<Value*> operands);
    static Operation* create(OpBuilder* builder, Location loc, Type type, ArrayRef<Value*> operands);
  }];
}

class CompareOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    EltwiseOp<mnemonic, traits>,
    HasCanonicalizer {
  let arguments = (ins EltwiseAny:$lhs, EltwiseAny:$rhs, TypeAttr:$type);
  let results = (outs EltwiseBool:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ScalarType type, ArrayRef<Value*> operands", [{
      result.addOperands(operands);
      result.addAttribute("type", TypeAttr::get(type));
      result.addTypes(getResultType(operands));
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
    Arguments<(ins EltwiseInteger:$in, TypeAttr:$type)>,
    Results<(outs EltwiseInteger:$result)>;

class UnaryFloatOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    UnaryOp<mnemonic, traits>,
    Arguments<(ins EltwiseFloat:$in, TypeAttr:$type)>,
    Results<(outs EltwiseFloat:$result)>;

class BinaryIntOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    BinaryOp<mnemonic, traits>,
    Arguments<(ins EltwiseInteger:$lhs, EltwiseInteger:$rhs, TypeAttr:$type)>,
    Results<(outs EltwiseInteger:$result)>;

def AbsOp : UnaryOp<"abs">;
def ACosOp : UnaryFloatOp<"acos">;
def AddOp : BinaryOp<"add", [Commutative, NoSideEffect]>, HasFolder;
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

def SelectOp : EltwiseOp<"select">, HasCanonicalizer {
  let summary = "conditional selection";
  let arguments = (ins
    EltwiseBool:$cond,
    EltwiseAny:$tcase,
    EltwiseAny:$fcase,
    TypeAttr:$type);
  let results = (outs EltwiseAny:$result);

  let builders = [OpBuilder<
    "Builder* builder, OperationState& result, ScalarType type, ArrayRef<Value*> operands", [{
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

#endif // __PML_ELTWISE_OPS__
