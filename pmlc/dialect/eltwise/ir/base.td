#ifndef __PML_ELTWISE_BASE__
#define __PML_ELTWISE_BASE__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

def EltwiseDialect : Dialect {
  let name = "eltwise";
  let cppNamespace = "pmlc::dialect::eltwise";
  let hasConstantMaterializer = 1;
  let extraClassDeclaration = [{
    static std::string getCanonicalOpName(llvm::StringRef name);
  }];
}

class HasCanonicalizer {
  bit hasCanonicalizer = 1;
}

class HasFolder {
  bit hasFolder = 1;
}

def APFloatType : DialectType<EltwiseDialect,
  CPred<"$_self.isa<eltwise::APFloatType>()">,"fx">;

def APSignedIntegerType : DialectType<EltwiseDialect,
  CPred<"$_self.isa<eltwise::APSignedIntegerType>()">, "six">;

def APUnsignedIntegerType : DialectType<EltwiseDialect,
  CPred<"$_self.isa<eltwise::APUnsignedIntegerType>()">, "uix">;

def AnyScalar : Type<Or<[
    Index.predicate,
    AnyFloat.predicate,
    I1.predicate,
    AnySignedInteger.predicate,
    AnyUnsignedInteger.predicate,
    APFloatType.predicate,
    APSignedIntegerType.predicate,
    APUnsignedIntegerType.predicate
  ]>, "scalar">;

// Whether a type is a RankedTensorType.
def IsRankedTensorTypePred : CPred<"$_self.isa<RankedTensorType>()">;

class RankedTensorOf<list<Type> allowedTypes> : ShapedContainerType<
    allowedTypes, CPred<"$_self.isa<RankedTensorType>()">, "ranked-tensor">;

def EltwiseAny : TypeConstraint<Or<[
    AnyScalar.predicate,
    RankedTensorOf<[AnyScalar]>.predicate
  ]>, "eltwise-any">;

def EltwiseBool : TypeConstraint<Or<[
    I1.predicate,
    RankedTensorOf<[I1]>.predicate,
  ]>, "eltwise-bool">;

def EltwiseFloat : TypeConstraint<Or<[
    APFloatType.predicate,
    AnyFloat.predicate,
    RankedTensorOf<[AnyFloat, APFloatType]>.predicate
  ]>, "eltwise-float">;

def EltwiseSigned : TypeConstraint<Or<[
    APSignedIntegerType.predicate,
    AnySignedInteger.predicate,
    RankedTensorOf<[AnySignedInteger, APSignedIntegerType]>.predicate
  ]>, "eltwise-signed">;

def EltwiseUnsigned : TypeConstraint<Or<[
    APUnsignedIntegerType.predicate,
    AnyUnsignedInteger.predicate,
    RankedTensorOf<[AnyUnsignedInteger, APUnsignedIntegerType]>.predicate
  ]>, "eltwise-unsigned">;

def EltwiseInteger : TypeConstraint<Or<[
    APSignedIntegerType.predicate,
    APUnsignedIntegerType.predicate,
    AnySignedInteger.predicate,
    AnyUnsignedInteger.predicate,
    RankedTensorOf<[
      AnySignedInteger,
      AnyUnsignedInteger,
      APSignedIntegerType,
      APUnsignedIntegerType
    ]>.predicate
  ]>, "eltwise-integer">;

def EltwiseIndex : Type<Or<[
    Index.predicate,
    SI32.predicate,
    RankedTensorOf<[Index, SI32]>.predicate
  ]>, "eltwise-index">;

#endif // __PML_ELTWISE_BASE__
