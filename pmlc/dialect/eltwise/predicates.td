#ifndef __PML_ELTWISE_PREDICATES__
#define __PML_ELTWISE_PREDICATES__

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif

class HasCanonicalizer {
  bit hasCanonicalizer = 1;
}

class HasFolder {
  bit hasFolder = 1;
}

// Whether a type is a ScalarType.
def IsScalarPred : And<[
    CPred<"$_self.isa<ScalarType>()">,
    CPred<"$_self.cast<ScalarType>().type() != DataType::INVALID">
  ]>;

def AnyScalar : Type<Or<[
    Index.predicate,
    IsScalarPred,
  ]>, "scalar">;

def ScalarBool : Type<And<[
    IsScalarPred,
    CPred<"$_self.cast<ScalarType>().type() == DataType::BOOLEAN">
  ]>, "scalar-bool">;

def ScalarIndex : Type<And<[
    IsScalarPred,
    CPred<"$_self.cast<ScalarType>().type() == DataType::INTX">
  ]>, "scalar-index">;

def AnyScalarFloat : Type<And<[
    IsScalarPred,
    CPred<"is_float($_self.cast<ScalarType>().type())">
  ]>, "scalar-float">;

def AnyScalarInteger : Type<And<[
    IsScalarPred,
    CPred<"is_int($_self.cast<ScalarType>().type())">
  ]>, "scalar-integer">;

def AnyScalarUnsigned : Type<And<[
    IsScalarPred,
    CPred<"is_uint($_self.cast<ScalarType>().type())">
  ]>, "scalar-unsigned">;

def IndexLike : Type<Or<[
    Index.predicate,
    ScalarIndex.predicate,
  ]>, "index-like">;

// Whether a type is a RankedTensorType.
def IsRankedTensorTypePred : CPred<"$_self.isa<RankedTensorType>()">;

class RankedTensorOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes, IsRankedTensorTypePred, "ranked-tensor">;

def EltwiseAny : TypeConstraint<Or<[
    AnyScalar.predicate,
    RankedTensorOf<[AnyScalar]>.predicate
  ]>, "eltwise-any">;

def EltwiseBool : TypeConstraint<Or<[
    ScalarBool.predicate,
    RankedTensorOf<[ScalarBool]>.predicate,
  ]>, "eltwise-bool">;

def EltwiseFloat : TypeConstraint<Or<[
    AnyScalarFloat.predicate,
    RankedTensorOf<[AnyScalarFloat]>.predicate
  ]>, "eltwise-float">;

def EltwiseInteger : TypeConstraint<Or<[
    AnyScalarInteger.predicate,
    RankedTensorOf<[AnyScalarInteger]>.predicate
  ]>, "eltwise-integer">;

def EltwiseUnsigned : TypeConstraint<Or<[
    AnyScalarUnsigned.predicate,
    RankedTensorOf<[AnyScalarUnsigned]>.predicate
  ]>, "eltwise-unsigned">;

def EltwiseIndex : Type<Or<[
    ScalarIndex.predicate,
    RankedTensorOf<[ScalarIndex]>.predicate
  ]>, "eltwise-index">;

#endif // __PML_ELTWISE_PREDICATES__
