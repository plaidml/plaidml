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

def AnyScalar : Type<Or<[
    Index.predicate,
    AnyFloat.predicate,
    I1.predicate,
    AnySignedInteger.predicate,
    AnyUnsignedInteger.predicate
  ]>, "scalar">;

// Whether a type is a RankedTensorType.
def IsRankedTensorTypePred : CPred<"$_self.isa<RankedTensorType>()">;

class RankedTensorOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes, IsRankedTensorTypePred, "ranked-tensor">;

def EltwiseAny : TypeConstraint<Or<[
    AnyScalar.predicate,
    RankedTensorOf<[AnyScalar]>.predicate
  ]>, "eltwise-anyx">;

def EltwiseBool : TypeConstraint<Or<[
    I1.predicate,
    RankedTensorOf<[I1]>.predicate,
  ]>, "eltwise-bool">;

def EltwiseFloat : TypeConstraint<Or<[
    AnyFloat.predicate,
    RankedTensorOf<[AnyFloat]>.predicate
  ]>, "eltwise-float">;

def EltwiseInteger : TypeConstraint<Or<[
    AnySignedInteger.predicate,
    AnyUnsignedInteger.predicate,
    RankedTensorOf<[AnySignedInteger]>.predicate,
    RankedTensorOf<[AnyUnsignedInteger]>.predicate
  ]>, "eltwise-integer">;

def EltwiseSigned : TypeConstraint<Or<[
    AnySignedInteger.predicate,
    RankedTensorOf<[AnySignedInteger]>.predicate
  ]>, "eltwise-signed">;

def EltwiseUnsigned : TypeConstraint<Or<[
    AnyUnsignedInteger.predicate,
    RankedTensorOf<[AnyUnsignedInteger]>.predicate
  ]>, "eltwise-unsigned">;

def EltwiseIndex : Type<Or<[
    Index.predicate,
    SI32.predicate,
    RankedTensorOf<[Index]>.predicate,
    RankedTensorOf<[SI32]>.predicate
  ]>, "eltwise-index">;

#endif // __PML_ELTWISE_PREDICATES__
