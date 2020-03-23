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

// Type constraint for integer-like types: integers, indices,
// vectors of integers, tensors of integers.
// def IntegerLike : TypeConstraint<Or<[
//         Index.predicate,
//         AnySignedInteger.predicate,
//         VectorOf<[AnySignedInteger]>.predicate,
//         TensorOf<[AnySignedInteger]>.predicate,
//         AnyUnsignedInteger.predicate,
//         VectorOf<[AnyUnsignedInteger]>.predicate,
//         TensorOf<[AnyUnsignedInteger]>.predicate]>,
//     "integer-like">;

// // Type constraint for integer-like or float-like types.
// def IntegerOrFloatLike : TypeConstraint<Or<[
//         IntegerLike.predicate,
//         FloatLike.predicate]>,
//     "integer-like or floating-point-like">;

// Whether a type is a ScalarType.
// def IsScalarPred : And<[
//     CPred<"$_self.isa<ScalarType>()">,
//     CPred<"$_self.cast<ScalarType>().type() != DataType::invalid">
//   ]>;

def AnyScalar : Type<Or<[
    Index.predicate,
    AnyFloat.predicate,
    I1.predicate,
    AnySignedInteger.predicate,
    AnyUnsignedInteger.predicate
  ]>, "scalar">;

// def ScalarBool : Type<And<[
//     IsScalarPred,
//     CPred<"$_self.cast<ScalarType>().type() == DataType::u1">
//   ]>, "scalar-bool">;

// def ScalarIndex : Type<And<[
//     IsScalarPred,
//     CPred<"$_self.cast<ScalarType>().type() == DataType::i32">
//   ]>, "scalar-index">;

// def ScalarUINT32 : Type<And<[
//     IsScalarPred,
//     CPred<"$_self.cast<ScalarType>().type() == DataType::u32">
//   ]>, "scalar-u32">;

// def AnyScalarFloat : Type<And<[
//     IsScalarPred,
//     CPred<"isFloat($_self.cast<ScalarType>().type())">
//   ]>, "scalar-float">;

// def AnyScalarSigned : Type<And<[
//     IsScalarPred,
//     CPred<"isSigned($_self.cast<ScalarType>().type())">
//   ]>, "scalar-signed">;

// def AnyScalarUnsigned : Type<And<[
//     IsScalarPred,
//     CPred<"isUnsigned($_self.cast<ScalarType>().type())">
//   ]>, "scalar-unsigned">;

// def AnyScalarInteger : Type<And<[
//     IsScalarPred,
//     CPred<"isInteger($_self.cast<ScalarType>().type())">
//   ]>, "scalar-integer">;

// def IndexLike : Type<Or<[
//     Index.predicate,
//     ScalarIndex.predicate,
//   ]>, "index-like">;

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
