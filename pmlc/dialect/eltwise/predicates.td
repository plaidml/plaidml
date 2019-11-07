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
def IsScalarPred : CPred<"$_self.isa<ScalarType>()">;

def AnyScalar : Type<IsScalarPred, "scalar">;

def ScalarBool :
    Type<And<[IsScalarPred, CPred<"$_self.cast<ScalarType>().type() == DataType::BOOLEAN">]>, "bool">;

def AnyScalarFloat :
    Type<And<[IsScalarPred, CPred<"is_float($_self.cast<ScalarType>().type())">]>, "floating-point">;

def AnyScalarInteger :
    Type<And<[IsScalarPred, CPred<"is_int($_self.cast<ScalarType>().type())">]>, "integer">;

def AnyScalarUnsigned :
    Type<And<[IsScalarPred, CPred<"is_uint($_self.cast<ScalarType>().type())">]>, "unsigned">;

// Whether a type is a RankedTensorType.
def IsRankedTensorTypePred : CPred<"$_self.isa<RankedTensorType>()">;

class RankedTensorOf<list<Type> allowedTypes> :
    ShapedContainerType<allowedTypes, IsRankedTensorTypePred, "ranked-tensor">;

def Eltwise_AnyTensor :
    TypeConstraint<Or<[
        Index.predicate,
        RankedTensorOf<[AnyScalar]>.predicate
    ]>,
    "eltwise-any">;

def Eltwise_BoolTensor : RankedTensorOf<[ScalarBool]>;
def Eltwise_FloatTensor : RankedTensorOf<[AnyScalarFloat]>;
def Eltwise_IntegerTensor : RankedTensorOf<[AnyScalarInteger]>;
def Eltwise_UnsignedTensor : RankedTensorOf<[AnyScalarUnsigned]>;

def IsScalarTensor :
    Type<And<[IsRankedTensorTypePred, CPred<"$_self.cast<ShapedType>().getRank() == 0">]>, "scalar-tensor">;

def Eltwise_Integer :
    Type<Or<[
        Index.predicate,
        And<[
          RankedTensorOf<[AnyScalarInteger]>.predicate,
          IsScalarTensor.predicate
        ]>
    ]>,
    "eltwise-integer">;

#endif // __PML_ELTWISE_PREDICATES__
