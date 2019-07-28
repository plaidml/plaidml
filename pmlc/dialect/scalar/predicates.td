#ifndef __PML_SCALAR_PREDICATES__
#define __PML_SCALAR_PREDICATES__

include "mlir/IR/OpBase.td"

// Whether a type is a ScalarType.
def IsScalarPred : CPred<"$_self.isa<ScalarType>()">;

def ScalarType : Type<IsScalarPred, "scalar">;

def BooleanType : 
    Type<And<[IsScalarPred, CPred<"$_self.cast<ScalarType>().type() == DataType::BOOLEAN">]>, "boolean">;

def ScalarIntegerLike : 
    TypeConstraint<And<[IsScalarPred, CPred<"is_int($_self.cast<ScalarType>().type())">]>, "integer">;

def ScalarFloatLike : 
    TypeConstraint<And<[IsScalarPred, CPred<"is_float($_self.cast<ScalarType>().type())">]>, "floating-point">;

class HasFolder {
  bit hasFolder = 1;
}

#endif // __PML_SCALAR_PREDICATES__
