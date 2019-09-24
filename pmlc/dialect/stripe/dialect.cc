// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/dialect.h"

#include "mlir/IR/Dialect.h"

#include "pmlc/dialect/stripe/ops.h"

namespace pmlc {
namespace dialect {
namespace stripe {

static mlir::DialectRegistration<Dialect> StripeOps;

Dialect::Dialect(mlir::MLIRContext* ctx) : mlir::Dialect(getDialectNamespace(), ctx) {
  addTypes<            //
      AffineType,      //
      DeviceIDType,    //
      DevicePathType,  //
      PrngType,        //
      TensorType,      //
      TensorRefType>();
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/stripe/ops.cpp.inc"
      >();
}

mlir::Type Dialect::parseType(llvm::StringRef tyData, mlir::Location loc) const {
  throw std::runtime_error("Unimplemented");
}

static void print(AffineType type, llvm::raw_ostream& os) { os << "affine"; }

static void print(TensorType type, llvm::raw_ostream& os) {
  os << "tensor: ";
  os << type.getElementType() << "(";
  auto shape = type.getShape();
  for (int64_t i = 0; i < type.getRank(); i++) {
    const auto& dim = shape[i];
    if (i) {
      os << ", ";
    }
    if (!dim.size) {
      os << "?";
    } else {
      os << std::to_string(dim.size);
    }
    if (dim.stride) {
      os << ": " << std::to_string(dim.stride);
    }
  }
  os << ")";
}

static void print(TensorRefType type, llvm::raw_ostream& os) {
  os << "tensor_ref " << type.getElementType() << ":" << std::to_string(type.getRank());
}

static void print(PrngType type, llvm::raw_ostream& os) { os << "prng"; }

static void print(DeviceIDType type, llvm::raw_ostream& os) { os << "device_id"; }

static void print(DevicePathType type, llvm::raw_ostream& os) { os << "device_path"; }

void Dialect::printType(mlir::Type type, llvm::raw_ostream& os) const {
  if (auto affineType = type.dyn_cast<AffineType>()) {
    print(affineType, os);
  } else if (auto deviceIDType = type.dyn_cast<DeviceIDType>()) {
    print(deviceIDType, os);
  } else if (auto devicePathType = type.dyn_cast<DevicePathType>()) {
    print(devicePathType, os);
  } else if (auto prngType = type.dyn_cast<PrngType>()) {
    print(prngType, os);
  } else if (auto tensorType = type.dyn_cast<TensorType>()) {
    print(tensorType, os);
  } else if (auto tensorRefType = type.dyn_cast<TensorRefType>()) {
    print(tensorRefType, os);
  } else {
    llvm_unreachable("unhandled Plaid type");
  }
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
