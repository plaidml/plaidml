// Copyright 2019, Intel Corporation

#include "mlir/IR/Dialect.h"
#include "pmlc/dialect/stripe/ops.h"

namespace pmlc {
namespace dialect {
namespace stripe {

class Dialect : public mlir::Dialect {
 public:
  explicit Dialect(mlir::MLIRContext* ctx);
  mlir::Type parseType(llvm::StringRef tyData, mlir::Location loc) const override;
  void printType(mlir::Type type, llvm::raw_ostream& os) const override;
  void printAttribute(mlir::Attribute attr, llvm::raw_ostream& os) const override;
};

static mlir::DialectRegistration<Dialect> StripeOps;

Dialect::Dialect(mlir::MLIRContext* ctx) : mlir::Dialect("stripe", ctx) {
  addTypes<AffineType, DeviceIDType, DevicePathType, PrngType, TensorType>();
  addAttributes<TensorLayoutAttr>();
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/stripe/ops.cpp.inc"
      >();
}

mlir::Type Dialect::parseType(llvm::StringRef tyData, mlir::Location loc) const {
  throw std::runtime_error("Unimplemented");
}

static void print(AffineType t, llvm::raw_ostream& os) { os << "affine"; }

static void print(TensorType t, llvm::raw_ostream& os) {
  os << "tensor ";
  os << t.base() << ":" << std::to_string(t.ndim());
}

static void print(PrngType t, llvm::raw_ostream& os) { os << "prng"; }

static void print(DeviceIDType t, llvm::raw_ostream& os) { os << "device_id"; }

static void print(DevicePathType t, llvm::raw_ostream& os) { os << "device_path"; }

void Dialect::printType(mlir::Type type, llvm::raw_ostream& os) const {
  if (auto t = type.dyn_cast<AffineType>()) {
    print(t, os);
  } else if (auto t = type.dyn_cast<DeviceIDType>()) {
    print(t, os);
  } else if (auto t = type.dyn_cast<DevicePathType>()) {
    print(t, os);
  } else if (auto t = type.dyn_cast<PrngType>()) {
    print(t, os);
  } else if (auto t = type.dyn_cast<TensorType>()) {
    print(t, os);
  } else {
    llvm_unreachable("unhandled Plaid type");
  }
}

static void print(TensorLayoutAttr a, llvm::raw_ostream& os) {
  os << "tensor ";
  os << a.base() << "(";
  bool first = true;
  for (const auto& dim : a.dims()) {
    if (!first) {
      os << ", ";
    }
    first = false;
    if (dim.unit != "") {
      os << dim.unit << ":";
    }
    if (dim.size == 0) {
      os << "?";
    } else {
      os << std::to_string(dim.size);
    }
    if (dim.stride != 0) {
      os << ":" << std::to_string(dim.stride);
    }
  }
  os << ")";
}

void Dialect::printAttribute(Attribute attr, llvm::raw_ostream& os) const {
  if (auto a = attr.dyn_cast<TensorLayoutAttr>()) {
    print(a, os);
  } else {
    llvm_unreachable("unhandled Plaid Attribute");
  }
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
