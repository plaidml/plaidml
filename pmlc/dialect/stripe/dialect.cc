// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/dialect.h"

#include <utility>

#include "mlir/IR/Dialect.h"
#include "mlir/Parser.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"

#include "pmlc/dialect/stripe/ops.h"

namespace pmlc {
namespace dialect {
namespace stripe {

static mlir::DialectRegistration<Dialect> StripeOps;

Dialect::Dialect(mlir::MLIRContext* ctx) : mlir::Dialect(getDialectNamespace(), ctx) {
  addTypes<          //
      AffineType,    //
      ExecutorType,  //
      TensorType,    //
      TensorRefType>();
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/stripe/ops.cpp.inc"
      >();
}

mlir::Type Dialect::parseTensor(llvm::StringRef tyData, mlir::Location loc) const {
  static llvm::Regex re{R"(([[:alnum:]_]+)\[([[:digit:]]+):([[:digit:]]+)\])"};
  bool is_const = tyData.consume_back("const");
  StringRef typeSpec, sizeSpec;
  std::tie(typeSpec, sizeSpec) = tyData.trim().rsplit('(');
  auto t = mlir::parseType(typeSpec.trim(), getContext());
  if (!t) {
    emitError(loc, "invalid type specification: '") << typeSpec << "'";
    return Type();
  }
  if (!sizeSpec.consume_back(")")) {
    emitError(loc, "invalid tensor type, no ()'s on size spec");
    return Type();
  }
  auto dims = llvm::SmallVector<StringRef, 8>();
  auto odims = llvm::SmallVector<TensorDim, 8>();
  auto matches = llvm::SmallVector<StringRef, 4>();
  sizeSpec.split(dims, ",");
  for (auto dim : dims) {
    if (!re.match(dim, &matches)) {
      emitError(loc, "invalid tensor dimension '") << dim << "'";
      return Type();
    }
    auto odim = TensorDim{0, 0, mlir::Identifier::get(matches[1], getContext())};
    matches[2].getAsInteger(10, odim.size);
    matches[3].getAsInteger(10, odim.stride);
    odims.emplace_back(std::move(odim));
  }
  return TensorType::get(t, odims, OffsetsMap(), is_const);
}

mlir::Type Dialect::parseTensorRef(llvm::StringRef tyData, mlir::Location loc) const {
  bool is_const = tyData.consume_back("const");
  StringRef typeSpec, ndimSpec;
  std::tie(typeSpec, ndimSpec) = tyData.rsplit(':');
  auto t = mlir::parseType(typeSpec.trim(), getContext());
  if (!t) {
    emitError(loc, "invalid type specification: '") << typeSpec << "'";
    return Type();
  }
  size_t ndims;
  if (ndimSpec.trim().consumeInteger(0, ndims)) {
    emitError(loc, "invalid ndims'") << ndims << "'";
    return Type();
  }
  return TensorRefType::get(t, ndims, is_const);
}

std::string Dialect::getDialectAttrName(llvm::StringRef name) {
  return llvm::formatv("{0}.{1}", stripe::Dialect::getDialectNamespace(), name).str();
}

mlir::Type Dialect::parseType(llvm::StringRef tyData, mlir::Location loc) const {
  if (tyData == "affine") {
    return AffineType::get(getContext());
  } else if (tyData == "executor") {
    return ExecutorType::get(getContext());
  } else if (tyData.consume_front("tensor ")) {
    return parseTensor(tyData, loc);
  } else if (tyData.consume_front("tensor_ref ")) {
    return parseTensorRef(tyData, loc);
  } else {
    emitError(loc, "unknown stripe type: '" + tyData + "'");
    return Type();
  }
}

static void print(AffineType type, llvm::raw_ostream& os) { os << "affine"; }

static void print(ExecutorType type, llvm::raw_ostream& os) { os << "executor"; }

static void print(TensorType type, llvm::raw_ostream& os) {
  os << "tensor ";
  os << type.getElementType() << "(";
  auto shape = type.getShape();
  for (int64_t i = 0; i < type.getRank(); i++) {
    const auto& dim = shape[i];
    if (i) {
      os << ", ";
    }
    os << dim.cls << '[' << dim.size << ":" << dim.stride << ']';
  }
  os << ")";
  if (type.is_const()) {
    os << " const";
  }
}

static void print(TensorRefType type, llvm::raw_ostream& os) {
  os << "tensor_ref " << type.getElementType() << ":" << std::to_string(type.getRank());
  if (type.is_const()) {
    os << " const";
  }
}

void Dialect::printType(mlir::Type type, llvm::raw_ostream& os) const {
  if (auto affineType = type.dyn_cast<AffineType>()) {
    print(affineType, os);
  } else if (auto executorType = type.dyn_cast<ExecutorType>()) {
    print(executorType, os);
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
