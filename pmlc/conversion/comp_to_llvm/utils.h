// Copyright 2020, Intel Corporation
#pragma once

#include <map>
#include <string>
#include <type_traits>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/comp_to_llvm/pass_detail.h"
#include "pmlc/dialect/comp/ir/dialect.h"

namespace mlir {
class OpBuilder;
class Location;
class Value;
} // namespace mlir

namespace pmlc::conversion::comp_to_llvm {
/// Creates specified string as llvm global constant.
mlir::LLVM::GlobalOp addGlobalString(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::StringRef symbol,
                                     mlir::StringRef string);

/// Creates operations to extract pointer to global string.
mlir::Value getPtrToGlobalString(mlir::OpBuilder &builder, mlir::Location &loc,
                                 mlir::LLVM::GlobalOp globalOp);

/// Creates operations to extract pointer to binary and binary size in bytes
/// from BinaryKernelInfo. After returning `pointer` will contain
/// pointer to binary of type !llvm.ptr<i8> and `bytes` will contain
/// number of bytes in binary of type !llvm.i32.
void getPtrToBinaryModule(mlir::OpBuilder &builder, mlir::Location &loc,
                          const BinaryModuleInfo &binaryInfo,
                          mlir::Value &pointer, mlir::Value &bytes);

/// Helper functions to check whether operation belongs to specified runtime
/// and should be handled by lowering for that runtime.
template <class Op>
typename std::enable_if<
    Op::template hasTrait<pmlc::dialect::comp::ExecEnvOpInterface::Trait>(),
    bool>::type
isMatchingRuntimeOperation(Op op, pmlc::dialect::comp::ExecEnvRuntime runtime) {
  mlir::Operation *operation = op.getOperation();
  auto interface =
      mlir::cast<pmlc::dialect::comp::ExecEnvOpInterface>(operation);
  auto execEnvType =
      interface.getExecEnv().getType().cast<pmlc::dialect::comp::ExecEnvType>();
  return execEnvType.getRuntime() == runtime;
}

inline bool
isMatchingRuntimeOperation(pmlc::dialect::comp::CreateExecEnv op,
                           pmlc::dialect::comp::ExecEnvRuntime runtime) {
  auto execEnvType =
      op.getResult().getType().cast<pmlc::dialect::comp::ExecEnvType>();
  return execEnvType.getRuntime() == runtime;
}

inline bool
isMatchingRuntimeOperation(pmlc::dialect::comp::DestroyExecEnv op,
                           pmlc::dialect::comp::ExecEnvRuntime runtime) {
  auto execEnvType =
      op.execEnv().getType().cast<pmlc::dialect::comp::ExecEnvType>();
  return execEnvType.getRuntime() == runtime;
}

inline bool
isMatchingRuntimeOperation(pmlc::dialect::comp::Wait op,
                           pmlc::dialect::comp::ExecEnvRuntime runtime) {
  if (op.events().empty())
    return false;
  mlir::Value firstEvent = op.events().front();
  auto eventType = firstEvent.getType().cast<pmlc::dialect::comp::EventType>();
  return eventType.getRuntime() == runtime;
}

/// Helper base class for comp -> llvm conversion patterns.
/// Forces creation with type converter and provides some utility functions.
template <class Op>
struct ConvertCompOpBasePattern : mlir::OpConversionPattern<Op> {
  ConvertCompOpBasePattern(pmlc::dialect::comp::ExecEnvRuntime runtime,
                           mlir::TypeConverter &typeConverter,
                           mlir::MLIRContext *context)
      : mlir::OpConversionPattern<Op>(typeConverter, context),
        runtime(runtime) {}

  /// Returns type conversion result for `type`.
  mlir::Type convertType(mlir::Type type) const {
    return this->getTypeConverter()->convertType(type);
  }
  /// Materializes default conversion for specified value.
  mlir::Value materializeConversion(mlir::OpBuilder &builder,
                                    mlir::Location loc,
                                    mlir::Value value) const {
    return this->getTypeConverter()->materializeTargetConversion(
        builder, loc, convertType(value.getType()), value);
  }
  /// Returns true if operation has runtime that is handled by this pattern.
  /// This should be used when matching pattern to provide interoperability
  /// between multiple runtimes in one module.
  bool isMatchingRuntime(Op op) const {
    return isMatchingRuntimeOperation(op, runtime);
  }

  pmlc::dialect::comp::ExecEnvRuntime runtime;
};

} // namespace pmlc::conversion::comp_to_llvm
