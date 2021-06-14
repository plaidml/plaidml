// Copyright 2019, Intel Corporation

#include "pmlc/util/util.h"

// #include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Process.h"

using namespace mlir; // NOLINT

namespace pmlc::util {

static constexpr StringLiteral kTagAttribute = "tags";

uint64_t getByteSize(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t, 8> strides;
  if (failed(getStridesAndOffset(type, strides, offset))) {
    throw std::runtime_error("Could not retrieve strides");
  }
  auto sizes = type.getShape();
  uint64_t total = 0;
  for (unsigned i = 0; i < type.getRank(); i++) {
    if (!sizes[i]) {
      return 0;
    }
    if (strides[i] > 0) {
      total += (sizes[i] - 1) * strides[i];
    }
  }
  unsigned elem_bytes = llvm::divideCeil(type.getElementTypeBitWidth(), 8);
  return (total + 1) * elem_bytes;
}

DiagnosticCounter::DiagnosticCounter() : counter(0), threshold(0) {
  auto env = llvm::sys::Process::GetEnv("PLAIDML_COUNTER");
  if (env) {
    threshold = std::atoi(env->c_str());
  }
}

DiagnosticCounter::Result DiagnosticCounter::next() {
  if (!threshold) {
    return Result::Continue;
  }
  ++counter;
  if (counter < threshold) {
    return Result::Continue;
  }
  if (counter > threshold) {
    return Result::Break;
  }
  return Result::Match;
}

void wrapFunctionAndPackArguments(llvm::Module *module, StringRef funcName,
                                  StringRef newName) {
  auto &ctx = module->getContext();
  llvm::IRBuilder<> builder(ctx);
  auto *func = module->getFunction(funcName);
  if (!func) {
    throw std::runtime_error("Could not find function: " + funcName.str());
  }

  // Given a function `foo(<...>) -> T`, define the interface function
  // `mlir_foo(i8**) -> T`.
  auto newType = llvm::FunctionType::get(func->getReturnType(),
                                         builder.getInt8PtrTy()->getPointerTo(),
                                         /*isVarArg=*/false);
  auto funcCst = module->getOrInsertFunction(newName, newType);
  llvm::Function *interfaceFunc = cast<llvm::Function>(funcCst.getCallee());

  // Extract the arguments from the type-erased argument list and cast them to
  // the proper types.
  auto bb = llvm::BasicBlock::Create(ctx);
  bb->insertInto(interfaceFunc);
  builder.SetInsertPoint(bb);
  llvm::Value *argList = interfaceFunc->arg_begin();
  SmallVector<llvm::Value *, 8> args;
  args.reserve(llvm::size(func->args()));
  for (auto &indexedArg : llvm::enumerate(func->args())) {
    llvm::Value *argIndex = llvm::Constant::getIntegerValue(
        builder.getInt64Ty(), APInt(64, indexedArg.index()));
    llvm::Value *argPtrPtr = builder.CreateGEP(argList, argIndex);
    llvm::Value *argPtr = builder.CreateLoad(builder.getInt8PtrTy(), argPtrPtr);
    llvm::Type *dstType = indexedArg.value().getType();
    llvm::Value *arg = dstType->isIntegerTy()
                           ? builder.CreatePtrToInt(argPtr, dstType)
                           : builder.CreateBitCast(argPtr, dstType);
    args.push_back(arg);
  }

  // Call the implementation function with the extracted arguments + return
  llvm::Value *val = builder.CreateCall(func, args);
  if (func->getReturnType()->isVoidTy()) {
    builder.CreateRetVoid();
  } else {
    builder.CreateRet(val);
  }
}

AffineValueMap getRangesValueMap(AffineParallelOp op) {
  AffineValueMap out;
  AffineValueMap::difference(op.getUpperBoundsValueMap(),
                             op.getLowerBoundsValueMap(), &out);
  return out;
}

void splitAffineMaps(AffineMap from, SmallVectorImpl<AffineMap> &into) {
  for (AffineExpr expr : from.getResults()) {
    into.push_back(
        AffineMap::get(from.getNumDims(), from.getNumSymbols(), expr));
  }
}

} // namespace pmlc::util
