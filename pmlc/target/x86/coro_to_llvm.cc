//===- AsyncToLLVM.cpp - Convert Coro to LLVM dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/target/x86/passes.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace {
/// Async Runtime API function types.
///
/// Because we can't create API function signature for type parametrized
/// async.value type, we use opaque pointers (!llvm.ptr<i8>) instead. After
/// lowering all async data types become opaque pointers at runtime.
struct AsyncAPI {
  // All async types are lowered to opaque i8* LLVM pointers at runtime.
  static LLVM::LLVMPointerType opaquePointerType(MLIRContext *ctx) {
    return LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
  }

  static LLVM::LLVMTokenType tokenType(MLIRContext *ctx) {
    return LLVM::LLVMTokenType::get(ctx);
  }
};

//===----------------------------------------------------------------------===//
// Convert async.coro.id to @llvm.coro.id intrinsic.
//===----------------------------------------------------------------------===//

struct CoroIdOpConversion : public ConvertOpToLLVMPattern<async::CoroIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(async::CoroIdOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto token = AsyncAPI::tokenType(op->getContext());
    auto loc = op->getLoc();

    // Constants for initializing coroutine frame.
    auto constZero = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto nullPtr = rewriter.create<LLVM::NullOp>(loc, getVoidPtrType());

    // Get coroutine id: @llvm.coro.id.
    rewriter.replaceOpWithNewOp<LLVM::CoroIdOp>(
        op, token, ValueRange({constZero, nullPtr, nullPtr, nullPtr}));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert async.coro.begin to @llvm.coro.begin intrinsic.
//===----------------------------------------------------------------------===//

struct CoroBeginOpConversion
    : public ConvertOpToLLVMPattern<async::CoroBeginOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(async::CoroBeginOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Get coroutine frame size: @llvm.coro.size.i64.
    auto coroSize =
        rewriter.create<LLVM::CoroSizeOp>(loc, rewriter.getI64Type());

    // Allocate memory for the coroutine frame.
    auto allocFuncOp = LLVM::lookupOrCreateMallocFn(
        op->getParentOfType<ModuleOp>(), getIndexType());
    auto coroAlloc = LLVM::createLLVMCall(rewriter, loc, allocFuncOp,
                                          {coroSize}, getVoidPtrType());

    // Begin a coroutine: @llvm.coro.begin.
    auto coroId = async::CoroBeginOpAdaptor(operands).id();
    rewriter.replaceOpWithNewOp<LLVM::CoroBeginOp>(
        op, getVoidPtrType(), ValueRange({coroId, coroAlloc[0]}));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert async.coro.free to @llvm.coro.free intrinsic.
//===----------------------------------------------------------------------===//

struct CoroFreeOpConversion : public ConvertOpToLLVMPattern<async::CoroFreeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(async::CoroFreeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Get a pointer to the coroutine frame memory: @llvm.coro.free.
    auto coroMem =
        rewriter.create<LLVM::CoroFreeOp>(loc, getVoidPtrType(), operands);

    // Free the memory.
    auto freeFunc = LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>());

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange(), rewriter.getSymbolRefAttr(freeFunc),
        coroMem.getResult());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert async.coro.end to @llvm.coro.end intrinsic.
//===----------------------------------------------------------------------===//

struct CoroEndOpConversion : public ConvertOpToLLVMPattern<async::CoroEndOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(async::CoroEndOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // We are not in the block that is part of the unwind sequence.
    auto constFalse = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(false));

    // Mark the end of a coroutine: @llvm.coro.end.
    auto coroHdl = async::CoroEndOpAdaptor(operands).handle();
    rewriter.create<LLVM::CoroEndOp>(op->getLoc(), rewriter.getI1Type(),
                                     ValueRange({coroHdl, constFalse}));
    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert async.coro.save to @llvm.coro.save intrinsic.
//===----------------------------------------------------------------------===//

struct CoroSaveOpConversion : public ConvertOpToLLVMPattern<async::CoroSaveOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(async::CoroSaveOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Save the coroutine state: @llvm.coro.save
    rewriter.replaceOpWithNewOp<LLVM::CoroSaveOp>(
        op, AsyncAPI::tokenType(op->getContext()), operands);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Convert async.coro.suspend to @llvm.coro.suspend intrinsic.
//===----------------------------------------------------------------------===//

/// Convert async.coro.suspend to the @llvm.coro.suspend intrinsic call, and
/// branch to the appropriate block based on the return code.
///
/// Before:
///
///   ^suspended:
///     "opBefore"(...)
///     async.coro.suspend %state, ^suspend, ^resume, ^cleanup
///   ^resume:
///     "op"(...)
///   ^cleanup: ...
///   ^suspend: ...
///
/// After:
///
///   ^suspended:
///     "opBefore"(...)
///     %suspend = llmv.intr.coro.suspend ...
///     switch %suspend [-1: ^suspend, 0: ^resume, 1: ^cleanup]
///   ^resume:
///     "op"(...)
///   ^cleanup: ...
///   ^suspend: ...
///
struct CoroSuspendOpConversion
    : public ConvertOpToLLVMPattern<async::CoroSuspendOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(async::CoroSuspendOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto i8 = rewriter.getIntegerType(8);
    auto i32 = rewriter.getI32Type();
    auto loc = op->getLoc();

    // This is not a final suspension point.
    auto constFalse = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));

    // Suspend a coroutine: @llvm.coro.suspend
    auto coroState = async::CoroSuspendOpAdaptor(operands).state();
    auto coroSuspend = rewriter.create<LLVM::CoroSuspendOp>(
        loc, i8, ValueRange({coroState, constFalse}));

    // Cast return code to i32.

    // After a suspension point decide if we should branch into resume, cleanup
    // or suspend block of the coroutine (see @llvm.coro.suspend return code
    // documentation).
    llvm::SmallVector<int32_t, 2> caseValues = {0, 1};
    llvm::SmallVector<Block *, 2> caseDest = {op.resumeDest(),
                                              op.cleanupDest()};
    rewriter.replaceOpWithNewOp<LLVM::SwitchOp>(
        op, rewriter.create<LLVM::SExtOp>(loc, i32, coroSuspend.getResult()),
        /*defaultDestination=*/op.suspendDest(),
        /*defaultOperands=*/ValueRange(),
        /*caseValues=*/caseValues,
        /*caseDestinations=*/caseDest,
        /*caseOperands=*/ArrayRef<ValueRange>(),
        /*branchWeights=*/ArrayRef<int32_t>());

    return success();
  }
};

} // namespace

void populateCoroToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns) {
  converter.addConversion([](Type type) -> Optional<Type> {
    if (type.isa<async::CoroIdType, async::CoroStateType>())
      return AsyncAPI::tokenType(type.getContext());
    if (type.isa<async::CoroHandleType>())
      return AsyncAPI::opaquePointerType(type.getContext());

    return llvm::None;
  });

  patterns.add<CoroBeginOpConversion, //
               CoroEndOpConversion,   //
               CoroFreeOpConversion,  //
               CoroIdOpConversion,    //
               CoroSaveOpConversion,  //
               CoroSuspendOpConversion>(converter);
}

} // namespace pmlc::target::x86
