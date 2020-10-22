// Copyright 2020, Intel Corporation

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/conversion/abi_to_llvm/passes.h"
#include "pmlc/dialect/abi/ir/dialect.h"
#include "pmlc/util/ids.h"

// #pragma GCC diagnostic ignored "-Wunused-function"
// #pragma GCC diagnostic ignored "-Wunused-variable"

namespace abi = pmlc::dialect::abi;
namespace LLVM = mlir::LLVM;
using LLVMType = LLVM::LLVMType;

namespace pmlc::conversion::abi_to_llvm {

static LLVM::LLVMFuncOp importFunc(mlir::StringRef name, mlir::Operation *op,
                                   LLVMType funcTy, mlir::OpBuilder &builder) {
  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
  auto func = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (!func) {
    mlir::OpBuilder::InsertionGuard insertionGuard{builder};
    builder.setInsertionPointToStart(moduleOp.getBody());
    func =
        builder.create<LLVM::LLVMFuncOp>(builder.getUnknownLoc(), name, funcTy);
  }
  return func;
}

namespace {

struct CreateNetworkOpLowering final
    : public mlir::ConvertOpToLLVMPattern<abi::CreateNetworkOp> {
  using mlir::ConvertOpToLLVMPattern<
      abi::CreateNetworkOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto createNetworkOp = mlir::cast<abi::CreateNetworkOp>(op);

    auto mallocFunc = importFunc(
        "malloc", createNetworkOp,
        LLVMType::getFunctionTy(getVoidPtrType(),
                                mlir::ArrayRef<LLVMType>{getIndexType()},
                                /*isVarArg=*/false),
        rewriter);

    rewriter.startRootUpdate(createNetworkOp);
    rewriter.setInsertionPoint(createNetworkOp);

    // Figure out the type of the network structure.
    mlir::SmallVector<LLVMType, 8> networkFieldTypes;
    mlir::SmallVector<mlir::Value, 8> networkFieldValues;
    for (auto srcValue : createNetworkOp.getOperands()) {
      auto convValue = typeConverter.materializeTargetConversion(
          rewriter, createNetworkOp.getLoc(),
          typeConverter.convertType(srcValue.getType()), srcValue);
      if (!convValue) {
        rewriter.cancelRootUpdate(createNetworkOp);
        return mlir::failure();
      }
      networkFieldTypes.emplace_back(convValue.getType().cast<LLVMType>());
      networkFieldValues.emplace_back(convValue);
    }

    auto networkTy =
        LLVMType::getStructTy(rewriter.getContext(), networkFieldTypes);

    // Fill in a local stack copy of the network structure.
    auto initNetworkValue =
        rewriter.create<LLVM::UndefOp>(rewriter.getUnknownLoc(), networkTy);
    for (unsigned idx = 0; idx < networkFieldValues.size(); ++idx) {
      rewriter.create<LLVM::InsertValueOp>(
          rewriter.getUnknownLoc(), initNetworkValue, networkFieldValues[idx],
          rewriter.getI64ArrayAttr(idx));
    }

    // malloc the structure instance, and store the stack local into it.
    auto initNullNetworkValue = rewriter.create<LLVM::NullOp>(
        rewriter.getUnknownLoc(), networkTy.getPointerTo());
    auto initConstOne =
        createIndexConstant(rewriter, rewriter.getUnknownLoc(), 1);
    auto initNextNetworkValue = rewriter.create<LLVM::GEPOp>(
        rewriter.getUnknownLoc(), networkTy.getPointerTo(),
        initNullNetworkValue, mlir::ValueRange{initConstOne});
    auto initNetworkSize = rewriter.create<LLVM::PtrToIntOp>(
        rewriter.getUnknownLoc(), getIndexType(), initNextNetworkValue);
    auto initNetworkRawPtr =
        rewriter
            .create<LLVM::CallOp>(rewriter.getUnknownLoc(), mallocFunc,
                                  mlir::ValueRange{initNetworkSize})
            .getResult(0);
    auto initNetworkPtr = rewriter.create<LLVM::BitcastOp>(
        rewriter.getUnknownLoc(), networkTy.getPointerTo(), initNetworkRawPtr);
    rewriter.create<LLVM::StoreOp>(rewriter.getUnknownLoc(), initNetworkValue,
                                   initNetworkPtr);

    // Add the new LLVM terminator, replacing the abi.create_network op.
    rewriter.create<LLVM::ReturnOp>(rewriter.getUnknownLoc(),
                                    mlir::ValueRange{initNetworkPtr});
    rewriter.eraseOp(createNetworkOp);

    rewriter.finalizeRootUpdate(createNetworkOp);

    return mlir::success();
  }
};

struct DoneOpLowering final : public mlir::ConvertOpToLLVMPattern<abi::DoneOp> {
  using mlir::ConvertOpToLLVMPattern<abi::DoneOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto doneOp = mlir::cast<abi::DoneOp>(op);
    rewriter.updateRootInPlace(doneOp, [&] {
      rewriter.setInsertionPoint(doneOp);
      rewriter.create<LLVM::ReturnOp>(doneOp.getLoc(), mlir::None);
      rewriter.eraseOp(doneOp);
    });
    return mlir::success();
  }
};

struct LoopOpLowering final : public mlir::ConvertOpToLLVMPattern<abi::LoopOp> {
  using mlir::ConvertOpToLLVMPattern<abi::LoopOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loopOp = mlir::cast<abi::LoopOp>(op);

    rewriter.startRootUpdate(loopOp);

    mlir::SmallVector<LLVMType, 8> networkFieldTypes;
    for (auto ty : loopOp.getNetworkFieldTypes()) {
      networkFieldTypes.emplace_back(
          typeConverter.convertType(ty).cast<LLVMType>());
    }

    auto networkTy =
        LLVMType::getStructTy(rewriter.getContext(), networkFieldTypes);

    rewriter.setInsertionPoint(loopOp);

    if (mlir::failed( //
            buildInit(loopOp, networkTy, rewriter)) ||
        mlir::failed( //
            buildExec(loopOp, networkTy, networkFieldTypes, rewriter)) ||
        mlir::failed( //
            buildFini(loopOp, networkTy, rewriter))) {
      rewriter.cancelRootUpdate(loopOp);
      return mlir::failure();
    }

    rewriter.eraseOp(loopOp);
    rewriter.finalizeRootUpdate(loopOp);

    return mlir::success();
  }

private:
  mlir::LogicalResult
  buildInit(abi::LoopOp loopOp, LLVMType networkTy,
            mlir::ConversionPatternRewriter &rewriter) const {
    // Create plaidml_init().

    mlir::OpBuilder::InsertionGuard insertionGuard{rewriter};

    auto initArgTypes = loopOp.initRegion().getArgumentTypes();
    auto initFuncType = rewriter.getFunctionType(
        initArgTypes, mlir::TypeRange{networkTy.getPointerTo()});
    mlir::TypeConverter::SignatureConversion initSigConversion{
        loopOp.initRegion().getNumArguments()};
    auto initLLVMType = typeConverter.convertFunctionSignature(
        initFuncType,
        /*isVariadic=*/false, initSigConversion);
    auto initFunc = rewriter.create<LLVM::LLVMFuncOp>(
        loopOp.getLoc(), pmlc::util::kPlaidmlInit, initLLVMType);

    rewriter.cloneRegionBefore(loopOp.initRegion(), initFunc.getBody(),
                               initFunc.getBody().end());

    return rewriter.convertRegionTypes(&initFunc.getBody(), typeConverter,
                                       &initSigConversion);
  }

  mlir::LogicalResult
  buildExec(abi::LoopOp loopOp, LLVMType networkTy,
            const mlir::SmallVectorImpl<LLVMType> &networkFieldTypes,
            mlir::ConversionPatternRewriter &rewriter) const {
    // Create plaidml_exec().
    //
    // The inputs to the exec function are a pointer toa network object, and a
    // pointer to a struct with per-iteration parameters; each struct field is a
    // pointer to the LLVM type of the corresponding entry block argument
    // (the arguments after the arguments passed via the network structure).

    mlir::OpBuilder::InsertionGuard insertionGuard{rewriter};

    mlir::SmallVector<LLVMType, 8> iterationTypes;
    mlir::SmallVector<LLVMType, 8> iterationFieldTypes;
    for (unsigned idx = loopOp.getNumNetworkFields();
         idx < loopOp.bodyRegion().getNumArguments(); ++idx) {
      auto ty = typeConverter
                    .convertType(loopOp.bodyRegion().getArgument(idx).getType())
                    .cast<LLVMType>();
      iterationTypes.emplace_back(ty);
      iterationFieldTypes.emplace_back(ty.getPointerTo());
    }
    auto iterationTy =
        LLVMType::getStructTy(rewriter.getContext(), iterationFieldTypes);

    auto execFunc = rewriter.create<LLVM::LLVMFuncOp>(
        loopOp.getLoc(), pmlc::util::kPlaidmlExec,
        LLVMType::getFunctionTy(
            getVoidType(),
            mlir::ArrayRef<LLVMType>{networkTy.getPointerTo(),
                                     iterationTy.getPointerTo()},
            /*isVarArg=*/false));

    // Create an entry block, and materialize the arguments for the loop body.
    auto *execEntryBlock = rewriter.createBlock(
        &execFunc.getBody(), execFunc.getBody().end(),
        mlir::TypeRange{networkTy.getPointerTo(), iterationTy.getPointerTo()});
    rewriter.setInsertionPointToStart(execEntryBlock);
    mlir::BlockAndValueMapping mapping;
    unsigned bodyArgIdx = 0;

    // Materialize the network structure arguments.
    auto networkValue = rewriter.create<LLVM::LoadOp>(
        rewriter.getUnknownLoc(), execEntryBlock->getArgument(0));
    for (auto fieldIdx = 0; fieldIdx < networkFieldTypes.size(); ++fieldIdx) {
      auto fieldVal = rewriter.create<LLVM::ExtractValueOp>(
          rewriter.getUnknownLoc(), networkFieldTypes[fieldIdx], networkValue,
          rewriter.getI64ArrayAttr(fieldIdx));
      auto bodyArg = loopOp.bodyRegion().getArgument(bodyArgIdx++);
      auto argVal = typeConverter.materializeSourceConversion(
          rewriter, rewriter.getUnknownLoc(), bodyArg.getType(),
          mlir::ValueRange{fieldVal});
      mapping.map(bodyArg, argVal);
    }

    // Materialize the iteration arguments.
    auto iterationValue = rewriter.create<LLVM::LoadOp>(
        rewriter.getUnknownLoc(), execEntryBlock->getArgument(1));
    for (auto fieldIdx = 0; fieldIdx < iterationFieldTypes.size(); ++fieldIdx) {
      auto ptrVal = rewriter.create<LLVM::ExtractValueOp>(
          rewriter.getUnknownLoc(), iterationFieldTypes[fieldIdx],
          iterationValue, rewriter.getI64ArrayAttr(fieldIdx));
      auto fieldVal =
          rewriter.create<LLVM::LoadOp>(rewriter.getUnknownLoc(), ptrVal);
      auto bodyArg = loopOp.bodyRegion().getArgument(bodyArgIdx++);
      auto argVal = typeConverter.materializeSourceConversion(
          rewriter, rewriter.getUnknownLoc(), bodyArg.getType(),
          mlir::ValueRange{fieldVal});
      mapping.map(bodyArg, argVal);
    }

    // Clone the loop body, replacing the entry block arguments
    // with our materialized mappings.
    rewriter.cloneRegionBefore(loopOp.bodyRegion(), execFunc.getBody(),
                               execFunc.getBody().end(), mapping);

    // Connect the entry block to the initial loop block.  Note that it's
    // always safe to merge the blocks: the entry block has no terminator,
    // the loop's entry block has no predecessors (since it's an entry
    // block), and the loop's entry block's arguments were remapped in
    // the cloning process.
    auto it = execFunc.getRegion().begin();
    ++it;
    rewriter.mergeBlocks(&*it, execEntryBlock, mlir::None);

    return rewriter.convertRegionTypes(&execFunc.getBody(), typeConverter);
  }

  mlir::LogicalResult
  buildFini(abi::LoopOp loopOp, LLVMType networkTy,
            mlir::ConversionPatternRewriter &rewriter) const {
    // Create plaidml_fini().
    //
    // For now, we don't have a fini region; nothing is being cleaned up
    // except for the actual network pointer.
    // So we build plaidml_fini as a simple function that frees the network.

    mlir::OpBuilder::InsertionGuard insertionGuard{rewriter};

    auto freeFunc = importFunc(
        "free", loopOp,
        LLVMType::getFunctionTy(getVoidType(),
                                mlir::ArrayRef<LLVMType>{getVoidPtrType()},
                                /*isVarArg=*/false),
        rewriter);

    auto finiFunc = rewriter.create<LLVM::LLVMFuncOp>(
        loopOp.getLoc(), pmlc::util::kPlaidmlFini,
        LLVMType::getFunctionTy(getVoidType(), {networkTy.getPointerTo()},
                                /*isVarArg=*/false));

    auto *entryBlock =
        rewriter.createBlock(&finiFunc.getBody(), finiFunc.getBody().end(),
                             mlir::TypeRange{networkTy.getPointerTo()});

    rewriter.setInsertionPointToStart(entryBlock);

    auto networkRawPtr = rewriter.create<LLVM::BitcastOp>(
        loopOp.getLoc(), getVoidPtrType(), entryBlock->getArgument(0));
    rewriter.create<LLVM::CallOp>(loopOp.getLoc(), freeFunc,
                                  mlir::ValueRange{networkRawPtr});
    rewriter.create<LLVM::ReturnOp>(loopOp.getLoc(), mlir::None);

    return mlir::success();
  }
};

} // namespace

void populateABIToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter,
    mlir::OwningRewritePatternList &patterns) {
  patterns.insert<CreateNetworkOpLowering, DoneOpLowering, LoopOpLowering>(
      converter);
}

} // namespace pmlc::conversion::abi_to_llvm
