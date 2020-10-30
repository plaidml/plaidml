// Copyright 2020, Intel Corporation

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

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

constexpr char emptyNetworkSingletonName[] = "plaidml_empty_network";

struct CreateNetworkOpLowering final
    : public mlir::ConvertOpToLLVMPattern<abi::CreateNetworkOp> {
  using mlir::ConvertOpToLLVMPattern<
      abi::CreateNetworkOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto createNetworkOp = mlir::cast<abi::CreateNetworkOp>(op);

    rewriter.startRootUpdate(createNetworkOp);
    rewriter.setInsertionPoint(createNetworkOp);

    mlir::Value networkPtr;

    if (createNetworkOp.getNumOperands() == 0) {
      // There's no data being passed from initialization to the body:
      // we don't want to return nullptr (since that's our failure indication),
      // but returning a Network* is a little simpler than managing an output
      // parameter.  So: we create a global to represent an empty network
      // singleton, and return a pointer to it.
      auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
      LLVM::GlobalOp emptyNetwork =
          moduleOp.lookupSymbol<LLVM::GlobalOp>(emptyNetworkSingletonName);
      if (!emptyNetwork) {
        mlir::OpBuilder::InsertionGuard insertionGuard{rewriter};
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        emptyNetwork = rewriter.create<LLVM::GlobalOp>(
            rewriter.getUnknownLoc(),
            LLVMType::getInt8Ty(rewriter.getContext()), /*isConstant=*/true,
            LLVM::Linkage::Internal, emptyNetworkSingletonName,
            rewriter.getI8IntegerAttr(0));
      }

      networkPtr = rewriter.create<LLVM::AddressOfOp>(rewriter.getUnknownLoc(),
                                                      emptyNetwork);

    } else {
      auto mallocFunc = importFunc(
          "malloc", createNetworkOp,
          LLVMType::getFunctionTy(getVoidPtrType(),
                                  mlir::ArrayRef<LLVMType>{getIndexType()},
                                  /*isVarArg=*/false),
          rewriter);

      // Figure out the type of the network structure.
      mlir::SmallVector<LLVMType, 8> networkFieldTypes;
      mlir::SmallVector<mlir::Value, 8> networkFieldValues;
      for (auto srcValue : createNetworkOp.getOperands()) {
        auto convTy = typeConverter.convertType(srcValue.getType())
                          .dyn_cast_or_null<LLVMType>();
        if (!convTy) {
          rewriter.cancelRootUpdate(createNetworkOp);
          return mlir::failure();
        }
        auto convValue = typeConverter.materializeTargetConversion(
            rewriter, createNetworkOp.getLoc(), convTy, srcValue);
        if (!convValue) {
          rewriter.cancelRootUpdate(createNetworkOp);
          return mlir::failure();
        }
        networkFieldTypes.emplace_back(convTy);
        networkFieldValues.emplace_back(convValue);
      }

      auto networkTy =
          LLVMType::getStructTy(rewriter.getContext(), networkFieldTypes);

      // Fill in a local stack copy of the network structure.
      mlir::Value networkValue =
          rewriter.create<LLVM::UndefOp>(rewriter.getUnknownLoc(), networkTy);
      for (unsigned idx = 0; idx < networkFieldValues.size(); ++idx) {
        networkValue = rewriter.create<LLVM::InsertValueOp>(
            rewriter.getUnknownLoc(), networkTy, networkValue,
            networkFieldValues[idx], rewriter.getI64ArrayAttr(idx));
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
      networkPtr = rewriter.create<LLVM::BitcastOp>(rewriter.getUnknownLoc(),
                                                    networkTy.getPointerTo(),
                                                    initNetworkRawPtr);
      rewriter.create<LLVM::StoreOp>(rewriter.getUnknownLoc(), networkValue,
                                     networkPtr);
    }

    // Add the new LLVM terminator, replacing the abi.create_network op.
    rewriter.create<LLVM::ReturnOp>(rewriter.getUnknownLoc(),
                                    mlir::ValueRange{networkPtr});
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
    // llvm::errs() << "Lowering " << loopOp << "\n";
    // llvm::DebugFlag = true;

    mlir::SmallVector<LLVMType, 8> networkFieldTypes;
    for (auto ty : loopOp.getNetworkFieldTypes()) {
      auto newTy = typeConverter.convertType(ty).dyn_cast_or_null<LLVMType>();
      if (!newTy) {
        return mlir::failure();
      }
      networkFieldTypes.emplace_back(newTy);
    }

    auto networkTy =
        LLVMType::getStructTy(rewriter.getContext(), networkFieldTypes);

    rewriter.startRootUpdate(loopOp);
    rewriter.setInsertionPoint(loopOp);

    bool hasNetworkFields = networkFieldTypes.size() != 0;

    if (mlir::failed( //
            buildInit(loopOp, networkTy, hasNetworkFields, rewriter)) ||
        mlir::failed( //
            buildExec(loopOp, networkTy, hasNetworkFields, networkFieldTypes,
                      rewriter)) ||
        mlir::failed( //
            buildFini(loopOp, networkTy, hasNetworkFields, rewriter))) {
      rewriter.cancelRootUpdate(loopOp);
      return mlir::failure();
    }

    rewriter.eraseOp(loopOp);
    rewriter.finalizeRootUpdate(loopOp);

    return mlir::success();
  }

private:
  mlir::LogicalResult
  buildInit(abi::LoopOp loopOp, LLVMType networkTy, bool hasNetworkFields,
            mlir::ConversionPatternRewriter &rewriter) const {
    // Create plaidml_init().

    mlir::OpBuilder::InsertionGuard insertionGuard{rewriter};
    auto argTypes = loopOp.initRegion().getArgumentTypes();
    auto argIt = argTypes.begin();

    LLVMType devicePtrTy;
    if (loopOp.initRegion().getNumArguments() == 0) {
      devicePtrTy = getVoidPtrType();
    } else {
      devicePtrTy = (*argIt++).cast<LLVMType>();
    }

    mlir::SmallVector<LLVMType, 8> initFieldTypes;
    LLVMType initTy;
    LLVMType initPtrTy;
    if (loopOp.initRegion().getNumArguments() <= 1) {
      initPtrTy = getVoidPtrType();
    } else {
      for (unsigned idx = 1; idx < loopOp.initRegion().getNumArguments();
           ++idx) {
        auto ty =
            typeConverter.convertType(*argIt++).dyn_cast_or_null<LLVMType>();
        if (!ty) {
          return mlir::failure();
        }
        initFieldTypes.emplace_back(ty.getPointerTo());
      }
      initTy = LLVMType::getStructTy(rewriter.getContext(), initFieldTypes);
      initPtrTy = initTy.getPointerTo();
    }

    auto initType = LLVMType::getFunctionTy(
        hasNetworkFields ? networkTy.getPointerTo() : getVoidPtrType(),
        {devicePtrTy, initPtrTy},
        /*isVarArg=*/false);

    auto initFunc = rewriter.create<LLVM::LLVMFuncOp>(
        loopOp.getLoc(), pmlc::util::kPlaidmlInit, initType);

    // Create an entry block, and materialize the arguments for the
    // initialization.
    auto *initEntryBlock =
        rewriter.createBlock(&initFunc.getBody(), initFunc.getBody().end(),
                             {devicePtrTy, initPtrTy});
    rewriter.setInsertionPointToStart(initEntryBlock);
    mlir::BlockAndValueMapping mapping;
    mapping.map(loopOp.initRegion().getArgument(0),
                initEntryBlock->getArgument(0));

    if (initTy) {
      unsigned initArgIdx = 1;
      auto initValue = rewriter.create<LLVM::LoadOp>(
          rewriter.getUnknownLoc(), initEntryBlock->getArgument(1));
      for (auto fieldIdx = 0; fieldIdx < initFieldTypes.size(); ++fieldIdx) {
        auto ptrVal = rewriter.create<LLVM::ExtractValueOp>(
            rewriter.getUnknownLoc(), initFieldTypes[fieldIdx], initValue,
            rewriter.getI64ArrayAttr(fieldIdx));
        auto fieldVal =
            rewriter.create<LLVM::LoadOp>(rewriter.getUnknownLoc(), ptrVal);
        auto bodyArg = loopOp.initRegion().getArgument(initArgIdx++);
        auto argVal = typeConverter.materializeSourceConversion(
            rewriter, rewriter.getUnknownLoc(), bodyArg.getType(),
            mlir::ValueRange{fieldVal});
        mapping.map(bodyArg, argVal);
      }
    }

    rewriter.cloneRegionBefore(loopOp.initRegion(), initFunc.getBody(),
                               initFunc.getBody().end(), mapping);

    // Connect the entry block to the initial init block.  Note that it's
    // always safe to merge the blocks: the entry block has no terminator,
    // the initialization entry block has no predecessors (since it's an entry
    // block), and the initialization's entry block's arguments were remapped in
    // the cloning process.
    auto it = initFunc.getRegion().begin();
    ++it;
    rewriter.mergeBlocks(&*it, initEntryBlock, mlir::None);

    auto result =
        rewriter.convertRegionTypes(&initFunc.getBody(), typeConverter);
    // lvm::errs() << "Constructed " << initFunc << "\n";
    return result;
  }

  mlir::LogicalResult
  buildExec(abi::LoopOp loopOp, LLVMType networkTy, bool hasNetworkFields,
            const mlir::SmallVectorImpl<LLVMType> &networkFieldTypes,
            mlir::ConversionPatternRewriter &rewriter) const {
    // Create plaidml_exec().
    //
    // The inputs to the exec function are a pointer to a network object, and a
    // pointer to a struct with per-iteration parameters; each struct field is a
    // pointer to the LLVM type of the corresponding entry block argument
    // (the arguments after the arguments passed via the network structure).

    mlir::OpBuilder::InsertionGuard insertionGuard{rewriter};
    mlir::SmallVector<LLVMType, 8> iterationFieldTypes;
    for (unsigned idx = loopOp.getNumNetworkFields();
         idx < loopOp.bodyRegion().getNumArguments(); ++idx) {
      auto ty = typeConverter
                    .convertType(loopOp.bodyRegion().getArgument(idx).getType())
                    .dyn_cast_or_null<LLVMType>();
      if (!ty) {
        return mlir::failure();
      }
      iterationFieldTypes.emplace_back(ty.getPointerTo());
    }
    auto iterationTy =
        LLVMType::getStructTy(rewriter.getContext(), iterationFieldTypes);

    auto execFunc = rewriter.create<LLVM::LLVMFuncOp>(
        loopOp.getLoc(), pmlc::util::kPlaidmlExec,
        LLVMType::getFunctionTy(
            getVoidType(),
            mlir::ArrayRef<LLVMType>{hasNetworkFields ? networkTy.getPointerTo()
                                                      : getVoidPtrType(),
                                     iterationTy.getPointerTo()},
            /*isVarArg=*/false));

    // Create an entry block, and materialize the arguments for the loop body.
    auto *execEntryBlock = rewriter.createBlock(
        &execFunc.getBody(), execFunc.getBody().end(),
        mlir::TypeRange{hasNetworkFields ? networkTy.getPointerTo()
                                         : getVoidPtrType(),
                        iterationTy.getPointerTo()});
    rewriter.setInsertionPointToStart(execEntryBlock);
    mlir::BlockAndValueMapping mapping;
    unsigned bodyArgIdx = 0;

    if (hasNetworkFields) {
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

    auto result =
        rewriter.convertRegionTypes(&execFunc.getBody(), typeConverter);
    // llvm::errs() << "Constructed " << execFunc << "\n";
    return result;
  }

  mlir::LogicalResult
  buildFini(abi::LoopOp loopOp, LLVMType networkTy, bool hasNetworkFields,
            mlir::ConversionPatternRewriter &rewriter) const {
    // Create plaidml_fini().
    //
    // For now, we don't have a fini region; nothing is being cleaned up
    // except for the actual network pointer.
    // So we build plaidml_fini as a simple function that frees the network.

    mlir::OpBuilder::InsertionGuard insertionGuard{rewriter};
    auto argType =
        hasNetworkFields ? networkTy.getPointerTo() : getVoidPtrType();
    auto finiFunc = rewriter.create<LLVM::LLVMFuncOp>(
        loopOp.getLoc(), pmlc::util::kPlaidmlFini,
        LLVMType::getFunctionTy(getVoidType(), {argType},
                                /*isVarArg=*/false));

    auto *entryBlock =
        rewriter.createBlock(&finiFunc.getBody(), finiFunc.getBody().end(),
                             mlir::TypeRange{argType});

    if (hasNetworkFields) {
      auto freeFunc = importFunc(
          "free", loopOp,
          LLVMType::getFunctionTy(getVoidType(),
                                  mlir::ArrayRef<LLVMType>{getVoidPtrType()},
                                  /*isVarArg=*/false),
          rewriter);

      rewriter.setInsertionPointToStart(entryBlock);

      auto networkRawPtr = rewriter.create<LLVM::BitcastOp>(
          loopOp.getLoc(), getVoidPtrType(), entryBlock->getArgument(0));
      rewriter.create<LLVM::CallOp>(loopOp.getLoc(), freeFunc,
                                    mlir::ValueRange{networkRawPtr});
    }

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
