// Copyright 2020, Intel Corporation

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/conversion/abi_to_llvm/passes.h"
#include "pmlc/dialect/abi/ir/dialect.h"
#include "pmlc/util/ids.h"

#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"

namespace abi = pmlc::dialect::abi;
namespace LLVM = mlir::LLVM;
using LLVMType = LLVM::LLVMType;

namespace pmlc::conversion::abi_to_llvm {

static abi::LoopOp findLoopToLower(LLVM::LLVMFuncOp funcOp) {
  auto moduleOp = funcOp.getParentOfType<mlir::ModuleOp>();
  auto existingInit = moduleOp.lookupSymbol(pmlc::util::kPlaidmlInit);
  if (existingInit) {
    return abi::LoopOp{};
  }

  abi::LoopOp loopOp;
  for (auto childOp : funcOp.getOps<abi::LoopOp>()) {
    if (loopOp) {
      return abi::LoopOp{}; // Two loop ops => failure
    }
    loopOp = childOp;
  }

  return loopOp;
}

namespace {

struct LoopLowering : public mlir::ConvertOpToLLVMPattern<LLVM::LLVMFuncOp> {
  using mlir::ConvertOpToLLVMPattern<LLVM::LLVMFuncOp>::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    llvm::DebugFlag = true;
    auto mainFunc = mlir::cast<LLVM::LLVMFuncOp>(op);
    auto loopOp = findLoopToLower(mainFunc);
    if (!loopOp) {
      return mlir::failure();
    }

    llvm::errs() << "Converting fn: " << mainFunc << "\n";

    for (auto operand : loopOp.getOperation()->getOperands()) {
      llvm::errs() << "Found loop operand: " << operand
                   << " type=" << operand.getType() << "\n";
    }

    auto ctx = rewriter.getContext();
    auto networkTy = LLVM::LLVMStructType::getIdentified(ctx, "Network");

    auto initFunc = rewriter.create<LLVM::LLVMFuncOp>(
        mainFunc.getLoc(), pmlc::util::kPlaidmlInit,
        LLVMType::getFunctionTy(
            networkTy.getPointerTo(),
            mainFunc.getType().cast<LLVM::LLVMFunctionType>().getParams(),
            /*isVarArg=*/false));

    // The inputs to the exec function are the network object, followed by the
    // raw buffer pointers of all inputs.
    mlir::SmallVector<LLVMType, 8> execArgTypes;
    execArgTypes.emplace_back(networkTy.getPointerTo());

    for (auto operand : loopOp.getOperation()->getOperands()) {
      auto ty = getTypeConverter()
                    ->convertType(operand.getType())
                    .dyn_cast_or_null<LLVM::LLVMStructType>();
      if (!ty) {
        continue;
      }
      execArgTypes.emplace_back(ty.getPointerTo());
    }

    auto execFunc = rewriter.create<LLVM::LLVMFuncOp>(
        mainFunc.getLoc(), pmlc::util::kPlaidmlExec,
        LLVMType::getFunctionTy(getVoidType(), execArgTypes,
                                /*isVarArg=*/false));

    auto finiFunc = rewriter.create<LLVM::LLVMFuncOp>(
        mainFunc.getLoc(), pmlc::util::kPlaidmlFini,
        LLVMType::getFunctionTy(getVoidType(), {networkTy.getPointerTo()},
                                /*isVarArg=*/false));

    // Okay, so now I need an algorithm for this.  Hmm.
    // We have a pile of instructions; we need them to be in
    // our three different routines...

    // We have a single region.  We can walk through the region's blocks,
    // cloning the instructions one by one; we can also clone entire regions
    // as a single thing, and insert a region's blocks into an existing region.

    // One very obvious thing to do is to clone the entire loop region
    // into plaidml_exec.  To do this, we need to build a BlockAndValueMapping.

    // Suppose we walked every operation before the loop, and created a
    // "Get this value from the Network" placeholder instruction in exec().
    // Then, we could clone the loop body (appending it) into exec(),
    // using the mapper to rewrite accesses.  We could then delete unused
    // placeholders, and we'd know exactly which values we needed from the
    // network... Hmm.

    // And then our accessor operations form a single block, whose final
    // instruction is the loop...?  But what if the loop contains multiple
    // blocks? Hmm.

    // Actually, I think that'll be fine, by my reading of both the MLIR spec
    // and the SCFToStandard lowering logic for If statements -- it's pretty
    // clear that operations can access any value defined within their region,
    // or there's a lot of code that's very, very broken.

    // So.  Where're we at?

    // INIT
    // We can split the abi.loop block before the loop.  The before block will
    // then not contain a terminator; this is where we will insert the
    // network structure creation and return.  Then that block, and all its
    // predecessors (which had better not contain the loop) can be moved to
    // plaidml_init().  This assumes we know what to put into the network.

    // EXEC
    // We can produce an entry block containing network extraction operation
    // for the region arguments and every value produced by the logical
    // predecessors of the loop block.  We can then inline the loop's region
    // after the entry block, and remove every unused extraction operation;
    // the remainder are values we need to squirrel away in the
    // network structure.

    // FINI
    // We can split the abi.loop block after the loop.  The after block and
    // all successors become plaidml_fini.  As with exec, we produce an entry
    // block containing network extraction operations for the region arguments
    // and the predecessors of the fini block (which is just the immediate
    // successor of the loop), and then move them over.

    // No, that all doesn't work, because we need to stick with actions that
    // can be expressed with a rewriter.  Bother.  Okay, what can we do?

    // Well: we can still split the block at the loop.  So that's great;
    // now the loop is the first instruction in a new block.

    // We can't clone blocks, but we can clone regions.  Hmm.

    // Suppose we create a mapper, and walk the loop region.  For each
    // operation, for each operand, we can look at the mapper, and ask
    // whether we the value comes from the loop's region.  If it does,
    // and we don't have a definition for it, it's a value that needs
    // to come from the Network: so we can create an accessor for it in
    // exec(), and add it to the mapping.  Finally, we can clone the loop
    // region, replace loop.done with llvm.return, and call exec() done.

    // Let's write that up and see how far we get.

    llvm::errs() << "Initial plaidml_exec: " << execFunc << "\n";

    rewriter.startRootUpdate(mainFunc);

    // Build plaidml_exec().
    //
    // To do this, we start by walking the loop region.  Every operand that
    // comes from the main function region is going to have to come from the
    // Network once the loop code's been moved over, so when we see an operand
    // that's from the main function region and not already in the mapper, we
    // create an accessor in plaidml_exec(), and then add it to the mapper.
    // Once we're done, we can clone the entire loop region using the mapper,
    // and the cloned code will wind up using the correct input values.
    mlir::BlockAndValueMapping mapper;
    mlir::Block *execEntryBlock = execFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(execEntryBlock);

    auto execNetworkPtr = execEntryBlock->getArgument(0);
    auto execNetwork =
        rewriter.create<LLVM::LoadOp>(rewriter.getUnknownLoc(), execNetworkPtr);
    mlir::SmallVector<LLVMType, 16> networkFieldTypes;

    // Build the non-input-memref-buffer mapper entries and value extraction
    // operations.
    loopOp.getRegion().walk([&](mlir::Operation *op) {
      for (auto operand : op->getOperands()) {
        if (operand.getParentRegion() == loopOp.getParentRegion() &&
            !mapper.contains(operand) &&
            operand.getKind() != mlir::Value::Kind::BlockArgument) {
          if (auto constOp = mlir::dyn_cast_or_null<mlir::ConstantOp>(
                  operand.getDefiningOp())) {
            // TODO: It might be useful to copy additional operations into the
            // generated plaidml_exec, instead of plumbing them through from
            // the network, to increase the number of values we can compute
            // or elide at compilation time.
            auto newOpSrc = rewriter.create<mlir::ConstantOp>(
                constOp.getLoc(), constOp.getValue());
            mapper.map(constOp, newOpSrc);
          } else {
            llvm::errs() << "Creating extractor for: "
                         << mlir::debugString(*operand.getDefiningOp()) << "\n";
            LLVMType opType =
                typeConverter.convertType(operand.getType()).cast<LLVMType>();
            llvm::errs() << "OpType for " << operand.getType() << " is "
                         << opType << "\n";
            auto newOpSrc = rewriter.create<LLVM::ExtractValueOp>(
                rewriter.getUnknownLoc(), opType, execNetwork,
                rewriter.getI64ArrayAttr(networkFieldTypes.size()));
            mapper.map(operand, newOpSrc);
            networkFieldTypes.emplace_back(opType);
          }
        }
      }
    });

    // Build the arguments for the loop's entry block; it's convenient to
    // do this now, since the rewriter's in a good place.
    mlir::SmallVector<mlir::Value, 8> loopInputs;
    for (auto idx = 1; idx < execEntryBlock->getNumArguments(); ++idx) {
      loopInputs.emplace_back(rewriter.create<LLVM::LoadOp>(
          rewriter.getUnknownLoc(), execEntryBlock->getArgument(idx)));
    }

    // Clone the loop's region into plaidml_exec().
    rewriter.cloneRegionBefore(loopOp.getRegion(), execFunc.getRegion(),
                               execFunc.getRegion().end(), mapper);

    llvm::errs() << "After clone: plaidml_exec: " << execFunc << "\n";

    auto conversionResult =
        rewriter.convertRegionTypes(&execFunc.getRegion(), *getTypeConverter());
    if (mlir::failed(conversionResult)) {
      llvm::errs() << "Region type conversion failed\n";
      rewriter.cancelRootUpdate(mainFunc);
      return mlir::failure();
    }

    execEntryBlock = conversionResult.getValue();

    llvm::errs() << "After region type conversion: plaidml_exec: " << execFunc
                 << "\n";

    // Connect the entry block to the initial loop block.  Note that it's
    // always safe to merge the blocks: the entry block has no terminator
    // (yet), the loop's entry block has no predecessors (since it's an entry
    // block), and the loop's entry block's arguments are the memref descriptors
    // (now that they've been converted), which means we can replace them with
    // loads from the corresponding input arguments.
    auto it = execFunc.getRegion().begin();
    ++it;
    rewriter.mergeBlocks(&*it, execEntryBlock, loopInputs);

    llvm::errs() << "After merging: plaidml_exec: " << execFunc << "\n";

    // Replace all abi.done operations within the clone with llvm.return.
    execFunc.walk([&](abi::DoneOp doneOp) {
      rewriter.setInsertionPoint(doneOp);
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(doneOp, llvm::None);
    });

    llvm::errs() << "After replacement: plaidml_exec: " << execFunc << "\n";
    llvm::errs() << "After replacement: main: " << mainFunc << "\n";

    // Build plaidml_fini().
    //
    // This is a little different from the plaidml_exec build, since we're
    // restricted to the operations supported by the rewriter and we don't have
    // the code we need in a convenient sub-region.
    //
    // Hmm.  What if we did?  That's a fascinating idea...
    // Suppose we had a pass that added pre- and post- operations?
    // Then the outputs of the pre- operation would be exactly the
    // values that would be wrapped up into our Network object.
    //
    // EVEN BETTER: abi.Loop can have multiple regions.
    // When we hoist, we can move code from the loop body to the initialization
    // region.  The main function DISAPPEARS.  It becomes abi.loop.  ZOMG.

    llvm::errs() << "After eraseage: main: " << mainFunc << "\n";

    rewriter.eraseOp(loopOp);
    LLVMType::setStructTyBody(networkTy, networkFieldTypes);

    rewriter.finalizeRootUpdate(mainFunc);

    llvm::errs() << "Final: plaidml_exec: " << execFunc << "\n";

    return mlir::success();
  }
};

} // namespace

void populateABIToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter,
    mlir::OwningRewritePatternList &patterns) {
  patterns.insert<LoopLowering>(converter);
}

void addLoopLegality(mlir::ConversionTarget &target) {
  target.addDynamicallyLegalOp<LLVM::LLVMFuncOp>([](LLVM::LLVMFuncOp op) {
    if (findLoopToLower(op)) {
      // We have a loop to lower, so this op is not legal.
      return false;
    }
    // With no loop to lower, there's nothing for us to do.
    return true;
  });
}

} // namespace pmlc::conversion::abi_to_llvm
