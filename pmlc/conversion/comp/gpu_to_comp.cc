// Copyright 2020, Intel Corporation
#include <iostream>
#include <vector>

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/comp/pass_detail.h"
#include "pmlc/dialect/comp/ir/dialect.h"

namespace pmlc::conversion::comp {

namespace comp = pmlc::dialect::comp;
namespace gpu = mlir::gpu;

class ConvertGpuToComp : public ConvertGpuToCompBase<ConvertGpuToComp> {
public:
  void runOnOperation();
};

namespace {

struct RewriteLaunchFunc : public mlir::OpRewritePattern<gpu::LaunchFuncOp> {
  /// Rewrite gpu.launch_func into chain of comp dialect operations.
  /// Wraps original launch operation into comp.schedule_func.
  /// Execution environment and device memory are created and destroyed
  /// separately for each launch_func, so further optimizations to reuse are
  /// needed.
  RewriteLaunchFunc(mlir::MLIRContext *context, comp::ExecEnvType execEnvType)
      : mlir::OpRewritePattern<gpu::LaunchFuncOp>(context),
        execEnvType(execEnvType) {}

  mlir::LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp op,
                  mlir::PatternRewriter &rewriter) const override;

  comp::ExecEnvType execEnvType;
};

struct RewriteGpuFunc : public mlir::OpConversionPattern<gpu::GPUFuncOp> {
  /// Update gpu.func to signature with memrefs in device memory space.
  RewriteGpuFunc(mlir::MLIRContext *context, unsigned memorySpace)
      : mlir::OpConversionPattern<gpu::GPUFuncOp>(context),
        memorySpace(memorySpace) {}

  mlir::LogicalResult
  matchAndRewrite(gpu::GPUFuncOp op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override;
  unsigned memorySpace;
};

struct ReuseExecEnv : public mlir::OpRewritePattern<comp::DestroyExecEnv> {
  /// Rewrites environment destruction and creation into reuse of previously
  /// created environment. Does not cross block boundaries.
  using mlir::OpRewritePattern<comp::DestroyExecEnv>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(comp::DestroyExecEnv op,
                  mlir::PatternRewriter &rewriter) const override;
};

struct ReuseAllocatedMemory : public mlir::OpRewritePattern<comp::Dealloc> {
  /// Rewrites chain of device memory dealloc and alloc into reuse of previously
  /// allocated memory. Does not cross block boundaries.
  using mlir::OpRewritePattern<comp::Dealloc>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(comp::Dealloc op,
                  mlir::PatternRewriter &rewriter) const override;
};

} // namespace

mlir::LogicalResult
RewriteLaunchFunc::matchAndRewrite(gpu::LaunchFuncOp op,
                                   mlir::PatternRewriter &rewriter) const {
  // Replace gpu.launch_func %indexes..., %args... with:
  // %env = comp.create_execenv
  // %new_args = (comp.alloc %env %args)...
  // %ev = comp.schedule_func %env {
  //   gpu.launch_func %indexes..., %new_args...
  // }
  // %wr_events = comp.schedule_read %new_args to %args on %env wait for %ev
  // %final_ev = comp.group_events %wr_events...
  // comp.wait %final_ev
  // comp.dealloc %new_args...
  // comp.destroy_execenv %env

  auto loc = op.getLoc();
  auto context = rewriter.getContext();
  auto runtime = execEnvType.getRuntime();

  // Create execution environment
  // TODO How to control what execenv gets created?
  auto execEnvOp = rewriter.create<comp::CreateExecEnv>(loc, execEnvType);
  auto execEnv = execEnvOp.getResult();
  auto eventType = rewriter.getType<comp::EventType>(runtime);

  // Allocate memory on device
  std::vector<mlir::Value> newOperands;
  std::vector<mlir::Value> allocOriginal;
  std::vector<mlir::Value> allocNew;
  newOperands.reserve(op.getNumKernelOperands());
  for (auto &operand : op.getOperands()) {
    if (auto memRefType =
            operand.getType().dyn_cast_or_null<mlir::MemRefType>()) {
      mlir::MemRefType newMemRefType =
          mlir::MemRefType::Builder(memRefType)
              .setMemorySpace(execEnvType.getDefaultMemorySpace());
      auto allocOp =
          rewriter.create<comp::Alloc>(loc, newMemRefType, execEnv, operand);
      auto newMemRef = allocOp.getResult();
      newOperands.push_back(newMemRef);
      allocOriginal.push_back(operand);
      allocNew.push_back(newMemRef);
    } else {
      newOperands.push_back(operand);
    }
  }
  // Update operands for launch
  op.getOperation()->setOperands(newOperands);
  mlir::SymbolRefAttr kernel = op.kernel();

  // Move original op into schedule func
  auto scheduleFuncOp = rewriter.create<comp::ScheduleFunc>(
      loc, eventType, execEnv, mlir::Value());
  auto postScheduleIP = rewriter.saveInsertionPoint();
  auto newBlock =
      rewriter.createBlock(&scheduleFuncOp.body(), scheduleFuncOp.body().end());
  // TODO Is there a better way to move operation into block insted of
  // clone->erase
  newBlock->push_back(op.clone());
  rewriter.create<comp::ScheduleEnd>(loc);
  // TODO Isn't there RAII-idiomatic way to save/restore insertion point
  rewriter.restoreInsertionPoint(postScheduleIP);
  auto funcEvent = scheduleFuncOp.getResult();
  // Read back the memory
  std::vector<mlir::Value> readEvents;
  readEvents.reserve(allocNew.size());
  for (size_t allocIdx = 0; allocIdx < allocNew.size(); ++allocIdx) {
    auto &allocated = allocNew[allocIdx];
    auto &original = allocOriginal[allocIdx];
    auto readOp = rewriter.create<comp::ScheduleRead>(
        loc, eventType, original, allocated, execEnv, funcEvent);
    readEvents.push_back(readOp.getResult());
  }
  // Wait for all operations to finish
  auto groupOp = rewriter.create<comp::GroupEvents>(loc, eventType, readEvents);
  auto allEvent = groupOp.getResult();
  rewriter.create<comp::Wait>(loc, allEvent);
  // Deallocate memory on device
  for (size_t allocIdx = 0; allocIdx < allocNew.size(); ++allocIdx) {
    auto &allocated = allocNew[allocIdx];
    rewriter.create<comp::Dealloc>(loc, allocated);
  }
  // Destroy environment
  rewriter.create<comp::DestroyExecEnv>(loc, execEnv);
  // Remove original op
  rewriter.eraseOp(op.getOperation());
  return mlir::success();
}

mlir::LogicalResult RewriteGpuFunc::matchAndRewrite(
    gpu::GPUFuncOp op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::TypeConverter converter;
  // Default pass-through conversion
  converter.addConversion([](mlir::Type type) { return type; });
  // Change memrefs memory space
  converter.addConversion([&](mlir::MemRefType type) {
    return mlir::MemRefType::get(type.getShape(), type.getElementType(),
                                 type.getAffineMaps(), memorySpace);
  });

  auto funcType = op.getType();
  mlir::TypeConverter::SignatureConversion signatureConversion(
      funcType.getNumInputs());
  for (auto argType : llvm::enumerate(funcType.getInputs())) {
    auto convertedType = converter.convertType(argType.value());
    signatureConversion.addInputs(argType.index(), convertedType);
  }

  if (mlir::failed(rewriter.convertRegionTypes(&op.body(), converter,
                                               &signatureConversion)))
    return mlir::failure();

  // gpu.func_op should have no results
  rewriter.updateRootInPlace(op, [&] {
    op.setType(
        rewriter.getFunctionType(signatureConversion.getConvertedTypes(), {}));
  });

  return mlir::success();
}

mlir::LogicalResult
ReuseExecEnv::matchAndRewrite(comp::DestroyExecEnv op,
                              mlir::PatternRewriter &rewriter) const {
  // Pattern:
  // comp.destroy_execenv %env1 : execenv<A>
  // %env2 = comp.create_execenv : execenv<A>
  auto destroyEnv = op.execEnv();
  auto destroyType = destroyEnv.getType();

  auto operation = op.getOperation()->getNextNode();
  while (operation) {
    if (auto createOp = mlir::dyn_cast<comp::CreateExecEnv>(operation)) {
      auto createType = createOp.execEnv().getType();
      if (createType == destroyType) {
        rewriter.replaceOp(operation, {destroyEnv});
        rewriter.eraseOp(op);
        return mlir::success();
      }
    }

    operation = operation->getNextNode();
  }
  return mlir::failure();
}

mlir::LogicalResult
ReuseAllocatedMemory::matchAndRewrite(comp::Dealloc op,
                                      mlir::PatternRewriter &rewriter) const {
  // Pattern:
  // %device = comp.alloc %env %host
  // ... (1) - use %device
  // comp.dealloc %device
  // ... (2) - use %host
  // %device2 = comp.alloc %env %host
  // ... (3) - use %device2
  // comp.dealloc %device2
  //
  // Into:
  // %device = comp.alloc %env %host
  // ... (1)
  // ... (2)
  // %ev = comp.write %host %device
  // comp.wait %ev
  // ... (3)
  // comp.dealloc %device

  auto deviceMem = op.deviceMem();
  auto allocOp = mlir::cast<comp::Alloc>(deviceMem.getDefiningOp());
  if (!allocOp)
    return mlir::failure();
  auto hostMem = allocOp.hostMem();
  auto execEnv = allocOp.execEnv();
  auto execEnvType = execEnv.getType().cast<comp::ExecEnvType>();
  auto eventType = rewriter.getType<comp::EventType>(execEnvType.getRuntime());

  // TODO More general memory reuse
  if (!hostMem)
    return mlir::failure();

  auto operation = op.getOperation()->getNextNode();
  while (operation) {
    if (auto secondAlloc = mlir::dyn_cast<comp::Alloc>(operation)) {
      if (secondAlloc.hostMem() == hostMem &&
          secondAlloc.execEnv() == execEnv) {
        rewriter.setInsertionPoint(operation);
        auto writeOp = rewriter.create<comp::ScheduleWrite>(
            operation->getLoc(), eventType, hostMem, deviceMem, execEnv,
            mlir::Value());
        auto writeEvent = writeOp.getResult();
        rewriter.create<comp::Wait>(operation->getLoc(), writeEvent);
        rewriter.eraseOp(op);
        rewriter.replaceOp(operation, deviceMem);
        return mlir::success();
      }
    }

    operation = operation->getNextNode();
  }
  return mlir::failure();
}

void ConvertGpuToComp::runOnOperation() {
  comp::ExecEnvRuntime runtime =
      static_cast<comp::ExecEnvRuntime>(execEnvRuntime.getValue());
  unsigned memorySpace = execEnvMemorySpace.getValue();
  auto execEnvType =
      comp::ExecEnvType::get(&getContext(), runtime, 0, {memorySpace});

  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<comp::COMPDialect>();
  target.addLegalDialect<gpu::GPUDialect>();
  target.addDynamicallyLegalOp<gpu::LaunchFuncOp>(
      [](gpu::LaunchFuncOp op) -> bool {
        auto parent = op.getParentOp();
        return parent && mlir::isa<comp::ScheduleFunc>(parent);
      });
  target.addDynamicallyLegalOp<gpu::GPUFuncOp>([&](gpu::GPUFuncOp op) -> bool {
    for (auto arg : op.getArguments()) {
      if (auto memRefType =
              arg.getType().dyn_cast_or_null<mlir::MemRefType>()) {
        if (memorySpace != memRefType.getMemorySpace())
          return false;
      }
    }
    return true;
  });
  target.addIllegalOp<gpu::LaunchOp>();
  // Everything else should be legal
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });

  // Setup rewrite patterns
  mlir::OwningRewritePatternList patterns;
  patterns.insert<RewriteLaunchFunc>(&getContext(), execEnvType);
  patterns.insert<RewriteGpuFunc>(&getContext(), memorySpace);

  // Run the conversion
  if (mlir::failed(
          mlir::applyFullConversion(getOperation(), target, patterns))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> createConvertGpuToCompPass() {
  return std::make_unique<ConvertGpuToComp>();
}

} // namespace pmlc::conversion::comp
