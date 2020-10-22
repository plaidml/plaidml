// Copyright 2020, Intel Corporation
#include <memory>
#include <vector>

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/gpu_to_comp/pass_detail.h"
#include "pmlc/conversion/gpu_to_comp/passes.h"
#include "pmlc/dialect/comp/ir/dialect.h"

namespace pmlc::conversion::gpu_to_comp {

namespace comp = pmlc::dialect::comp;
namespace gpu = mlir::gpu;

namespace {

/// Rewrites gpu.launch_func operation to be contained inside
/// comp.schedule_func. Creates new execution environment before function,
/// then allocates required memory on device. After scheduling function
/// creates memory reads and wait for completion. Finally deallocates
/// device memory and destroys execution environment.
struct RewriteLaunchFunc : public mlir::OpRewritePattern<gpu::LaunchFuncOp> {
  RewriteLaunchFunc(mlir::MLIRContext *context, comp::ExecEnvType execEnvType)
      : mlir::OpRewritePattern<gpu::LaunchFuncOp>(context),
        execEnvType(execEnvType),
        eventType(comp::EventType::get(context, execEnvType.getRuntime())) {}

  mlir::LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp op,
                  mlir::PatternRewriter &rewriter) const override;

  mlir::LogicalResult
  allocateDeviceMemory(mlir::PatternRewriter &rewriter, mlir::Location location,
                       mlir::Value execEnv, const mlir::ValueRange &host,
                       std::vector<mlir::Value> &device) const;
  mlir::LogicalResult createScheduleFuncOp(mlir::PatternRewriter &rewriter,
                                           mlir::Location loc,
                                           mlir::Value execEnv,
                                           gpu::LaunchFuncOp op,
                                           const mlir::ValueRange &operands,
                                           mlir::Value &event) const;
  mlir::LogicalResult readDeviceMemory(mlir::PatternRewriter &rewriter,
                                       mlir::Location loc, mlir::Value execEnv,
                                       const mlir::ValueRange &hostArgs,
                                       const mlir::ValueRange &deviceArgs,
                                       mlir::Value dependency) const;
  mlir::LogicalResult
  deallocateDeviceMemory(mlir::PatternRewriter &rewriter, mlir::Location loc,
                         mlir::Value execEnv, const mlir::ValueRange &hostArgs,
                         const mlir::ValueRange &deviceArgs) const;

  comp::ExecEnvType execEnvType;
  comp::EventType eventType;
};

/// Converts memrefs in gpu function signature to reside in memory
/// spaces supported by execution environment.
struct ConvertGpuFunc : public mlir::OpConversionPattern<gpu::GPUFuncOp> {
  ConvertGpuFunc(mlir::MLIRContext *context, comp::ExecEnvType execEnvType)
      : mlir::OpConversionPattern<gpu::GPUFuncOp>(context),
        execEnvType(execEnvType) {}

  mlir::LogicalResult
  matchAndRewrite(gpu::GPUFuncOp op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override;

  comp::ExecEnvType execEnvType;
};

// Adds device parameters to entrypoint functions.
struct ConvertFunc final : public mlir::OpConversionPattern<mlir::FuncOp> {
  using mlir::OpConversionPattern<mlir::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::FuncOp op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final;
};

class ConvertGpuToComp : public ConvertGpuToCompBase<ConvertGpuToComp> {
public:
  ConvertGpuToComp() = default;
  ConvertGpuToComp(comp::ExecEnvRuntime runtime, unsigned memorySpace) {
    this->execEnvRuntime = static_cast<unsigned>(runtime);
    this->execEnvMemorySpace = memorySpace;
  }
  void runOnOperation();
};

mlir::LogicalResult
RewriteLaunchFunc::matchAndRewrite(gpu::LaunchFuncOp op,
                                   mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op.getLoc();
  // Look up the containing function.
  auto func = op.getParentOfType<mlir::FuncOp>();
  if (!func) {
    return mlir::failure();
  }
  auto funcTy = func.getType();
  if (!funcTy.getNumInputs() || !funcTy.getInput(0).isa<comp::DeviceType>()) {
    func.emitError(
        "Expected containing function to supply an execution device");
    return mlir::failure();
  }
  mlir::Value device = func.getArgument(0);
  // Create execution environment.
  auto execEnvOp =
      rewriter.create<comp::CreateExecEnv>(loc, execEnvType, device);
  mlir::Value execEnv = execEnvOp.getResult();
  // Allocate memory on device for memory operands.
  std::vector<mlir::Value> newOperands;
  if (mlir::failed(allocateDeviceMemory(rewriter, loc, execEnv, op.operands(),
                                        newOperands)))
    return mlir::failure();
  mlir::Value funcEvent;
  if (mlir::failed(createScheduleFuncOp(rewriter, loc, execEnv, op, newOperands,
                                        funcEvent)))
    return mlir::failure();
  // Read device memory back to host.
  if (mlir::failed(readDeviceMemory(rewriter, loc, execEnv, op.operands(),
                                    newOperands, funcEvent)))
    return mlir::failure();
  // Deallocate device memory.
  if (mlir::failed(deallocateDeviceMemory(rewriter, loc, execEnv, op.operands(),
                                          newOperands)))
    return mlir::failure();
  // Destroy execution environment.
  rewriter.create<comp::DestroyExecEnv>(loc, execEnv);
  // Remove original launch operation.
  rewriter.eraseOp(op.getOperation());

  return mlir::success();
}

mlir::LogicalResult RewriteLaunchFunc::allocateDeviceMemory(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value execEnv,
    const mlir::ValueRange &host, std::vector<mlir::Value> &device) const {
  device.reserve(host.size());
  for (mlir::Value hostArg : host) {
    mlir::Value newArg = hostArg;

    if (auto memRefType = hostArg.getType().dyn_cast<mlir::MemRefType>()) {
      if (!execEnvType.supportsMemorySpace(memRefType.getMemorySpace())) {
        mlir::MemRefType newMemRefType =
            mlir::MemRefType::Builder(memRefType)
                .setMemorySpace(execEnvType.getDefaultMemorySpace());
        auto allocOp =
            rewriter.create<comp::Alloc>(loc, newMemRefType, execEnv, hostArg);
        newArg = allocOp.getResult();
      }
    }

    device.push_back(newArg);
  }
  return mlir::success();
}

mlir::LogicalResult RewriteLaunchFunc::createScheduleFuncOp(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value execEnv,
    gpu::LaunchFuncOp op, const mlir::ValueRange &operands,
    mlir::Value &event) const {
  // Find kernel that operation launches.
  mlir::SymbolRefAttr kernelSymbol = op.kernel();
  auto kernelOp = mlir::SymbolTable::lookupNearestSymbolFrom<gpu::GPUFuncOp>(
      op.getOperation(), kernelSymbol);
  if (!kernelOp)
    return mlir::failure();

  auto scheduleFuncOp = rewriter.create<comp::ScheduleFunc>(
      loc, eventType, execEnv, mlir::ValueRange());
  event = scheduleFuncOp.getResult();

  // Add launch_func with new operands inside schedule_func.
  mlir::PatternRewriter::InsertionGuard insertionGuard(rewriter);
  rewriter.createBlock(&scheduleFuncOp.body(), scheduleFuncOp.body().end());
  rewriter.create<gpu::LaunchFuncOp>(loc, kernelOp,
                                     op.getGridSizeOperandValues(),
                                     op.getBlockSizeOperandValues(), operands);
  rewriter.create<comp::ScheduleEnd>(loc);

  return mlir::success();
}

mlir::LogicalResult RewriteLaunchFunc::readDeviceMemory(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value execEnv,
    const mlir::ValueRange &hostArgs, const mlir::ValueRange &deviceArgs,
    mlir::Value dependency) const {
  std::vector<mlir::Value> readEvents;
  for (size_t argIdx = 0; argIdx < hostArgs.size(); ++argIdx) {
    mlir::Value host = hostArgs[argIdx];
    mlir::Value device = deviceArgs[argIdx];
    if (host == device)
      continue;
    auto readOp = rewriter.create<comp::ScheduleRead>(
        loc, eventType, host, device, execEnv, dependency);
    readEvents.push_back(readOp.getResult());
  }
  // Wait for all read operations to finish or dependency if no reads.
  if (!readEvents.empty())
    rewriter.create<comp::Wait>(loc, readEvents);
  else
    rewriter.create<comp::Wait>(loc, dependency);

  return mlir::success();
}

mlir::LogicalResult RewriteLaunchFunc::deallocateDeviceMemory(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value execEnv,
    const mlir::ValueRange &hostArgs,
    const mlir::ValueRange &deviceArgs) const {
  for (size_t argIdx = 0; argIdx < hostArgs.size(); ++argIdx) {
    mlir::Value host = hostArgs[argIdx];
    mlir::Value device = deviceArgs[argIdx];
    if (host == device)
      continue;
    rewriter.create<comp::Dealloc>(loc, execEnv, device);
  }
  return mlir::success();
}

mlir::LogicalResult ConvertGpuFunc::matchAndRewrite(
    gpu::GPUFuncOp op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  if (!op.isKernel())
    return mlir::failure();

  mlir::TypeConverter converter;
  // Default pass-through conversion.
  converter.addConversion([](mlir::Type type) { return type; });
  // Change memrefs memory space if not supported by execution environment.
  converter.addConversion([&](mlir::MemRefType type) -> mlir::MemRefType {
    if (execEnvType.supportsMemorySpace(type.getMemorySpace()))
      return type;
    return mlir::MemRefType::Builder(type).setMemorySpace(
        execEnvType.getDefaultMemorySpace());
  });

  mlir::TypeConverter::SignatureConversion signatureConversion(
      op.getNumArguments());

  if (mlir::failed(converter.convertSignatureArgs(op.getArgumentTypes(),
                                                  signatureConversion)))
    return mlir::failure();

  if (mlir::failed(rewriter.convertRegionTypes(&op.body(), converter,
                                               &signatureConversion)))
    return mlir::failure();

  rewriter.updateRootInPlace(op, [&] {
    op.setType(
        rewriter.getFunctionType(signatureConversion.getConvertedTypes(), {}));
  });

  return mlir::success();
}

mlir::LogicalResult
ConvertFunc::matchAndRewrite(mlir::FuncOp op,
                             mlir::ArrayRef<mlir::Value> operands,
                             mlir::ConversionPatternRewriter &rewriter) const {
  if (op.isExternal() ||
      (op.getNumArguments() > 0 &&
       op.getArgument(0).getType().isa<comp::DeviceType>())) {
    return mlir::failure();
  }

  auto oldFuncTy = op.getType();
  auto deviceTy = rewriter.getType<comp::DeviceType>();
  std::vector<mlir::Type> inputs{deviceTy};
  inputs.insert(inputs.end(), oldFuncTy.getInputs().begin(),
                oldFuncTy.getInputs().end());
  auto newFuncTy = rewriter.getFunctionType(inputs, oldFuncTy.getResults());

  rewriter.updateRootInPlace(op, [&] {
    op.setType(newFuncTy);
    op.front().insertArgument(0u, deviceTy);
  });

  return mlir::success();
}

void ConvertGpuToComp::runOnOperation() {
  auto runtime = static_cast<comp::ExecEnvRuntime>(execEnvRuntime.getValue());
  unsigned memorySpace = execEnvMemorySpace.getValue();
  auto execEnvType =
      comp::ExecEnvType::get(&getContext(), runtime, /*tag=*/0, {memorySpace});

  // Setup conversion target.
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<comp::COMPDialect>();
  target.addLegalDialect<gpu::GPUDialect>();
  target.addDynamicallyLegalOp<gpu::LaunchFuncOp>(
      [](gpu::LaunchFuncOp op) -> bool {
        auto parent = op.getParentOp();
        return parent && mlir::isa<comp::ScheduleFunc>(parent);
      });
  target.addDynamicallyLegalOp<gpu::GPUFuncOp>([=](gpu::GPUFuncOp op) -> bool {
    if (!op.isKernel())
      return true;

    for (auto arg : op.getArguments()) {
      if (auto memRefType =
              arg.getType().dyn_cast_or_null<mlir::MemRefType>()) {
        if (!execEnvType.supportsMemorySpace(memRefType.getMemorySpace()))
          return false;
      }
    }
    return true;
  });
  target.addDynamicallyLegalOp<mlir::FuncOp>([](mlir::FuncOp op) -> bool {
    return op.isExternal() ||
           ((op.getNumArguments() > 0) &&
            op.getArgument(0).getType().isa<comp::DeviceType>());
  });
  target.addIllegalOp<gpu::LaunchOp>();

  // Setup rewrite patterns.
  mlir::OwningRewritePatternList patterns;
  populateGpuToCompPatterns(&getContext(), execEnvType, patterns);

  // Run the conversion.
  if (mlir::failed(
          mlir::applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();
}

} // namespace

void populateGpuToCompPatterns(mlir::MLIRContext *context,
                               const comp::ExecEnvType &execEnvType,
                               mlir::OwningRewritePatternList &patterns) {
  patterns.insert<RewriteLaunchFunc>(context, execEnvType);
  patterns.insert<ConvertGpuFunc>(context, execEnvType);
  patterns.insert<ConvertFunc>(context);
}

std::unique_ptr<mlir::Pass> createConvertGpuToCompPass() {
  return std::make_unique<ConvertGpuToComp>();
}

std::unique_ptr<mlir::Pass>
createConvertGpuToCompPass(comp::ExecEnvRuntime runtime, unsigned memorySpace) {
  return std::make_unique<ConvertGpuToComp>(runtime, memorySpace);
}

} // namespace pmlc::conversion::gpu_to_comp
