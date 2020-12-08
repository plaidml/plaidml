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

struct valueComparator {
  bool operator()(const mlir::Value a, const mlir::Value b) const {
    return a.getAsOpaquePointer() < b.getAsOpaquePointer();
  }
};

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

  mlir::LogicalResult allocateDeviceMemory(
      mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value execEnv,
      std::vector<gpu::LaunchFuncOp> &ops,
      std::vector<std::vector<mlir::Value>> &newOperands,
      std::map<mlir::Value, mlir::Value, valueComparator> &bufferPool) const;
  mlir::LogicalResult
  createScheduleFuncOps(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        mlir::Value execEnv,
                        std::vector<gpu::LaunchFuncOp> &ops,
                        std::vector<std::vector<mlir::Value>> &newOperands,
                        std::vector<mlir::Value> &events) const;
  mlir::LogicalResult readDeviceMemory(
      mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value execEnv,
      std::vector<mlir::Value> &funcEvents,
      std::vector<std::vector<mlir::Value>> &newOperands,
      std::map<mlir::Value, mlir::Value, valueComparator> &bufferPool) const;
  mlir::LogicalResult deallocateDeviceMemory(
      mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value execEnv,
      std::map<mlir::Value, mlir::Value, valueComparator> &bufferPool) const;
  template <typename T>
  mlir::LogicalResult getConsecutiveOps(T op, std::vector<T> &ops) const;

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
  // Create execution environment.
  mlir::Value device = func.getArgument(0);
  auto execEnvOp =
      rewriter.create<comp::CreateExecEnv>(loc, execEnvType, device);
  mlir::Value execEnv = execEnvOp.getResult();
  // Collect consecutive gpu.launch_func Ops starting from current op.
  std::vector<gpu::LaunchFuncOp> launchOps;
  if (mlir::failed(getConsecutiveOps<gpu::LaunchFuncOp>(op, launchOps)))
    return mlir::failure();

  std::map<mlir::Value, mlir::Value, valueComparator> bufferPool;
  std::vector<std::vector<mlir::Value>> newOperands(launchOps.size());
  // Allocate memory on device for memory operands.
  if (mlir::failed(allocateDeviceMemory(rewriter, loc, execEnv, launchOps,
                                        newOperands, bufferPool)))
    return mlir::failure();
  std::vector<mlir::Value> funcEvents;
  // Create schedule functions within the same execution environment.
  if (mlir::failed(createScheduleFuncOps(rewriter, loc, execEnv, launchOps,
                                         newOperands, funcEvents)))
    return mlir::failure();
  // Read device memory back to host.
  if (mlir::failed(readDeviceMemory(rewriter, loc, execEnv, funcEvents,
                                    newOperands, bufferPool)))
    return mlir::failure();
  // Deallocate device memory.
  if (mlir::failed(deallocateDeviceMemory(rewriter, loc, execEnv, bufferPool)))
    return mlir::failure();
  // Destroy execution environment.
  rewriter.create<comp::DestroyExecEnv>(loc, execEnv);
  // Remove original launch operations.
  for (auto launchOp : launchOps) {
    rewriter.eraseOp(launchOp.getOperation());
  }
  return mlir::success();
}

template <typename T>
mlir::LogicalResult
RewriteLaunchFunc::getConsecutiveOps(T op, std::vector<T> &ops) const {
  // Collect consecutive ops of type T starting from a given op.
  auto operation = op.getOperation();
  while (operation != nullptr) {
    if (auto opT = mlir::dyn_cast<T>(operation)) {
      ops.push_back(opT);
      operation = operation->getNextNode();
    } else {
      return mlir::success();
    }
  }
  return mlir::success();
}

mlir::LogicalResult RewriteLaunchFunc::allocateDeviceMemory(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value execEnv,
    std::vector<gpu::LaunchFuncOp> &ops,
    std::vector<std::vector<mlir::Value>> &newOperands,
    std::map<mlir::Value, mlir::Value, valueComparator> &bufferPool) const {
  for (size_t i = 0; i < ops.size(); i++) {
    for (mlir::Value hostArg : ops[i].operands()) {
      if (bufferPool.count(hostArg) == 0) {
        if (auto memRefType = hostArg.getType().dyn_cast<mlir::MemRefType>()) {
          if (execEnvType.supportsMemorySpace(memRefType.getMemorySpace())) {
            break;
          }
          mlir::MemRefType newMemRefType =
              mlir::MemRefType::Builder(memRefType)
                  .setMemorySpace(execEnvType.getDefaultMemorySpace());
          auto allocOp =
              rewriter.create<comp::Alloc>(loc, newMemRefType, execEnv);
          auto deviceBuffer = allocOp.getResult();
          comp::EventType eventType = execEnvType.getEventType();
          mlir::Value event = rewriter.create<comp::ScheduleWrite>(
              loc, eventType, hostArg, deviceBuffer, execEnv,
              mlir::ValueRange{});
          rewriter.create<comp::Wait>(loc, event);
          bufferPool.insert({hostArg, deviceBuffer});
        }
      }
      newOperands[i].push_back(bufferPool[hostArg]);
    }
  }
  return mlir::success();
}

mlir::LogicalResult RewriteLaunchFunc::createScheduleFuncOps(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value execEnv,
    std::vector<gpu::LaunchFuncOp> &ops,
    std::vector<std::vector<mlir::Value>> &newOperands,
    std::vector<mlir::Value> &events) const {
  for (size_t i = 0; i < ops.size(); i++) {
    auto op = ops[i];
    // Find kernel that operation launches.
    mlir::SymbolRefAttr kernelSymbol = op.kernel();
    auto kernelOp = mlir::SymbolTable::lookupNearestSymbolFrom<gpu::GPUFuncOp>(
        op.getOperation(), kernelSymbol);
    if (!kernelOp)
      return mlir::failure();
    std::vector<mlir::Value> deps;
    for (size_t ie = 0; ie < events.size(); ie++) {
      for (auto newOperand : newOperands[i]) {
        if (std::find(newOperands[ie].begin(), newOperands[ie].end(),
                      newOperand) != newOperands[ie].end()) {
          deps.push_back(events[ie]);
          break;
        }
      }
    }
    if (deps.size() > 0) {
      rewriter.create<comp::Wait>(loc, deps);
    }
    auto scheduleFuncOp = rewriter.create<comp::ScheduleFunc>(
        loc, eventType, execEnv, mlir::ValueRange());
    events.push_back(scheduleFuncOp.getResult());
    // Add launch_func with new operands inside schedule_func.
    mlir::PatternRewriter::InsertionGuard insertionGuard(rewriter);
    rewriter.createBlock(&scheduleFuncOp.body(), scheduleFuncOp.body().end());
    if (newOperands[i].size() == 0) {
      rewriter.create<gpu::LaunchFuncOp>(
          loc, kernelOp, op.getGridSizeOperandValues(),
          op.getBlockSizeOperandValues(), op.operands());
    } else {
      rewriter.create<gpu::LaunchFuncOp>(
          loc, kernelOp, op.getGridSizeOperandValues(),
          op.getBlockSizeOperandValues(), newOperands[i]);
    }
    rewriter.create<comp::ScheduleEnd>(loc);
  }
  return mlir::success();
}

mlir::LogicalResult RewriteLaunchFunc::readDeviceMemory(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value execEnv,
    std::vector<mlir::Value> &funcEvents,
    std::vector<std::vector<mlir::Value>> &newOperands,
    std::map<mlir::Value, mlir::Value, valueComparator> &bufferPool) const {
  std::vector<mlir::Value> readEvents;
  for (auto pair : bufferPool) {
    mlir::Value host = pair.first;
    mlir::Value device = pair.second;
    std::vector<mlir::Value> deps;
    for (size_t i = 0; i < newOperands.size(); i++) {
      if (std::find(newOperands[i].begin(), newOperands[i].end(), device) !=
          newOperands[i].end()) {
        deps.push_back(funcEvents[i]);
      }
    }
    auto readOp = rewriter.create<comp::ScheduleRead>(loc, eventType, host,
                                                      device, execEnv, deps);
    readEvents.push_back(readOp.getResult());
  }
  // Wait for all read operations to finish or dependency if no reads.
  if (!readEvents.empty())
    rewriter.create<comp::Wait>(loc, readEvents);
  else
    rewriter.create<comp::Wait>(loc, funcEvents);

  return mlir::success();
}

mlir::LogicalResult RewriteLaunchFunc::deallocateDeviceMemory(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value execEnv,
    std::map<mlir::Value, mlir::Value, valueComparator> &bufferPool) const {
  for (auto pair : bufferPool) {
    mlir::Value device = pair.second;
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
