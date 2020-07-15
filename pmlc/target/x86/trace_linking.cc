// Copyright 2020 Intel Corporation

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace {

struct TraceLinkingPass : public TraceLinkingBase<TraceLinkingPass> {
  void runOnOperation() override {
    getOperation().walk([](LLVM::LLVMFuncOp op) {
      if (!op.getAttrOfType<UnitAttr>("trace")) {
        return;
      }
      if (!op.empty()) {
        return;
      }
      auto loc = op.getLoc();
      auto context = op.getContext();
      OpBuilder builder(context);
      auto llvmDialect = context->getRegisteredDialect<LLVM::LLVMDialect>();
      auto module = op.getParentOfType<ModuleOp>();
      auto traceRef = getOrInsertTrace(loc, builder, module, llvmDialect);
      auto block = op.addEntryBlock();
      builder.setInsertionPointToStart(block);
      auto id = op.getAttrOfType<IntegerAttr>("id").getValue().getZExtValue();
      auto msgStr = op.getAttrOfType<StringAttr>("msg").getValue().str();
      auto msg = StringRef(msgStr.c_str(), msgStr.size() + 1);
      auto msgSymbol = llvm::formatv("__trace_msg_{0}", id).str();
      auto msgValue = getOrCreateGlobalString(loc, builder, msgSymbol, msg,
                                              module, llvmDialect);
      builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{}, traceRef,
                                   ArrayRef<Value>{msgValue});
      builder.create<LLVM::ReturnOp>(loc, ArrayRef<Value>{});
      op.removeAttr("id");
      op.removeAttr("msg");
      op.removeAttr("trace");
    });
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module,
                                       LLVM::LLVMDialect *llvmDialect) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMType::getArrayTy(
          LLVM::LLVMType::getInt8Ty(llvmDialect), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, LLVM::LLVMType::getInt64Ty(llvmDialect),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect), globalPtr,
        ArrayRef<Value>({cst0, cst0}));
  }

  static FlatSymbolRefAttr getOrInsertTrace(Location loc, OpBuilder &builder,
                                            ModuleOp module,
                                            LLVM::LLVMDialect *llvmDialect) {
    const char *symbol = "plaidml_rt_trace";
    auto context = module.getContext();
    if (module.lookupSymbol(symbol)) {
      return SymbolRefAttr::get(symbol, context);
    }
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto voidTy = LLVM::LLVMType::getVoidTy(llvmDialect);
    auto msgTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto funcType = LLVM::LLVMType::getFunctionTy(voidTy, {msgTy}, false);
    builder.create<LLVM::LLVMFuncOp>(loc, symbol, funcType);
    return SymbolRefAttr::get(symbol, context);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createTraceLinkingPass() {
  return std::make_unique<TraceLinkingPass>();
}

} // namespace pmlc::target::x86
