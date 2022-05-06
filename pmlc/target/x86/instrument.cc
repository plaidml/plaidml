// Copyright 2021 Intel Corporation

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "pmlc/target/x86/pass_detail.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace {

static constexpr StringRef kInstrument = "plaidml_rt_instrument";
static constexpr StringRef kMlirCifaceInstrument =
    "_mlir_ciface_plaidml_rt_instrument";

static std::string printLocation(Location loc) {
  return TypeSwitch<Location, std::string>(loc)
      .Case<CallSiteLoc>([&](CallSiteLoc loc) { return "CallSiteLoc"; })
      .Case<FileLineColLoc>(
          [&](FileLineColLoc loc) { return "FileLineColLoc"; })
      .Case<FusedLoc>([&](FusedLoc loc) {
        std::string str;
        llvm::raw_string_ostream os(str);
        interleave(
            loc.getLocations(), [&](Location loc) { os << printLocation(loc); },
            [&]() { os << "|"; });
        return os.str();
      })
      .Case<NameLoc>([&](NameLoc loc) { return loc.getName().str(); })
      .Case<OpaqueLoc>([&](OpaqueLoc loc) { return "opaque"; })
      .Case<UnknownLoc>([&](UnknownLoc loc) { return "unknown"; });
}

static FlatSymbolRefAttr lookupOrCreateFn(ModuleOp module, StringRef name,
                                          TypeRange argTypes,
                                          TypeRange resultTypes) {
  OpBuilder builder(module.getBodyRegion());
  if (auto fn = module.lookupSymbol<func::FuncOp>(name))
    return FlatSymbolRefAttr::get(fn);

  FunctionType funcType = builder.getFunctionType(argTypes, resultTypes);
  auto fn = builder.create<func::FuncOp>(module.getLoc(), name, funcType,
                                   builder.getStringAttr("private"));
  return FlatSymbolRefAttr::get(fn);
}

struct ProfileKernelsPass : public ProfileKernelsBase<ProfileKernelsPass> {
  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    Type i64Type = IntegerType::get(context, 64);
    FlatSymbolRefAttr func =
        lookupOrCreateFn(module, kInstrument, {i64Type, i64Type}, TypeRange{});

    int64_t id = 0;
    module.walk<WalkOrder::PreOrder>([&](AffineParallelOp op) {
      Location loc = op->getLoc();
      OpBuilder builder(op);
      Value idValue = builder.create<arith::ConstantIntOp>(loc, id++, 64);
      Value tagZero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
      auto call = builder.create<LLVM::CallOp>(loc, TypeRange{}, func,
                                         ValueRange{idValue, tagZero});
      builder.setInsertionPointAfter(op);
      Value tagOne = builder.create<arith::ConstantIntOp>(loc, 1, 64);
      builder.create<LLVM::CallOp>(loc, TypeRange{}, func,
                             ValueRange{idValue, tagOne});

      return WalkResult::skip();
    });
  }
};

struct ProfileLinkingPass : public ProfileLinkingBase<ProfileLinkingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    LLVM::LLVMFuncOp stub = module.lookupSymbol<LLVM::LLVMFuncOp>(kInstrument);
    if (!stub)
      return;

    LLVM::LLVMFuncOp decl =
        module.lookupSymbol<LLVM::LLVMFuncOp>(kMlirCifaceInstrument);
    if (!decl)
      return;

    rewriteFunc(decl, /*rewriteBody=*/false);
    rewriteFunc(stub, /*rewriteBody=*/true);

    int64_t id = 0;
    MLIRContext *context = module.getContext();
    module.walk([&](LLVM::CallOp op) {
      if (op.getCallee().getValue() != kInstrument)
        return;

      Location loc = op.getLoc();
      OpBuilder builder(op);
      std::string locSymbol = llvm::formatv("__loc_str_{0}", id++).str();
      std::string locStr = printLocation(loc);
      StringRef locStrRef = StringRef(locStr.c_str(), locStr.size() + 1);
      Value globalStr =
          getOrCreateGlobalString(loc, builder, locSymbol, locStrRef, module);
      builder.create<LLVM::CallOp>(
          loc, ArrayRef<Type>{}, *op.getCallee(),
          ArrayRef<Value>{op.getOperand(0), op.getOperand(1), globalStr});
      op.erase();
    });
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type =
          LLVM::LLVMArrayType::get(builder.getIntegerType(8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(builder.getIntegerType(8)), globalPtr,
        ArrayRef<Value>({cst0, cst0}));
  }

  static void rewriteFunc(LLVM::LLVMFuncOp oldFunc, bool rewriteBody) {
    OpBuilder builder(oldFunc);
    Location loc = oldFunc.getLoc();
    MLIRContext *context = builder.getContext();
    auto voidTy = LLVM::LLVMVoidType::get(context);
    Type i64Ty = builder.getI64Type();
    auto ptrTy = LLVM::LLVMPointerType::get(builder.getIntegerType(8));
    auto funcType = LLVM::LLVMFunctionType::get(voidTy, {i64Ty, i64Ty, ptrTy},
                                                /*isVarArg=*/false);
    auto func =
        builder.create<LLVM::LLVMFuncOp>(loc, oldFunc.getName(), funcType);
    if (rewriteBody) {
      Block *block = func.addEntryBlock();
      builder.setInsertionPointToStart(block);
      builder.create<LLVM::CallOp>(loc, ArrayRef<Type>{}, kMlirCifaceInstrument,
                                   func.getArguments());
      builder.create<LLVM::ReturnOp>(loc, ValueRange{});
    }
    oldFunc.erase();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createProfileKernelsPass() {
  return std::make_unique<ProfileKernelsPass>();
}

std::unique_ptr<mlir::Pass> createProfileLinkingPass() {
  return std::make_unique<ProfileLinkingPass>();
}

} // namespace pmlc::target::x86
