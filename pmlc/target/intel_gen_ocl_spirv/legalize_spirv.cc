// Copyright 2020, Intel Corporation

#include <vector>

#include "pmlc/target/intel_gen_ocl_spirv/pass_detail.h"
#include "pmlc/target/intel_gen_ocl_spirv/passes.h"

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace pmlc::target::intel_gen_ocl_spirv {
namespace spirv = mlir::spirv;

namespace {

/// Performs conversion of builtins with i32 type to i64.
struct BuiltInConversion final
    : public mlir::OpConversionPattern<spirv::GlobalVariableOp> {
  using mlir::OpConversionPattern<spirv::GlobalVariableOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(spirv::GlobalVariableOp op,
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override;
};

/// Extracts builtin this global declares.
/// If it doesn't declare builtin returns llvm::None.
mlir::Optional<spirv::BuiltIn> getGlobalBuiltIn(spirv::GlobalVariableOp op);
/// Returns true if this is global variable declaring builtin with
/// either i32 or vector of i32 type.
bool isI32BuiltIn(spirv::GlobalVariableOp op);

class IntelGenOclLegalizeSpirvPass final
    : public IntelGenOclLegalizeSpirvBase<IntelGenOclLegalizeSpirvPass> {
  void runOnOperation() {
    mlir::OwningRewritePatternList patterns;
    patterns.insert<BuiltInConversion>(&getContext());

    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<spirv::SPIRVDialect>();
    target.addDynamicallyLegalOp<spirv::GlobalVariableOp>(
        [](spirv::GlobalVariableOp op) { return !isI32BuiltIn(op); });

    spirv::ModuleOp module = getOperation();
    if (mlir::failed(
            applyFullConversion(module.getOperation(), target, patterns)))
      signalPassFailure();
  }
};

mlir::Optional<spirv::BuiltIn> getGlobalBuiltIn(spirv::GlobalVariableOp op) {
  mlir::StringAttr builtinAttr = op.getAttrOfType<mlir::StringAttr>(
      spirv::SPIRVDialect::getAttributeName(spirv::Decoration::BuiltIn));
  if (!builtinAttr)
    return llvm::None;
  return spirv::symbolizeBuiltIn(builtinAttr.getValue());
}

bool isI32BuiltIn(spirv::GlobalVariableOp op) {
  mlir::Optional<spirv::BuiltIn> builtIn = getGlobalBuiltIn(op);
  if (!builtIn.hasValue())
    return false;
  spirv::PointerType pointerType = op.type().dyn_cast<spirv::PointerType>();
  if (!pointerType)
    return false;
  mlir::Type elementType = pointerType.getPointeeType();
  if (auto vecType = elementType.dyn_cast<mlir::VectorType>())
    elementType = vecType.getElementType();
  return elementType.isInteger(32);
}

mlir::LogicalResult BuiltInConversion::matchAndRewrite(
    spirv::GlobalVariableOp op, mlir::ArrayRef<mlir::Value> operands,
    mlir::ConversionPatternRewriter &rewriter) const {
  // ==========================================================================
  // Match stage.
  // Check that this is builtin with incorrect type.
  if (!isI32BuiltIn(op))
    return mlir::failure();
  // Check user chain is address_of -> load.
  // Gather address_of users for rewrite stage.
  std::vector<spirv::AddressOfOp> addressOps;
  mlir::Optional<mlir::SymbolTable::UseRange> optSymbolUses =
      op.getSymbolUses(op.getParentOp());
  if (!optSymbolUses.hasValue())
    return mlir::failure();
  for (mlir::SymbolTable::SymbolUse use : optSymbolUses.getValue()) {
    mlir::Operation *user = use.getUser();
    if (mlir::isa<spirv::EntryPointOp>(user))
      continue;
    auto addressOp = mlir::dyn_cast<spirv::AddressOfOp>(user);
    if (!addressOp)
      return mlir::failure();
    addressOps.push_back(addressOp);
    for (mlir::Operation *usersUser : user->getUsers()) {
      auto loadOp = mlir::dyn_cast<spirv::LoadOp>(usersUser);
      if (!loadOp)
        return mlir::failure();
    }
  }
  // ==========================================================================
  // Rewrite stage.
  // Create new builtin with correct type.
  spirv::BuiltIn builtIn = getGlobalBuiltIn(op).getValue();
  mlir::StringRef name = op.sym_name();

  spirv::PointerType pointerType = op.type().cast<spirv::PointerType>();
  mlir::Type newInnerType;
  // Builtin can either be vector of i32 or i32, convert inner type to i64.
  if (auto vecType = pointerType.getPointeeType().dyn_cast<mlir::VectorType>())
    newInnerType =
        mlir::VectorType::get(vecType.getShape(), rewriter.getIntegerType(64));
  else
    newInnerType = rewriter.getIntegerType(64);
  auto newPointerType =
      spirv::PointerType::get(newInnerType, pointerType.getStorageClass());

  // Replacement split into create and erase to get access to new
  // global variable op.
  auto newGlobal = rewriter.create<spirv::GlobalVariableOp>(
      op.getLoc(), newPointerType, name, builtIn);
  rewriter.eraseOp(op.getOperation());
  // Update users chain.
  for (spirv::AddressOfOp addressOp : addressOps) {
    // Gather load ops using this address.
    // If done after replacement there will be no users, due to lazy rewrite.
    std::vector<spirv::LoadOp> loadOps;
    for (mlir::Operation *addrUser : addressOp.getOperation()->getUsers())
      loadOps.push_back(mlir::cast<spirv::LoadOp>(addrUser));

    // Replacement split into create and replace to get new value.
    // Conversion infrastructure will perform actual replacements only after
    // all patterns finish.
    rewriter.setInsertionPoint(addressOp.getOperation());
    auto newAddr =
        rewriter.create<spirv::AddressOfOp>(addressOp.getLoc(), newGlobal);
    rewriter.replaceOp(addressOp, mlir::ValueRange{newAddr.getResult()});

    for (spirv::LoadOp loadOp : loadOps) {
      rewriter.setInsertionPoint(loadOp.getOperation());
      mlir::Value load = rewriter.create<spirv::LoadOp>(
          loadOp.getLoc(), newAddr, loadOp.memory_accessAttr(),
          loadOp.alignmentAttr());
      // Insert cast back to previous i32 semantic after load.
      rewriter.replaceOpWithNewOp<spirv::UConvertOp>(loadOp, loadOp.getType(),
                                                     load);
    }
  }
  return mlir::success();
}

} // namespace

std::unique_ptr<mlir::Pass> createLegalizeSpirvPass() {
  return std::make_unique<IntelGenOclLegalizeSpirvPass>();
}

} // namespace pmlc::target::intel_gen_ocl_spirv
