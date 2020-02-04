// Copyright 2020, Intel Corporation

#include "pmlc/conversion/stdx_to_llvm/stdx_to_llvm.h"

#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/stdx/ir/ops.h"

using namespace mlir;  // NOLINT[build/namespaces]

namespace pmlc::conversion::stdx_to_llvm {

namespace stdx = dialect::stdx;

namespace {

// Creates a constant Op producing a value of `resultType` from an index-typed
// integer attribute.
Value createIndexAttrConstant(OpBuilder& builder, Location loc, Type resultType, int64_t value) {
  return builder.create<LLVM::ConstantOp>(loc, resultType, builder.getIntegerAttr(builder.getIndexType(), value));
}

// Base class for Standard to LLVM IR op conversions.  Matches the Op type
// provided as template argument.  Carries a reference to the LLVM dialect in
// case it is necessary for rewriters.
template <typename SourceOp>
class LLVMLegalizationPattern : public LLVMOpLowering {
 public:
  // Construct a conversion pattern.
  explicit LLVMLegalizationPattern(LLVM::LLVMDialect& dialect_, LLVMTypeConverter& lowering_)
      : LLVMOpLowering(SourceOp::getOperationName(), dialect_.getContext(), lowering_), dialect(dialect_) {}

  // Get the LLVM IR dialect.
  LLVM::LLVMDialect& getDialect() const { return dialect; }

  // Get the LLVM context.
  llvm::LLVMContext& getContext() const { return dialect.getLLVMContext(); }

  // Get the LLVM module in which the types are constructed.
  llvm::Module& getModule() const { return dialect.getLLVMModule(); }

  // Get the MLIR type wrapping the LLVM integer type whose bit width is defined
  // by the pointer size used in the LLVM module.
  LLVM::LLVMType getIndexType() const {
    return LLVM::LLVMType::getIntNTy(&dialect, getModule().getDataLayout().getPointerSizeInBits());
  }

  LLVM::LLVMType getVoidType() const { return LLVM::LLVMType::getVoidTy(&dialect); }

  // Get the MLIR type wrapping the LLVM i8* type.
  LLVM::LLVMType getVoidPtrType() const { return LLVM::LLVMType::getInt8PtrTy(&dialect); }

  // Create an LLVM IR pseudo-operation defining the given index constant.
  Value createIndexConstant(ConversionPatternRewriter& builder, Location loc, uint64_t value) const {
    return createIndexAttrConstant(builder, loc, getIndexType(), value);
  }

 protected:
  LLVM::LLVMDialect& dialect;
};

// Common base for load and store operations on MemRefs.  Restricts the match
// to supported MemRef types.  Provides functionality to emit code accessing a
// specific element of the underlying data buffer.
template <typename Derived>
struct LoadStoreOpLowering : public LLVMLegalizationPattern<Derived> {
  using LLVMLegalizationPattern<Derived>::LLVMLegalizationPattern;
  using Base = LoadStoreOpLowering<Derived>;

  // This is a strided getElementPtr variant that linearizes subscripts as:
  //   `base_offset + index_0 * stride_0 + ... + index_n * stride_n`.
  Value getStridedElementPtr(Location loc, Type elementTypePtr, Value descriptor, ArrayRef<Value> indices,
                             ArrayRef<int64_t> strides, int64_t offset, ConversionPatternRewriter& rewriter) const {
    MemRefDescriptor memRefDescriptor(descriptor);

    Value base = memRefDescriptor.alignedPtr(rewriter, loc);
    Value offsetValue = offset == MemRefType::getDynamicStrideOrOffset()
                            ? memRefDescriptor.offset(rewriter, loc)
                            : this->createIndexConstant(rewriter, loc, offset);

    for (int i = 0, e = indices.size(); i < e; ++i) {
      Value stride = strides[i] == MemRefType::getDynamicStrideOrOffset()
                         ? memRefDescriptor.stride(rewriter, loc, i)
                         : this->createIndexConstant(rewriter, loc, strides[i]);
      Value additionalOffset = rewriter.create<LLVM::MulOp>(loc, indices[i], stride);
      offsetValue = rewriter.create<LLVM::AddOp>(loc, offsetValue, additionalOffset);
    }
    return rewriter.create<LLVM::GEPOp>(loc, elementTypePtr, base, offsetValue);
  }

  Value getDataPtr(Location loc, MemRefType type, Value memRefDesc, ArrayRef<Value> indices,
                   ConversionPatternRewriter& rewriter, llvm::Module& module) const {
    LLVM::LLVMType ptrType = MemRefDescriptor(memRefDesc).getElementType();
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(type, strides, offset);
    assert(succeeded(successStrides) && "unexpected non-strided memref");
    (void)successStrides;
    return getStridedElementPtr(loc, ptrType, memRefDesc, indices, strides, offset, rewriter);
  }
};

struct SimpleAtomicMatch {
  LLVM::AtomicBinOp binOp;
  Value val;
};

template <typename OpType>
Optional<SimpleAtomicMatch> matchAtomicBinaryOp(stdx::AtomicRMWOp atomicOp, OpType op, LLVM::AtomicBinOp binOp) {
  Value lhs = op.lhs();
  Value rhs = op.rhs();
  if (lhs == atomicOp.getInductionVar() && rhs.getParentRegion() != &atomicOp.body()) {
    return SimpleAtomicMatch{binOp, rhs};
  }
  if (rhs == atomicOp.getInductionVar() && lhs.getParentRegion() != &atomicOp.body()) {
    return SimpleAtomicMatch{binOp, lhs};
  }
  return llvm::None;
}

Optional<SimpleAtomicMatch> matchAtomicCmpOp(stdx::AtomicRMWOp atomicOp, SelectOp selectOp, CmpIOp cmpOp,
                                             LLVM::AtomicBinOp binOp) {
  auto iv = atomicOp.getInductionVar();
  auto trueValue = selectOp.true_value();
  auto falseValue = selectOp.false_value();
  auto lhs = cmpOp.lhs();
  auto rhs = cmpOp.rhs();
  if (trueValue == iv && trueValue == lhs && falseValue == rhs && rhs.getParentRegion() != &atomicOp.body()) {
    return SimpleAtomicMatch{binOp, rhs};
  }
  return llvm::None;
}

Optional<SimpleAtomicMatch> matchAtomicSelectOp(stdx::AtomicRMWOp atomicOp, SelectOp op) {
  auto cmpOp = dyn_cast_or_null<CmpIOp>(op.condition().getDefiningOp());
  if (!cmpOp) {
    return llvm::None;
  }
  auto predicate = cmpOp.getPredicate();
  switch (predicate) {
    case CmpIPredicate::sgt:
      return matchAtomicCmpOp(atomicOp, op, cmpOp, LLVM::AtomicBinOp::max);
    case CmpIPredicate::ugt:
      return matchAtomicCmpOp(atomicOp, op, cmpOp, LLVM::AtomicBinOp::umax);
    case CmpIPredicate::slt:
      return matchAtomicCmpOp(atomicOp, op, cmpOp, LLVM::AtomicBinOp::min);
    case CmpIPredicate::ult:
      return matchAtomicCmpOp(atomicOp, op, cmpOp, LLVM::AtomicBinOp::umin);
    default:
      break;
  }
  return llvm::None;
}

Optional<SimpleAtomicMatch> matchSimpleAtomicOp(stdx::AtomicRMWOp atomicOp) {
  auto* body = atomicOp.getBody();
  auto* terminator = body->getTerminator();
  auto yieldOp = cast<stdx::AtomicRMWYieldOp>(terminator);
  if (yieldOp.result().getParentRegion() != &atomicOp.body()) {
    return SimpleAtomicMatch{LLVM::AtomicBinOp::xchg, yieldOp.result()};
  }
  auto defOp = yieldOp.result().getDefiningOp();
  return TypeSwitch<Operation*, Optional<SimpleAtomicMatch>>(defOp)
      .Case<AddIOp>([&](auto op) { return matchAtomicBinaryOp(atomicOp, op, LLVM::AtomicBinOp::add); })
      .Case<SubIOp>([&](auto op) { return matchAtomicBinaryOp(atomicOp, op, LLVM::AtomicBinOp::sub); })
      .Case<AndOp>([&](auto op) { return matchAtomicBinaryOp(atomicOp, op, LLVM::AtomicBinOp::_and); })
      // TODO(missing NotOp): nand: ~(*ptr & val)
      .Case<OrOp>([&](auto op) { return matchAtomicBinaryOp(atomicOp, op, LLVM::AtomicBinOp::_or); })
      .Case<XOrOp>([&](auto op) { return matchAtomicBinaryOp(atomicOp, op, LLVM::AtomicBinOp::_xor); })
      .Case<SelectOp>([&](auto op) { return matchAtomicSelectOp(atomicOp, op); })
      .Case<AddFOp>([&](auto op) { return matchAtomicBinaryOp(atomicOp, op, LLVM::AtomicBinOp::fadd); })
      .Case<SubFOp>([&](auto op) { return matchAtomicBinaryOp(atomicOp, op, LLVM::AtomicBinOp::fsub); })
      .Default([](Operation* op) { return llvm::None; });
}

struct AtomicRMWOpLowering : public LoadStoreOpLowering<stdx::AtomicRMWOp> {
  using Base::Base;

  PatternMatchResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                     ConversionPatternRewriter& rewriter) const override {
    auto atomicOp = cast<stdx::AtomicRMWOp>(op);
    auto simpleMatch = matchSimpleAtomicOp(atomicOp);
    if (!simpleMatch) {
      return matchFailure();
    }
    OperandAdaptor<stdx::AtomicRMWOp> adaptor(operands);
    auto type = atomicOp.getMemRefType();
    auto resultType = lowering.convertType(simpleMatch->val.getType());
    auto dataPtr = getDataPtr(op->getLoc(), type, adaptor.memref(), adaptor.indices(), rewriter, getModule());
    rewriter.create<LLVM::AtomicRMWOp>(op->getLoc(), resultType, simpleMatch->binOp, dataPtr, simpleMatch->val,
                                       LLVM::AtomicOrdering::acq_rel);
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

// Wrap a llvm.cmpxchg operation in a while loop so that the operation can be
// retried until it succeeds in atomically storing a new value into memory.
//
//      +---------------------------------+
//      |   <code before the AtomicRMWOp> |
//      |   <compute initial %iv value>   |
//      |   br loop(%iv)                  |
//      +---------------------------------+
//             |
//  -------|   |
//  |      v   v
//  |   +--------------------------------+
//  |   | loop(%iv):                     |
//  |   |   <body contents>              |
//  |   |   %pair = cmpxchg              |
//  |   |   %ok = %pair[0]               |
//  |   |   %new = %pair[1]              |
//  |   |   cond_br %ok, end, loop(%new) |
//  |   +--------------------------------+
//  |          |        |
//  |-----------        |
//                      v
//      +--------------------------------+
//      | end:                           |
//      |   <code after the AtomicRMWOp> |
//      +--------------------------------+
//
struct CmpXchgLowering : public LoadStoreOpLowering<stdx::AtomicRMWOp> {
  using Base::Base;

  PatternMatchResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                     ConversionPatternRewriter& rewriter) const override {
    auto atomicOp = cast<stdx::AtomicRMWOp>(op);
    auto simpleMatch = matchSimpleAtomicOp(atomicOp);
    if (simpleMatch) {
      return matchFailure();
    }

    // Split the block into initial and ending parts.
    auto* initBlock = rewriter.getInsertionBlock();
    auto initPosition = rewriter.getInsertionPoint();
    auto* endBlock = rewriter.splitBlock(initBlock, initPosition);

    // The body of the atomic_rmw op will form the basis of the loop block.
    auto& loopRegion = atomicOp.body();
    auto* loopBlock = &loopRegion.front();
    auto* terminator = loopBlock->getTerminator();
    auto yieldOp = cast<stdx::AtomicRMWYieldOp>(terminator);

    // Compute the initial IV and branch to the loop block.
    rewriter.setInsertionPointToEnd(initBlock);
    OperandAdaptor<stdx::AtomicRMWOp> adaptor(operands);
    auto type = atomicOp.getMemRefType();
    auto dataPtr = getDataPtr(op->getLoc(), type, adaptor.memref(), adaptor.indices(), rewriter, getModule());
    auto init = rewriter.create<LLVM::LoadOp>(op->getLoc(), dataPtr);
    SmallVector<Value, 1> brProperOperands;
    SmallVector<Block*, 1> brDestinations{loopBlock};
    SmallVector<Value, 1> brRegionOperands{init.res()};
    SmallVector<ValueRange, 1> brOperands{brRegionOperands};
    rewriter.create<LLVM::BrOp>(op->getLoc(), brProperOperands, brDestinations, brOperands);

    // Prepare the epilog of the loop block.
    rewriter.setInsertionPointToEnd(loopBlock);
    // Append the cmpxchg op to the end of the loop block.
    // TODO:
    //   %pair = llvm.cmpxchg %ptr, %loaded, %max acq_rel monotonic : !llvm.float
    //   %new_loaded = llvm.extractvalue %pair[0] : !llvm<"{ float, i1 }">
    //   %success = llvm.extractvalue %pair[1] : !llvm<"{ float, i1 }">
    auto boolType = LLVM::LLVMType::getInt1Ty(&getDialect());
    auto trueAttr = rewriter.getI64IntegerAttr(1);
    auto condOp = rewriter.create<LLVM::ConstantOp>(op->getLoc(), boolType, trueAttr);
    // Conditionally branch to the end or back to the loop depending on success.
    // TODO:
    //   llvm.cond_br %success, ^end, ^loop(%new_loaded : !llvm.float)
    SmallVector<Value, 1> condBrProperOperands{condOp.res()};
    SmallVector<Block*, 2> condBrDestinations{endBlock, loopBlock};
    SmallVector<Value, 1> condBrRegionOperands{yieldOp.result()};
    SmallVector<ValueRange, 2> condBrOperands{{}, condBrRegionOperands};
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(terminator, condBrProperOperands, condBrDestinations, condBrOperands);

    // Now move the body of the atomic_rmw op into the outer region just before
    // the ending block.
    rewriter.inlineRegionBefore(loopRegion, endBlock);

    // Remove the original atomic_rmw op
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

void populateStdXToLLVMConversionPatterns(LLVMTypeConverter& converter, OwningRewritePatternList& patterns) {
  patterns.insert<AtomicRMWOpLowering, CmpXchgLowering>(*converter.getDialect(), converter);
}

/// A pass converting MLIR operations into the LLVM IR dialect.
struct LLVMLoweringPass : public ModulePass<LLVMLoweringPass> {
  // Run the dialect converter on the module.
  void runOnModule() override {
    ModuleOp module = getModule();
    auto context = module.getContext();
    LLVM::ensureDistinctSuccessors(module);
    LLVMTypeConverter typeConverter(context);

    OwningRewritePatternList patterns;
    populateLoopToStdConversionPatterns(patterns, context);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    populateStdXToLLVMConversionPatterns(typeConverter, patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    if (failed(applyPartialConversion(module, target, patterns, &typeConverter))) {
      signalPassFailure();
    }
  }
};

static PassRegistration<LLVMLoweringPass> pass("convert-stdx-to-llvm", "Convert stdx to the LLVM dialect");

}  // namespace

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() { return std::make_unique<LLVMLoweringPass>(); }

}  // namespace pmlc::conversion::stdx_to_llvm
