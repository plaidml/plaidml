// Copyright 2021, Intel Corporation

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

#include "pmlc/conversion/linalg_to_pxa/pass_detail.h"
#include "pmlc/conversion/tile_to_pxa/pass_detail.h"
#include "pmlc/dialect/pxa/analysis/uses.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/util/matchers.h"
#include "pmlc/util/util.h"

namespace pmlc::conversion::linalg_to_pxa {

namespace layer = dialect::layer;
namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;
namespace tile = dialect::tile;

using namespace mlir; // NOLINT

namespace {

static RankedTensorType getRankedTensorType(Type type) {
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
    return rankedTensorType;
  }
  return RankedTensorType::get({}, type);
}

struct BufferAllocator {
  Value resultMemRef;
  MemRefType memRefType;
  Type elementType;

  BufferAllocator(OpBuilder &builder, Operation *op, Type resultType) {
    // Gather some basic info
    Location loc = op->getLoc();

    if (resultType.isa<RankedTensorType>()) {
      LinalgToPXATypeConverter typeConverter;
      auto rankedTensorType = getRankedTensorType(resultType);
      elementType =
          typeConverter.convertType(rankedTensorType.getElementType());
      ArrayRef<int64_t> originalShape = rankedTensorType.getShape();
      auto shape = llvm::to_vector<8>(originalShape);
      // Make an allocation for the output
      memRefType = MemRefType::get(shape, elementType);
    } else if (resultType.isa<MemRefType>()) {
      memRefType = resultType.cast<MemRefType>();
      elementType = memRefType.getElementType();
    }
    resultMemRef = builder.create<memref::AllocOp>(loc, memRefType);
  }
};

// To create a new reduce op, we need to extract the aggregation kind, the
// scalar for op, and the related operation if there are multiple operations for
// the reduction. bufElem is the element value in the target buffer. So the
// scalar should be the other operand of the reduceOp
struct ReductionInfo {
  ReductionInfo(Operation *op, Value bufElem) : relatedOp(nullptr) {
    Value lhs, rhs;
    Value cond, trueValue, falseValue;
    if (matchPattern(op,
                     m_Op<arith::AddIOp>(m_Capture(&lhs), m_Capture(&rhs)))) {
      agg = arith::AtomicRMWKind::addi;
    } else if (matchPattern(
                   op, m_Op<arith::AddFOp>(m_Capture(&lhs), m_Capture(&rhs)))) {
      agg = arith::AtomicRMWKind::addf;
    } else if (matchPattern(
                   op, m_Op<arith::MulIOp>(m_Capture(&lhs), m_Capture(&rhs)))) {
      agg = arith::AtomicRMWKind::muli;
    } else if (matchPattern(
                   op, m_Op<arith::MulFOp>(m_Capture(&lhs), m_Capture(&rhs)))) {
      agg = arith::AtomicRMWKind::mulf;
    } else if (matchPattern(
                   op,
                   m_Op<arith::SelectOp>(
                       m_Capture(&cond, m_Op<arith::CmpIOp>(m_Capture(&lhs),
                                                            m_Capture(&rhs))),
                       m_Capture(&trueValue), m_Capture(&falseValue)))) {
      if (lhs == trueValue && rhs == falseValue) {
        relatedOp = cond.getDefiningOp();
        arith::CmpIPredicate intPred =
            cast<arith::CmpIOp>(relatedOp).getPredicate();
        switch (intPred) {
        case arith::CmpIPredicate::sgt:
        case arith::CmpIPredicate::sge:
          agg = arith::AtomicRMWKind::maxs;
          break;
        case arith::CmpIPredicate::ugt:
        case arith::CmpIPredicate::uge:
          agg = arith::AtomicRMWKind::maxu;
          break;
        case arith::CmpIPredicate::slt:
        case arith::CmpIPredicate::sle:
          agg = arith::AtomicRMWKind::mins;
          break;
        case arith::CmpIPredicate::ult:
        case arith::CmpIPredicate::ule:
          agg = arith::AtomicRMWKind::minu;
          break;
        default:
          op->emitError("Invalid integer cmp predicate for aggregation.");
        }
      }
    } else if (matchPattern(
                   op,
                   m_Op<arith::SelectOp>(
                       m_Capture(&cond, m_Op<arith::CmpFOp>(m_Capture(&lhs),
                                                            m_Capture(&rhs))),
                       m_Capture(&trueValue), m_Capture(&falseValue)))) {
      if (lhs == trueValue && rhs == falseValue) {
        relatedOp = cond.getDefiningOp();
        arith::CmpFPredicate floatPred =
            cast<arith::CmpFOp>(relatedOp).getPredicate();
        switch (floatPred) {
        case arith::CmpFPredicate::OGT:
        case arith::CmpFPredicate::OGE:
        case arith::CmpFPredicate::UGT:
        case arith::CmpFPredicate::UGE:
          agg = arith::AtomicRMWKind::maxf;
          break;
        case arith::CmpFPredicate::OLT:
        case arith::CmpFPredicate::OLE:
        case arith::CmpFPredicate::ULT:
        case arith::CmpFPredicate::ULE:
          agg = arith::AtomicRMWKind::minf;
          break;
        default:
          op->emitError("Invalid float cmp predicate for aggregation.");
        }
      }
    } else {
      op->emitError("Invalid operation(s) for reduction.");
    }
    scalar = (lhs == bufElem) ? rhs : lhs;
  }

  arith::AtomicRMWKind getAgg() { return agg; }
  Value getScalar() { return scalar; }
  Operation *getRelatedOp() { return relatedOp; }

private:
  arith::AtomicRMWKind agg;
  Value scalar;
  Operation *relatedOp;
};

// Copy the input buffer to the output buffer.
static AffineParallelOp copyBuffer(OpBuilder &builder, Location loc,
                                   Value input, Value output,
                                   MLIRContext *context) {
  ShapedType type = output.getType().cast<ShapedType>();
  auto forOp = builder.create<AffineParallelOp>(
      loc,
      /*resultTypes=*/TypeRange{type},
      /*reductions=*/
      ArrayRef<arith::AtomicRMWKind>{arith::AtomicRMWKind::assign},
      /*ranges=*/type.getShape());
  Block::BlockArgListType idxs = forOp.getBody()->getArguments();
  OpBuilder bodyBuilder = forOp.getBodyBuilder();
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(type.getRank(), context);
  auto loadOp =
      bodyBuilder.create<pxa::PxaLoadOp>(loc, input, identityMap, idxs);
  auto reduceOp = bodyBuilder.create<pxa::PxaReduceOp>(
      loc, arith::AtomicRMWKind::assign, loadOp, output, identityMap, idxs);
  bodyBuilder.create<AffineYieldOp>(loc, reduceOp.result());
  return forOp;
}

struct ConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    static int constCount = 0;
    Attribute origValue = adaptor.getValue();
    if (auto origType = origValue.getType().dyn_cast<ShapedType>()) {
      LinalgToPXATypeConverter typeConverter;
      MemRefType newType =
          typeConverter.convertType(origType).cast<MemRefType>();
      std::string funcName =
          llvm::formatv("cst_memref_{0}", constCount++).str();
      auto funcOp = op->getParentOfType<func::FuncOp>();
      rewriter.setInsertionPoint(funcOp);
      auto globalOp = rewriter.create<memref::GlobalOp>(
          funcOp.getLoc(),
          /*sym_name=*/funcName,
          /*sym_visibility=*/rewriter.getStringAttr("private"),
          /*type=*/newType,
          /*initial_value=*/origValue,
          /*constant=*/true,
          /*alignment=*/IntegerAttr());
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op, newType,
                                                       globalOp.sym_name());
      return success();
    } else if (origValue.getType().isF32()) {
      Type elementType = origValue.getType();
      MemRefType memRefType = MemRefType::get({}, elementType);
      auto shapeType = RankedTensorType::get({}, elementType);
      std::string funcName =
          llvm::formatv("cst_scalar_memref_{0}", constCount++).str();

      auto funcOp = op->getParentOfType<func::FuncOp>();
      rewriter.setInsertionPoint(funcOp);

      auto globalOp = rewriter.create<memref::GlobalOp>(
          funcOp.getLoc(),
          /*sym_name=*/funcName,
          /*sym_visibility=*/rewriter.getStringAttr("private"),
          /*type=*/memRefType,
          /*initial_value=*/
          DenseElementsAttr::get(shapeType, {origValue}),
          /*constant=*/true,
          /*alignment=*/IntegerAttr());

      rewriter.setInsertionPoint(op);

      auto getGlobalOp = rewriter.create<memref::GetGlobalOp>(
          op.getLoc(), memRefType, globalOp.sym_name());
      SmallVector<Value, 8> idxs;
      auto loadOp =
          rewriter.create<pxa::PxaLoadOp>(op.getLoc(), getGlobalOp, idxs);
      op.replaceAllUsesWith(loadOp.getResult());

      rewriter.eraseOp(op);

      return success();
    }
    return failure();
  }
};

template <typename FuncLikeOp>
struct FuncOpConversion : public OpConversionPattern<FuncLikeOp> {
  using OpConversionPattern<FuncLikeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncLikeOp op, typename FuncLikeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    FunctionType type = op.getFunctionType();

    // Convert the function signature
    LinalgToPXATypeConverter typeConverter;
    TypeConverter::SignatureConversion result(type.getNumInputs());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      result.addInputs(i, {typeConverter.convertType(type.getInput(i))});
    }
    SmallVector<Type, 8> resultTypes;
    for (Type resultType : type.getResults()) {
      Type newResultType = typeConverter.convertType(resultType);
      result.addInputs({newResultType});
      resultTypes.push_back(newResultType);
    }

    // Create a new function with an updated signature.
    FuncLikeOp newOp = rewriter.cloneWithoutRegions(op);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    newOp.setType(FunctionType::get(op.getContext(), result.getConvertedTypes(),
                                    resultTypes));

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newOp.getBody(), result, &typeConverter);

    // Finally cause the old func op to be erased
    rewriter.eraseOp(op);

    return success();
  }
};

// Most of Linalg ops are converted to GenericOp first.
struct GenericOpConversion : public OpConversionPattern<linalg::GenericOp> {
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    ValueRange inputs = adaptor.inputs();
    SmallVector<Value, 4> outputs;
    SmallVector<Type, 4> outputTypes;

    // Copy all output operands in case they are shared with others
    for (auto out : op.getOutputOperands()) {
      Value operand = out->get();
      BufferAllocator allocResult(rewriter, op, operand.getType());
      Value newOut;
      if (operand.isa<BlockArgument>() ||
          !isa<memref::AllocOp>(operand.getDefiningOp())) {
        auto copyOp = copyBuffer(rewriter, op.getLoc(), operand,
                                 allocResult.resultMemRef, op.getContext());
        newOut = copyOp.getResult(0);
      } else {
        newOut = allocResult.resultMemRef;
      }
      out->set(newOut);
      outputs.emplace_back(newOut);
      outputTypes.emplace_back(newOut.getType());
    }

    // Prepare for creating the reduce ops. The operations with output arguments
    // should be converted to reduce op. We need to extract the aggregation
    // kind, the scalar value and the related operations from these operations.
    llvm::SmallSet<Operation *, 4> toRemove;
    auto numInputs = inputs.size();
    auto numOutputs = outputs.size();
    // Make a parallel for loop to fill the result
    SmallVector<arith::AtomicRMWKind, 4> aggs(outputs.size(),
                                              arith::AtomicRMWKind::assign);
    auto outputArgs = op.getBody()->getArguments();
    for (unsigned i = 0; i < outputs.size(); ++i) {
      for (auto &use : outputArgs[numInputs + i].getUses()) {
        auto useOp = use.getOwner();
        if (toRemove.count(useOp) || useOp->getNumResults() != 1) {
          continue;
        }
        // Extract the aggregation kind, the scalar value, and the related op.
        ReductionInfo ri(useOp, use.get());
        useOp->getResult(0).replaceUsesWithIf(
            ri.getScalar(), [&](OpOperand &operand) {
              Operation *owner = operand.getOwner();
              if (!isa<linalg::YieldOp>(owner)) {
                op.emitError("Reduce op is not used by linalg.yield");
              }
              aggs[operand.getOperandNumber()] = ri.getAgg();
              return true;
            });
        // The reduction-like op and the related op will not be copied to the
        // parallel loop.
        toRemove.insert(useOp);
        if (auto relatedOp = ri.getRelatedOp()) {
          toRemove.insert(relatedOp);
        }
      }
    }

    SmallVector<AffineMap, 4> idxMaps = llvm::to_vector<4>(
        adaptor.indexing_maps().getAsValueRange<AffineMapAttr>());

    auto ranges = op.getStaticLoopRanges();
    if (!ranges.size()) {
      op.emitError("LinalgOp does not have static ranges.");
    }
    auto loc = op.getLoc();
    SmallVector<arith::AtomicRMWKind, 4> reductions(
        outputs.size(), arith::AtomicRMWKind::assign);
    auto forOp = rewriter.create<AffineParallelOp>(loc,
                                                   /*resultTypes=*/outputTypes,
                                                   /*reductions=*/reductions,
                                                   /*ranges=*/ranges);

    // Create the a load op for each block argument.
    Block *forBody = forOp.getBody();
    auto idxs = forBody->getArguments();
    rewriter.setInsertionPointToStart(forBody);

    // Add constraints
    Block *body = forBody;
    if (auto cons = op->getAttrOfType<IntegerSetAttr>("constraints")) {
      auto ifOp = rewriter.create<AffineIfOp>(loc, outputTypes, cons.getValue(),
                                              idxs, true);
      rewriter.create<AffineYieldOp>(loc, ifOp->getResults());
      rewriter.setInsertionPointToStart(&ifOp.elseRegion().front());
      rewriter.create<AffineYieldOp>(loc, outputs);
      body = &ifOp.thenRegion().front();
      rewriter.setInsertionPointToStart(body);
    }

    for (unsigned i = 0; i < numInputs; ++i) {
      if (inputs[i].getType().isa<ShapedType>()) {
        // input is a tensor
        auto loadOp =
            rewriter.create<pxa::PxaLoadOp>(loc, inputs[i], idxMaps[i], idxs);
        outputArgs[i].replaceAllUsesWith(loadOp.getResult());
      } else {
        // input is a scalar
        outputArgs[i].replaceAllUsesWith(inputs[i]);
      }
    }

    // Move the original ops except for toRemove into the parallel loop
    SmallVector<Operation *, 4> toMove;
    for (auto &origOp : *op.getBody()) {
      if (toRemove.count(&origOp) == 0) {
        toMove.emplace_back(&origOp);
      }
    }
    for (auto newOp : toMove) {
      newOp->moveBefore(body, body->getOperations().end());
    }

    // Must insert reduce ops here. Later on, the reduce op's memref information
    // may be lost while the generic op is erased.
    if (auto yieldOp = dyn_cast<linalg::YieldOp>(body->back())) {
      SmallVector<AffineMap, 4> maps(idxMaps.begin() + numInputs,
                                     idxMaps.end());
      SmallVector<Value, 4> outs(outputs.begin(), outputs.end());
      auto results = yieldOp.getOperands();
      rewriter.setInsertionPoint(yieldOp);
      for (unsigned i = 0; i < numOutputs; ++i) {
        if (outputs[i].getType().isa<ShapedType>()) {
          auto reduceOp = rewriter.create<pxa::PxaReduceOp>(
              yieldOp.getLoc(), aggs[i], results[i], outputs[i], maps[i], idxs);
          results[i].replaceUsesWithIf(reduceOp.getResult(),
                                       [&](OpOperand &operand) {
                                         Operation *owner = operand.getOwner();
                                         return isa<linalg::YieldOp>(owner);
                                       });
        }
      }
    } else {
      op->emitError("No linalg.yield in generic op.");
    }

    // auto m = op->getParentOfType<ModuleOp>();
    // llvm::errs() << "-----------------\n";
    // m.dump();
    // llvm::errs() << "------------------\n";
    rewriter.replaceOp(op, forOp.getResults());
    return success();
  }
};

struct YieldXOpConversion : public OpConversionPattern<stdx::YieldOp> {
  using OpConversionPattern<stdx::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(stdx::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<stdx::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

struct IndexOpConversion : public OpConversionPattern<linalg::IndexOp> {
  using OpConversionPattern<linalg::IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::IndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto idxs = op->getBlock()->getArguments();
    op.replaceAllUsesWith(idxs[op.dim()]);
    rewriter.eraseOp(op);
    return success();
  }
};

struct InitTensorOpConversion
    : public OpConversionPattern<linalg::InitTensorOp> {
  using OpConversionPattern<linalg::InitTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::InitTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto type = op.result().getType().cast<RankedTensorType>();
    if (llvm::none_of(type.getShape(), ShapedType::isDynamic)) {
      BufferAllocator allocResult(rewriter, op, type);
      op.replaceAllUsesWith(allocResult.resultMemRef);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct YieldOpConversion : public OpConversionPattern<linalg::YieldOp> {
  using OpConversionPattern<linalg::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<AffineYieldOp>(op, adaptor.getOperands());
    return success();
  }
};

struct LowerLinalgToPXAPass
    : public LowerLinalgToPXABase<LowerLinalgToPXAPass> {

  void performLinalgTransforms(ModuleOp op) {
    RewritePatternSet patterns(op.getContext());
    linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns);
    populateLinalgTensorCollapseOpGeneralizationPatterns(patterns);
    populateLinalgTensorExpandOpGeneralizationPatterns(patterns);
    patterns.add<linalg::PadOpTransformationPattern>(op.getContext());
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    // Convert named/structured ops to GenericOps.
    performLinalgTransforms(module);

    ConversionTarget target(getContext());
    LinalgToPXATypeConverter converter;
    target.addLegalDialect<AffineDialect, 
                           math::MathDialect,     
                           memref::MemRefDialect, 
                           scf::SCFDialect,       
                           layer::LayerDialect,   
                           pxa::PXADialect,       
                           arith::ArithmeticDialect, 
                           stdx::StdXDialect>();
   
    // Module op is legal.
    target.addLegalOp<ModuleOp, func::CallOp>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });

    target.addDynamicallyLegalOp<stdx::YieldOp>(
        [&](stdx::YieldOp op) { return converter.isLegal(op); });

    target.addDynamicallyLegalOp<stdx::ClosureOp>([&](stdx::ClosureOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });

    target.addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp op) {
      return (!op.getType().isa<TensorType>() && !op.getType().isF32());
    });

    // Linalg is illegal.
    target.addIllegalDialect<linalg::LinalgDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.insert<YieldOpConversion, ConstantOpConversion,
                    // YieldXOpConversion,
                    // FuncOpConversion<stdx::ClosureOp>,
                    FuncOpConversion<func::FuncOp>, GenericOpConversion,
                    // IndexOpConversion,
                    InitTensorOpConversion>(&getContext());

    tile_to_pxa::populateTileToPXASpecialPatterns(patterns);
    populateReturnOpTypeConversionPattern(patterns, converter);

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isLegalForReturnOpTypeConversionPattern(op, converter); });

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    
    for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
      for (func::ReturnOp returnOp : funcOp.getOps<func::ReturnOp>()) {
        connectResults(funcOp, returnOp);
        for (stdx::ClosureOp closureOp : funcOp.getOps<stdx::ClosureOp>()) {
          for (stdx::YieldOp yieldOp : closureOp.getOps<stdx::YieldOp>()) {
            connectResults(closureOp, yieldOp);
          }
        }
      }
    }
  }

  template <typename FuncLikeOp, typename ReturnLikeOp>
  void connectResults(FuncLikeOp funcOp, ReturnLikeOp returnOp) {
    unsigned argNumber =
        funcOp.getFunctionType().getNumInputs() - returnOp.getNumOperands();
    MLIRContext *context = &getContext();
    Location loc = returnOp->getLoc();
    for (OpOperand &operand : returnOp->getOpOperands()) {
      // Find very initial allocation of memref
      Value def = pxa::getIndirectDef(operand.get());
      BlockArgument outputArg = funcOp.getBody().getArgument(argNumber++);
      if (def != outputArg) {
        if (def.isa<BlockArgument>() ||
            isa<memref::GetGlobalOp>(def.getDefiningOp())) {
          OpBuilder builder(returnOp);
          auto forOp = copyBuffer(builder, loc, def, outputArg, context);
          operand.set(forOp.getResult(0));
          continue;
        }
        Value outside = pxa::getIndirectDefOutsideScope(operand.get(), funcOp);
        if (outside && !isa<memref::AllocOp>(outside.getDefiningOp())) {
          OpBuilder builder(funcOp.getBody());
          auto forOp = copyBuffer(builder, loc, outside, outputArg, context);
          outside.replaceUsesWithIf(forOp.getResult(0), [&](OpOperand &use) {
            Operation *op = use.getOwner();
            return !forOp->isProperAncestor(op) && funcOp->isProperAncestor(op);
          });
          continue;
        }
        def.replaceAllUsesWith(outputArg);
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createLowerLinalgToPXAPass() {
  return std::make_unique<LowerLinalgToPXAPass>();
}

} // namespace pmlc::conversion::linalg_to_pxa
