// Copyright 2021, Intel Corporation

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "pmlc/conversion/linalg_to_pxa/pass_detail.h"
#include "pmlc/util/matchers.h"
#include "pmlc/util/util.h"

namespace pmlc::conversion::linalg_to_pxa {

namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;

using namespace mlir;         // NOLINT
using namespace mlir::linalg; // NOLINT

namespace {

static int constCount = 0;

static RankedTensorType getRankedTensorType(Type type) {
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
    return rankedTensorType;
  }
  return RankedTensorType::get({}, type);
}

struct BufferAllocator {
  Value resultMemRef;
  RankedTensorType rankedTensorType;
  MemRefType memRefType;
  Type elementType;

  BufferAllocator(OpBuilder &builder, Operation *op, Type resultType) {
    // Gather some basic info
    LinalgToPXATypeConverter typeConverter;
    auto loc = op->getLoc();
    rankedTensorType = getRankedTensorType(resultType);
    elementType = typeConverter.convertType(rankedTensorType.getElementType());
    ArrayRef<int64_t> originalShape = rankedTensorType.getShape();
    auto shape = llvm::to_vector<8>(originalShape);

    // Make an allocation for the output
    memRefType = MemRefType::get(shape, elementType);
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
    if (matchPattern(op, m_Op<AddIOp>(m_Capture(&lhs), m_Capture(&rhs)))) {
      agg = AtomicRMWKind::addi;
    } else if (matchPattern(op,
                            m_Op<AddFOp>(m_Capture(&lhs), m_Capture(&rhs)))) {
      agg = AtomicRMWKind::addf;
    } else if (matchPattern(op,
                            m_Op<MulIOp>(m_Capture(&lhs), m_Capture(&rhs)))) {
      agg = AtomicRMWKind::muli;
    } else if (matchPattern(op,
                            m_Op<MulFOp>(m_Capture(&lhs), m_Capture(&rhs)))) {
      agg = AtomicRMWKind::mulf;
    } else if (matchPattern(
                   op, m_Op<SelectOp>(
                           m_Capture(&cond, m_Op<CmpIOp>(m_Capture(&lhs),
                                                         m_Capture(&rhs))),
                           m_Capture(&trueValue), m_Capture(&falseValue)))) {
      if (lhs == trueValue && rhs == falseValue) {
        relatedOp = cond.getDefiningOp();
        CmpIPredicate intPred = cast<CmpIOp>(relatedOp).predicate();
        switch (intPred) {
        case CmpIPredicate::sgt:
        case CmpIPredicate::sge:
          agg = AtomicRMWKind::maxs;
          break;
        case CmpIPredicate::ugt:
        case CmpIPredicate::uge:
          agg = AtomicRMWKind::maxu;
          break;
        case CmpIPredicate::slt:
        case CmpIPredicate::sle:
          agg = AtomicRMWKind::mins;
          break;
        case CmpIPredicate::ult:
        case CmpIPredicate::ule:
          agg = AtomicRMWKind::minu;
          break;
        default:
          op->emitError("Invalid integer cmp predicate for aggregation.");
        }
      }
    } else if (matchPattern(
                   op, m_Op<SelectOp>(
                           m_Capture(&cond, m_Op<CmpFOp>(m_Capture(&lhs),
                                                         m_Capture(&rhs))),
                           m_Capture(&trueValue), m_Capture(&falseValue)))) {
      if (lhs == trueValue && rhs == falseValue) {
        relatedOp = cond.getDefiningOp();
        CmpFPredicate floatPred = cast<CmpFOp>(relatedOp).predicate();
        switch (floatPred) {
        case CmpFPredicate::OGT:
        case CmpFPredicate::OGE:
        case CmpFPredicate::UGT:
        case CmpFPredicate::UGE:
          agg = AtomicRMWKind::maxf;
          break;
        case CmpFPredicate::OLT:
        case CmpFPredicate::OLE:
        case CmpFPredicate::ULT:
        case CmpFPredicate::ULE:
          agg = AtomicRMWKind::minf;
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

  AtomicRMWKind getAgg() { return agg; }
  Value getScalar() { return scalar; }
  Operation *getRelatedOp() { return relatedOp; }

private:
  AtomicRMWKind agg;
  Value scalar;
  Operation *relatedOp;
};

struct ConstantOpConversion : public OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto origValue = op.getValue();
    if (auto origType = origValue.getType().dyn_cast<ShapedType>()) {
      LinalgToPXATypeConverter typeConverter;
      auto newType = typeConverter.convertType(origType);
      auto name = llvm::formatv("cst_{0}", constCount++).str();
      auto funcOp = op->getParentOfType<FuncOp>();
      rewriter.setInsertionPoint(funcOp);
      auto globalOp = rewriter.create<memref::GlobalOp>(
          funcOp.getLoc(),
          /*sym_name=*/name,
          /*sym_visibility=*/rewriter.getStringAttr("private"),
          /*type=*/newType,
          /*initial_value=*/origValue,
          /*constant=*/true);
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op, newType,
                                                       globalOp.sym_name());
      return success();
    }
    return failure();
  }
};

struct FuncOpConversion : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    FunctionType type = op.getType();

    // Convert the function signature
    LinalgToPXATypeConverter typeConverter;
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      result.addInputs(i, {typeConverter.convertType(type.getInput(i))});
    }
    SmallVector<Type, 8> resultTypes;
    for (Type resultType : type.getResults()) {
      Type newResultType = typeConverter.convertType(resultType);
      if (!newResultType.isa<stdx::ArgpackType>()) {
        result.addInputs({newResultType});
      }
      resultTypes.push_back(newResultType);
    }

    // Create a new function with an updated signature.
    auto newOp = rewriter.cloneWithoutRegions(op);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());
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
struct GenericOpConversion : public OpConversionPattern<GenericOp> {
  using OpConversionPattern<GenericOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GenericOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    GenericOpAdaptor adaptor(operands, op.getOperation()->getAttrDictionary());
    auto inputs = adaptor.inputs();
    auto outputs = adaptor.outputs();
    auto idxMapAttrs = adaptor.indexing_maps().getValue();
    SmallVector<AffineMap, 4> idxMaps;
    for (auto attr : idxMapAttrs) {
      idxMaps.emplace_back(attr.cast<AffineMapAttr>().getValue());
    }

    // Get all inputs' and outputs' shapes. Note that inputs and outputs may be
    // scalar.
    SmallVector<ArrayRef<int64_t>, 4> shapes;
    for (auto input : inputs) {
      if (auto shapedType = input.getType().dyn_cast<ShapedType>()) {
        shapes.emplace_back(shapedType.getShape());
      } else {
        shapes.emplace_back(ArrayRef<int64_t>{1});
      }
    }
    for (auto output : outputs) {
      if (auto shapedType = output.getType().dyn_cast<ShapedType>()) {
        shapes.emplace_back(shapedType.getShape());
      } else {
        shapes.emplace_back(ArrayRef<int64_t>{1});
      }
    }

    // Get output types
    SmallVector<Type, 4> outputTypes;
    for (auto output : outputs) {
      outputTypes.emplace_back(output.getType());
    }

    // Prepare for creating the reduce ops. The operations with output arguments
    // should be converted to reduce op. We need to extract the aggregation
    // kind, the scalar value and the related operations from these operations.
    llvm::SmallSet<Operation *, 4> toRemove;
    auto numInputs = inputs.size();
    auto numOutputs = outputs.size();
    // Make a parallel for loop to fill the result
    SmallVector<AtomicRMWKind, 4> aggs(outputs.size(), AtomicRMWKind::assign);
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
            ri.getScalar(), [&](mlir::OpOperand &operand) {
              mlir::Operation *owner = operand.getOwner();
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

    auto ranges = op.getStaticLoopRanges();
    if (!ranges) {
      op.emitError("LiangOp does not have static ranges.");
    }
    auto forOp = rewriter.create<AffineParallelOp>(op.getLoc(),
                                                   /*resultTypes=*/outputTypes,
                                                   /*reductions=*/aggs,
                                                   /*ranges=*/*ranges);

    // Create the a load op for each block argument.
    auto forBody = forOp.getBody();
    auto idxs = forBody->getArguments();
    auto opArgs = op.getBody()->getArguments();
    rewriter.setInsertionPointToStart(forBody);
    for (unsigned i = 0; i < numInputs; ++i) {
      if (inputs[i].getType().isa<ShapedType>()) {
        // input is a tensor
        auto loadOp = rewriter.create<pxa::PxaLoadOp>(forOp.getLoc(), inputs[i],
                                                      idxMaps[i], idxs);
        opArgs[i].replaceAllUsesWith(loadOp.getResult());
      } else {
        // input is a scalar
        opArgs[i].replaceAllUsesWith(inputs[i]);
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
      newOp->moveBefore(forBody, forBody->getOperations().end());
    }

    // Must insert reduce ops here. Later on, the reduce op's memref information
    // may be lost while the generic op is erased.
    if (auto yieldOp = dyn_cast<linalg::YieldOp>(forBody->back())) {
      SmallVector<AffineMap, 4> maps(idxMaps.begin() + numInputs,
                                     idxMaps.end());
      SmallVector<Value, 4> outs(outputs.begin(), outputs.end());
      auto results = yieldOp.getOperands();
      rewriter.setInsertionPoint(yieldOp);
      for (unsigned i = 0; i < numOutputs; ++i) {
        if (outputs[i].getType().isa<ShapedType>()) {
          auto reduceOp = rewriter.create<pxa::PxaReduceOp>(
              yieldOp.getLoc(), aggs[i], results[i], outputs[i], maps[i], idxs);
          results[i].replaceUsesWithIf(
              reduceOp.getResult(), [&](mlir::OpOperand &operand) {
                mlir::Operation *owner = operand.getOwner();
                return isa<linalg::YieldOp>(owner);
              });
        }
      }
    } else {
      op->emitError("No linalg.yield in generic op.");
    }

    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      op.getResult(i).replaceAllUsesWith(forOp.getResult(i));
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct IndexOpConversion : public OpConversionPattern<IndexOp> {
  using OpConversionPattern<IndexOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IndexOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto idxs = op.getOperation()->getBlock()->getArguments();
    op.replaceAllUsesWith(idxs[op.dim()]);
    rewriter.eraseOp(op);
    return success();
  }
};

struct InitTensorOpConversion : public OpConversionPattern<InitTensorOp> {
  using OpConversionPattern<InitTensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InitTensorOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    BufferAllocator allocResult(rewriter, op.getOperation(),
                                op.result().getType());
    op.replaceAllUsesWith(allocResult.resultMemRef);
    rewriter.eraseOp(op);
    return success();
  }
};

struct RangeOpConversion : public OpConversionPattern<RangeOp> {
  using OpConversionPattern<RangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RangeOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    op.emitError("Conversion of linalg.range is not implemented.");
    return success();
  }
};

struct TiledLoopOpConversion : public OpConversionPattern<TiledLoopOp> {
  using OpConversionPattern<TiledLoopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TiledLoopOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    op.emitError("Conversion of linalg.tiled_loop is not implemented.");
    return success();
  }
};

struct YieldOpConversion : public OpConversionPattern<linalg::YieldOp> {
  using OpConversionPattern<linalg::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::YieldOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<AffineYieldOp>(op, operands);
    return success();
  }
};

struct LowerLinalgToPXAPass
    : public LowerLinalgToPXABase<LowerLinalgToPXAPass> {

  void performLinalgTransforms(ModuleOp op) {
    // We perform some Linalg transformations first in order to convert as many
    // as operations to GenericOp.
    RewritePatternSet patterns(op.getContext());
    populateLinalgConvGeneralizationPatterns(patterns);
    populateLinalgNamedOpsGeneralizationPatterns(patterns);
    populateLinalgTensorCollapseOpGeneralizationPatterns(patterns);
    populateLinalgTensorExpandOpGeneralizationPatterns(patterns);
    populateLinalgPoolingOpGeneralizationPatterns(patterns);
    patterns.add<PadTensorOpTransformationPattern>(op.getContext());
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }

  void runOnOperation() final {
    // Perform Linalg transformations to convert some operations to
    // linalg.generic
    auto module = getOperation();
    performLinalgTransforms(module);

    // Set up target (i.e. what is legal)
    ConversionTarget target(getContext());
    LinalgToPXATypeConverter converter;
    target.addLegalDialect<mlir::AffineDialect,         //
                           mlir::StandardOpsDialect,    //
                           mlir::math::MathDialect,     //
                           mlir::memref::MemRefDialect, //
                           mlir::scf::SCFDialect,       //
                           pxa::PXADialect,             //
                           stdx::StdXDialect>();
    target.addLegalOp<scf::ForOp,   //
                      scf::YieldOp, //
                      scf::IfOp>();
    target.addLegalOp<mlir::ModuleOp, //
                      ReturnOp>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    target.addDynamicallyLegalOp<ConstantOp>(
        [&](ConstantOp op) { return !op.getType().isa<TensorType>(); });

    // These ops should be converted by the Linalg transformations before
    target.addIllegalOp<ConvOp,       //
                        CopyOp,       //
                        FillOp,       //
                        PadTensorOp,  //
                        PoolingMaxOp, //
                        PoolingMinOp, //
                        PoolingSumOp>();

    // Setup rewrite patterns
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConstantOpConversion,   //
                    FuncOpConversion,       //
                    GenericOpConversion,    //
                    IndexOpConversion,      //
                    InitTensorOpConversion, //
                    RangeOpConversion,      //
                    TiledLoopOpConversion,  //
                    YieldOpConversion>(&getContext());

    // Run the conversion
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace

std::unique_ptr<Pass> createLowerLinalgToPXAPass() {
  return std::make_unique<LowerLinalgToPXAPass>();
}

} // namespace pmlc::conversion::linalg_to_pxa
