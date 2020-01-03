// Copyright 2019 Intel Corporation

#include <algorithm>
#include <utility>

#include "pmlc/conversion/stripe_to_affine/convert_stripe_to_affine.h"
#include "pmlc/dialect/eltwise/types.h"
#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/populate_tensor_ref_shape_analysis.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define PASS_NAME "convert-stripe-to-affine"
#define DEBUG_TYPE PASS_NAME

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::SmallDenseMap;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using mlir::AffineExpr;
using mlir::AffineForOp;
using mlir::AffineLoadOp;
using mlir::AffineMap;
using mlir::AffineStoreOp;
using mlir::ArrayAttr;
using mlir::ArrayRef;
using mlir::Block;
using mlir::ConstantIndexOp;
using mlir::ConversionPatternRewriter;
using mlir::DenseMap;
using mlir::FuncOp;
using mlir::IntegerAttr;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::OpConversionPattern;
using mlir::Operation;
using mlir::OwningRewritePatternList;
using mlir::PatternMatchResult;
using mlir::Type;
using mlir::TypeConverter;
using mlir::Value;
using mlir::edsc::AffineLoopNestBuilder;
using mlir::edsc::IndexHandle;
using mlir::edsc::ScopedContext;
using mlir::edsc::ValueHandle;
using mlir::edsc::intrinsics::constant_index;
using pmlc::dialect::eltwise::ScalarType;
using pmlc::dialect::stripe::AffinePolyOp;
using pmlc::dialect::stripe::AffineType;
using pmlc::dialect::stripe::ComputeAccess;
using pmlc::dialect::stripe::FlatTensorAccess;
using pmlc::dialect::stripe::LoadOp;
using pmlc::dialect::stripe::ParallelForOp;
using pmlc::dialect::stripe::PopulateTensorRefShape;
using pmlc::dialect::stripe::RefineOp;
using pmlc::dialect::stripe::StoreOp;
using pmlc::dialect::stripe::TensorDim;
using pmlc::dialect::stripe::TensorRefType;
using pmlc::dialect::stripe::TerminateOp;
using vertexai::tile::is_int;
using vertexai::tile::is_uint;

namespace {

/// Analysis that computes the flat tensor access information for every refine operation.
class AccessInfoAnalysis {
 public:
  explicit AccessInfoAnalysis(FuncOp f) : func(f) { recompute(); }

  // Recompute polynomial information for all the refine ops in the target FuncOp. Pre-existing analysis information
  // is cleared beforehand.
  void recompute() {
    accessInfoMap.clear();
    func.walk([&](Operation* op) {
      if (auto refine = dyn_cast<RefineOp>(op)) {
        // TODO: compute access information only for those refineOps that feed a memory operation?
        accessInfoMap[refine] = ComputeAccess(refine.getResult());
      }
    });
  }

  // Retrieve the polynomial information for a given Operation. Assert if there is no polynomial information available
  // for that Operation.
  const FlatTensorAccess& getAccessInfo(Operation* op) const {
    auto it = accessInfoMap.find(op);
    assert(it != accessInfoMap.end() && "AccessInfo not found.");
    return it->second;
  }

 private:
  FuncOp func;

  // Map between an operation and its computed flat polynomial information.
  DenseMap<Operation*, FlatTensorAccess> accessInfoMap;
};

/// Stripe to Affine type converter.
class StripeToAffineTypeConverter : public mlir::TypeConverter {
 public:
  StripeToAffineTypeConverter() : TypeConverter() {}

  Type convertType(Type t) override;

 private:
  /// Utility that converts a Stripe element type to Affine.
  Type convertEltType(ScalarType t, MLIRContext* context);
};

Type StripeToAffineTypeConverter::convertType(Type type) {
  if (auto tensorRef = type.dyn_cast<TensorRefType>()) {
    // Transform Stripe shape to Affine shape format and convert TensorRefType to MemRefType.
    auto stripeShape = tensorRef.getShape();
    assert(stripeShape.size() > 0 && "Unexpected shape in TensorRefType");
    SmallVector<int64_t, 8> affineShape;
    std::transform(stripeShape.begin(), stripeShape.end(), std::back_inserter(affineShape),
                   [](const TensorDim& dim) -> int64_t { return dim.size; });
    return MemRefType::get(affineShape, convertType(tensorRef.getElementType()), {/* no map used */}, 0);
  } else if (auto scalarType = type.dyn_cast<ScalarType>()) {
    return convertEltType(scalarType, type.getContext());
  }
  // No conversion neeeded. Return the input type.
  return type;
}

/// Utility that converts a Stripe element type to Affine.
Type StripeToAffineTypeConverter::convertEltType(ScalarType scalarType, MLIRContext* context) {
  // Process the non-MLIR Tile type in Eltwise ScalarType.
  auto tileType = scalarType.type();
  if (is_int(tileType) || is_uint(tileType)) {
    return mlir::IntegerType::get(bit_width(tileType), context);
  } else if (is_float(tileType)) {
    switch (bit_width(tileType)) {
      case 32:
        return mlir::FloatType::getF32(context);
      case 64:
        return mlir::FloatType::getF64(context);
    }
    llvm_unreachable("Unsupported size for float ScalarType");
  }
  llvm_unreachable("Unsupported ScalarType");
}

/// Context for Stripe to Affine conversion. It holds analysis and conversion state information.
struct StripeToAffineContext {
  explicit StripeToAffineContext(FuncOp func) : accessAnalysis(func) {}

  // Stripe memory access information.
  const AccessInfoAnalysis accessAnalysis;

  // Mapping between Stripe induction variable and Affine induction variable BlockArguments. Note that this mapping is
  // not part of ConversionPatternRewriter. ParallelForOp is replaced with an Affine loop nest as a whole, not piece by
  // piece.
  // TODO: Revisit this in the future if there is a way to convert BlockArguments using ConversionPatternRewriter.
  SmallDenseMap<Value, Value, 8> remappedIvs;
};

// Base class for Stripe Op conversion.
template <class OP>
class StripeOpConverter : public OpConversionPattern<OP> {
 public:
  using OpConversionPattern<OP>::OpConversionPattern;
  StripeOpConverter(MLIRContext* context, StripeToAffineContext& convContext)
      : OpConversionPattern<OP>(context, /*benefit=*/1), convContext(convContext) {}

 protected:
  /// Utility that converts a Stripe access to Affine given a `FlatTensorAccess` that contains the polynomial
  /// information. It returns:
  ///    - 'base': affine.load/affine.store base memref Value.
  ///    - 'indices': list of indices to be used as affine map arguments.
  ///    - 'affineMap': affine map with mapping for for the list of indices.
  void convertStripeAccessToAffine(RefineOp refine, ConversionPatternRewriter& rewriter, Value& base,
                                   SmallVectorImpl<Value>& indices, AffineMap& affineMap) const {
    SmallVector<AffineExpr, 4> resultExprs;
    // Process all the polynomials in the access. Each polynomial has a set of terms consisting of {index * constant
    // coefficient}, and an independent constant term. We create an affine expression for each polynomial.
    FlatTensorAccess accessInfo = convContext.accessAnalysis.getAccessInfo(refine);
    unsigned dimPos = 0;
    for (auto polynomial : accessInfo.access) {
      // Create an affine expression with the independent constant term.
      AffineExpr polyExpr = rewriter.getAffineConstantExpr(polynomial.constant);
      for (auto term : polynomial.terms) {
        auto remappedIvIt = convContext.remappedIvs.find(term.first);
        assert(remappedIvIt != convContext.remappedIvs.end() && "IV not found in Stripe->Affine mapping");
        indices.emplace_back(remappedIvIt->second);
        // Add index * coefficient expression to the affine expression.
        auto indexExpr = rewriter.getAffineDimExpr(dimPos);
        auto multiplierExpr = rewriter.getAffineConstantExpr(term.second);
        auto termExpr = indexExpr * multiplierExpr;
        polyExpr = termExpr + polyExpr;
        ++dimPos;
      }
      resultExprs.emplace_back(std::move(polyExpr));
    }

    base = rewriter.getRemappedValue(accessInfo.base);
    // Create an affine map with the resulting affine expressions. The affine map won't have symbols since all the
    // coefficients in Stripe polynomials are constant for now.
    affineMap = AffineMap::get(/*dimCount=*/dimPos, /*symbolCount=*/0, resultExprs);
  }

  // Contains analysis and temporary information needed for the Stripe to Affine conversion.
  StripeToAffineContext& convContext;
};

// Declaration of Stripe Ops converters for supported Ops.
#define STRIPE_OP(OP)                                                                       \
  struct OP##Converter : public StripeOpConverter<OP> {                                     \
    using StripeOpConverter<OP>::StripeOpConverter;                                         \
                                                                                            \
    PatternMatchResult matchAndRewrite(OP op, ArrayRef<Value> operands,                     \
                                       ConversionPatternRewriter& rewriter) const override; \
  };
#include "supported_ops.inc"

PatternMatchResult AffinePolyOpConverter::matchAndRewrite(AffinePolyOp constOp, ArrayRef<Value> operands,
                                                          ConversionPatternRewriter& rewriter) const {
  // This op is indirectly converted from the memory access operation by using AccessInfoAnalysis.
  rewriter.eraseOp(constOp);
  return matchSuccess();
}

PatternMatchResult ParallelForOpConverter::matchAndRewrite(ParallelForOp stripeForOp, ArrayRef<Value> operands,
                                                           ConversionPatternRewriter& rewriter) const {
  auto ranges = stripeForOp.ranges().getValue();
  size_t numRanges = ranges.size();
  auto& stripeForBody = stripeForOp.inner();
  assert(stripeForBody.getBlocks().size() == 1 && "Unexpected control flow in Stripe");

  if (numRanges == 0) {
    // This is ParallelForOp with no ranges so no affine.loop needed ("main" case). Move ParallelForOp's operations
    // (single block) into parent single block. ParallelForOp terminator is not moved since parent region already have
    // one.
    auto& parentBlockOps = stripeForOp.getOperation()->getBlock()->getOperations();
    auto& bodyOps = stripeForBody.front().getOperations();
    assert(isa<TerminateOp>(parentBlockOps.back()) && "Expected terminator");
    parentBlockOps.splice(Block::iterator(stripeForOp), bodyOps, bodyOps.begin(), std::prev(bodyOps.end()));
  } else {
    // Create an Affine loop nest following the order of ranges.
    ScopedContext scope(rewriter, stripeForOp.getLoc());
    SmallVector<ValueHandle, 8> affineIvs;
    SmallVector<ValueHandle*, 8> affineIvPtrs;
    SmallVector<ValueHandle, 8> affineLbs;
    SmallVector<ValueHandle, 8> affineUbs;
    SmallVector<int64_t, 8> affineSteps;

    for (size_t rangeIdx = 0; rangeIdx < numRanges; ++rangeIdx) {
      affineIvs.emplace_back(IndexHandle());
      affineIvPtrs.emplace_back(&affineIvs.back());
      affineLbs.emplace_back(constant_index(0));
      affineUbs.emplace_back(constant_index(ranges[rangeIdx].cast<IntegerAttr>().getInt()));
      affineSteps.emplace_back(1);
    }

    // Build the empty Affine loop nest with an innermost loop body containing a terminator.
    AffineLoopNestBuilder(affineIvPtrs, affineLbs, affineUbs, affineSteps)();

    const auto stripeIvValues = stripeForBody.front().getArguments();
    assert(stripeIvValues.size() == affineIvs.size() && "Stripe and Affine number of IVs doesn't match");
    for (size_t ivIdx = 0; ivIdx < numRanges; ++ivIdx) {
      convContext.remappedIvs.insert(std::pair<Value, Value>(stripeIvValues[ivIdx], affineIvs[ivIdx].getValue()));
    }

    // Move ParallelForOp's operations (single block) to Affine innermost loop. We skip the terminator since Affine loop
    // has one already.
    auto& innermostLoopOps = mlir::getForInductionVarOwner(affineIvs[numRanges - 1]).getBody()->getOperations();
    auto& stripeBodyOps = stripeForBody.front().getOperations();
    innermostLoopOps.splice(Block::iterator(innermostLoopOps.front()), stripeBodyOps, stripeBodyOps.begin(),
                            std::prev(stripeBodyOps.end()));
  }

  // We are done. Remove ParallelForOp.
  rewriter.replaceOp(stripeForOp, {});
  return matchSuccess();
}

PatternMatchResult LoadOpConverter::matchAndRewrite(LoadOp loadOp, ArrayRef<Value> operands,
                                                    ConversionPatternRewriter& rewriter) const {
  // Create map and indices to be used as map arguments.
  AffineMap map;
  SmallVector<Value, 8> indices;
  Value base;
  RefineOp refine = cast<RefineOp>(loadOp.from()->getDefiningOp());
  convertStripeAccessToAffine(refine, rewriter, base, indices, map);
  rewriter.replaceOpWithNewOp<AffineLoadOp>(loadOp, base, map, indices);
  return matchSuccess();
}

PatternMatchResult RefineOpConverter::matchAndRewrite(RefineOp refineOp, ArrayRef<Value> operands,
                                                      ConversionPatternRewriter& rewriter) const {
  // This op is indirectly converted from the memory access operation by using AccessInfoAnalysis.
  rewriter.eraseOp(refineOp);
  return matchSuccess();
}

PatternMatchResult StoreOpConverter::matchAndRewrite(StoreOp storeOp, ArrayRef<Value> operands,
                                                     ConversionPatternRewriter& rewriter) const {
  // Create map and indices to be used as map arguments.
  AffineMap map;
  SmallVector<Value, 8> indices;
  Value base;
  RefineOp refine = cast<RefineOp>(storeOp.into()->getDefiningOp());
  convertStripeAccessToAffine(refine, rewriter, base, indices, map);
  rewriter.replaceOpWithNewOp<AffineStoreOp>(storeOp, operands[1], base, map, indices);
  return matchSuccess();
}

// Converts TerminateOp to AffineTerminatorOp. If a TerminateOp requires a special context-dependent treatment, that
// must be implemented in the Op providing the context.
PatternMatchResult TerminateOpConverter::matchAndRewrite(TerminateOp terminateOp, ArrayRef<Value> operands,
                                                         ConversionPatternRewriter& rewriter) const {
  rewriter.replaceOpWithNewOp<mlir::AffineTerminatorOp>(terminateOp);
  return matchSuccess();
}

void populateStripeToAffineConversionPatterns(OwningRewritePatternList& patterns, MLIRContext* ctx,
                                              TypeConverter& typeConverter, StripeToAffineContext& convContext) {
#define STRIPE_OP(OP) OP##Converter,
#define STRIPE_LAST_OP(OP) OP##Converter
  patterns.insert<
#include "supported_ops.inc"  // NOLINT(build/include)
      >(ctx, convContext);

  mlir::populateFuncOpTypeConversionPattern(patterns, ctx, typeConverter);
}

// Pass to convert Stripe dialect to Affine dialect.
struct ConvertStripeToAffine : public mlir::FunctionPass<ConvertStripeToAffine> {
  void runOnFunction() override;
};

void ConvertStripeToAffine::runOnFunction() {
  // Add shape information to tensor ref types, which is needed for Stripe->Affine type conversion.
  getAnalysis<PopulateTensorRefShape>();

  StripeToAffineContext convContext(getFunction());

  StripeToAffineTypeConverter typeConverter;
  OwningRewritePatternList patterns;
  populateStripeToAffineConversionPatterns(patterns, &getContext(), typeConverter, convContext);

  // Add Affine/Std dialect legal ops to conversion target.
  mlir::ConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
  target.addLegalDialect<mlir::AffineOpsDialect, mlir::StandardOpsDialect>();
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp op) {
    // FuncOp is legal only if types have been converted to Std types.
    return typeConverter.isSignatureLegal(op.getType());
  });

  if (failed(mlir::applyFullConversion(getFunction(), target, patterns))) {
    signalPassFailure();
  }
}
}  // namespace

std::unique_ptr<mlir::FunctionPassBase> mlir::createConvertStripeToAffinePass() {
  return std::make_unique<ConvertStripeToAffine>();
}

static mlir::PassRegistration<ConvertStripeToAffine> pass(PASS_NAME, "Convert Stripe dialect to Affine dialect");
