// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/builder.h"

#include <limits>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/dialect/eltwise/ir/dialect.h"
#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/tile/gradient.h"
#include "pmlc/dialect/tile/ir/dialect.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/slice.h"
#include "pmlc/util/util.h"

namespace pmlc::dialect::tile {

using eltwise::ScalarConstantOp;
using eltwise::ScalarType;
using llvm::ArrayRef;
using llvm::SetVector;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::StringSwitch;
using mlir::AbstractOperation;
using mlir::Block;
using mlir::BlockAndValueMapping;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::MemRefType;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::OperationFolder;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::RewritePatternMatcher;
using mlir::Type;
using mlir::UnknownLoc;
using mlir::Value;
using util::AggregationKind;
using util::BufferPtr;
using util::CombinationKind;
using util::DataType;

struct DomainInfo {
  BlockAndValueMapping mapping;
};

struct TileBuilder::Impl {
  MLIRContext context;
  ModuleOp module;
  OpBuilder builder;
  llvm::DenseMap<Value, Value> implicitUpdates;
  llvm::DenseMap<Value, BufferPtr> implicitBindings;
  llvm::DenseMap<Value, RankedTensorType> shapeCache;
  NoneOp noneOp;
  Location loc;
  unsigned idxCounter = 0;

  Impl()
      : module(ModuleOp::create(UnknownLoc::get(&context))),
        builder(module.getBody()), loc(builder.getUnknownLoc()) {
    builder.setInsertionPointToStart(module.getBody());
  }

  Type inferElementType(ArrayRef<Type> types) {
    DataType ret = DataType::invalid;
    for (auto type : types) {
      auto rankedTensorType = eltwise::getRankedTensorType(type);
      auto dtype = rankedTensorType.getElementType().cast<ScalarType>().type();
      ret = promoteTypes(ret, dtype);
    }
    return builder.getType<ScalarType>(ret);
  }

  const AbstractOperation *lookupOperation(StringRef op) {
    auto opName = eltwise::Dialect::getCanonicalOpName(op);
    auto abstractOp = AbstractOperation::lookup(opName, &context);
    if (!abstractOp) {
      opName = tile::Dialect::getCanonicalOpName(op);
      abstractOp = AbstractOperation::lookup(opName, &context);
      if (!abstractOp) {
        throw std::runtime_error("Unknown EDSL primitive: " + op.str());
      }
    }
    return abstractOp;
  }

  std::vector<Value> getBackwardSliceOfAffine(const SetVector<Value> &values) {
    return util::getBackwardSlice(values, false, [](Value value) {
      if (auto scalarType = value.getType().dyn_cast<ScalarType>()) {
        return scalarType.type() == DataType::i32;
      }
      return false;
    });
  }

  Value makeIndexOp(ArrayRef<Value> args) {
    if (args.size() != 2) {
      throw std::runtime_error("'index' primitive expects 2 operands");
    }
    auto tensor = args[0];
    auto dim = args[1];
    auto resultType = IndexOp::getResultType(args.take_front());
    IntegerAttr dimAttr;
    if (!m_Constant(&dimAttr).match(dim.getDefiningOp())) {
      throw std::runtime_error(
          "'index' primitive expect argument 2 to be a constant integer");
    }
    auto op = builder.create<IndexOp>(loc, resultType, tensor, dimAttr);
    return op.result();
  }

  Value makePrngOp(ArrayRef<Value> args) {
    if (args.size() < 1) {
      throw std::runtime_error("'prng' primitive expects at least one operand");
    }
    auto state = args.front();
    auto dims = args.drop_front();
    auto resultType = PrngOp::getResultType(args);
    auto elementType = builder.getType<ScalarType>(DataType::u32);
    auto stateType = RankedTensorType::get({3, 2048}, elementType);
    auto op = builder.create<PrngOp>(loc, resultType, stateType, state, dims);
    implicitUpdates.insert(std::make_pair(op.new_state(), op.state()));
    return op.result();
  }

  Value makeScalarConstantOp(int64_t value) {
    auto type = builder.getType<ScalarType>(DataType::i32);
    return builder.create<ScalarConstantOp>(loc, type, value).result();
  }

  Value makeScalarConstantOp(double value) {
    auto type = builder.getType<ScalarType>(DataType::f32);
    return builder.create<ScalarConstantOp>(loc, type, value).result();
  }

  Value makeIdentity(ScalarType elemType, util::AggregationKind agg) {
    switch (agg) {
    case util::AggregationKind::assign:
    case util::AggregationKind::add:
      if (isFloat(elemType.type())) {
        return makeScalarConstantOp(0.0);
      } else {
        return makeScalarConstantOp(int64_t(0));
      }
    case util::AggregationKind::mul:
      if (isFloat(elemType.type())) {
        return makeScalarConstantOp(1.0);
      } else {
        return makeScalarConstantOp(int64_t(1));
      }
    case util::AggregationKind::min:
      if (isFloat(elemType.type())) {
        return makeScalarConstantOp(std::numeric_limits<double>::infinity());
      } else if (isSigned(elemType.type())) {
        return makeScalarConstantOp(std::numeric_limits<int64_t>::max());
      } else {
        return makeScalarConstantOp(
            static_cast<int64_t>(std::numeric_limits<uint64_t>::max()));
      }
    case util::AggregationKind::max:
      if (isFloat(elemType.type())) {
        return makeScalarConstantOp(-std::numeric_limits<double>::infinity());
      } else if (isSigned(elemType.type())) {
        return makeScalarConstantOp(std::numeric_limits<int64_t>::min());
      } else {
        return makeScalarConstantOp(int64_t(0));
      }
    }
    llvm_unreachable("Invalid aggregation kind");
  }
};

TileBuilder::TileBuilder() : impl(new Impl) {}

TileBuilder::~TileBuilder() = default;

void TileBuilder::Destroy(Value value) {
  IVLOG(5, "TileBuilder::Destroy> value");
  // impl->implicitBindings.erase(value);
  // TODO: fix memory mgmt issues, once purely MLIR path is complete
  // if (value && value->use_empty()) {
  //   auto op = value->getDefiningOp();
  //   if (op && op->use_empty()) {
  //     op->erase();
  //   }
  // }
}

MemRefType TileBuilder::MakeMemRefType(DataType dtype, ArrayRef<int64_t> sizes,
                                       ArrayRef<int64_t> strides) {
  auto elementType = impl->builder.getType<ScalarType>(dtype).toStandard();
  auto map = mlir::makeStridedLinearLayoutMap(strides, 0, &impl->context);
  return MemRefType::get(sizes, elementType, map);
}

MemRefType TileBuilder::IntoMemRefType(RankedTensorType type) {
  auto scalarType = type.getElementType().cast<ScalarType>();
  return MemRefType::get(type.getShape(), scalarType.toStandard());
}

void TileBuilder::BindShape(Value tensor, RankedTensorType type) {
  IVLOG(5, "TileBuilder::BindShape>");
  tensor.setType(type);
}

void TileBuilder::BindBuffer(Value tensor, BufferPtr buffer) {
  IVLOG(5, "TileBuilder::BindBuffer>");
  impl->implicitBindings[tensor] = buffer;
}

void TileBuilder::BindTensorDims(Value from, ArrayRef<Value *> intos) {
  if (!from) {
    throw std::runtime_error("BindTensorDim: from == nullptr");
  }
  IVLOG(5, "TileBuilder::BindTensorDim> from: " << mlir::debugString(from));
  for (unsigned i = 0; i < intos.size(); i++) {
    auto into = intos[i];
    if (!into) {
      throw std::runtime_error("BindTensorDim: into == nullptr");
    }
    if (*into) {
      IVLOG(6, "into: " << mlir::debugString(*into));
      auto fromType = from.getType().dyn_cast<RankedTensorType>();
      if (!fromType) {
        throw std::runtime_error("Unexpected type");
      }
      auto fromSize = fromType.getDimSize(i);
      if (!mlir::ShapedType::isDynamic(fromSize)) {
        auto op = (*into).getDefiningOp();
        if (!op) {
          throw std::runtime_error("No defining op");
        }
        if (auto const_op = llvm::dyn_cast<AffineConstantOp>(op)) {
          auto attr = const_op.getValue().dyn_cast<IntegerAttr>();
          if (!attr) {
            throw std::runtime_error(
                "Expected IntegerAttr for value of AffineConstantOp");
          }
          IVLOG(6, "dim: " << i << ", from: " << fromSize
                           << ", into: " << attr.getInt());
          if (fromSize != attr.getInt()) {
            std::string str;
            llvm::raw_string_ostream os(str);
            os << llvm::formatv(
                "bind_dims() mismatch on dim {0}. from: {1}, into: {2}", i,
                fromSize, attr.getInt());
            throw std::runtime_error(os.str());
          }
        }
      }
    }
    *into = MakeDimOp(from, i);
  }
}

RankedTensorType TileBuilder::ComputeShape(Value tensor) {
  IVLOG(5, "TileBuilder::ComputeShape>");
  auto type = eltwise::getRankedTensorType(tensor.getType());
  if (type.hasStaticShape()) {
    return type;
  }
  auto it = impl->shapeCache.find(tensor);
  if (it != impl->shapeCache.end()) {
    return it->second;
  }
  ProgramMutations mutations;
  mutations.outputs.emplace_back(tensor);
  auto program = MakeProgram("compute_shape", mutations);
  RankedTensorType shape;
  for (const auto &arg : program->arguments) {
    if (!arg.isInput) {
      shape = arg.shape;
      break;
    }
  }
  impl->shapeCache.insert(std::make_pair(tensor, shape));
  return shape;
}

Value TileBuilder::MakeCastOp(Value tensor, DataType dtype) {
  IVLOG(5, "TileBuilder::MakeCastOp> " << stringifyDataType(dtype).str());
  IVLOG(6, "  arg: " << mlir::debugString(tensor));
  auto elementType = impl->builder.getType<ScalarType>(dtype);
  auto tensorType = eltwise::getRankedTensorType(tensor.getType());
  auto resultType = RankedTensorType::get(tensorType.getShape(), elementType);
  return impl->builder.create<eltwise::CastOp>(impl->loc, resultType, tensor)
      .result();
}

Value TileBuilder::MakeTraceOp(Value tensor, const char *msg) {
  IVLOG(5, "TileBuilder::MakeTraceOp> " << msg);
  return impl->builder.create<TraceOp>(impl->loc, tensor, msg).out();
}

Value TileBuilder::MakePrimitiveOp(StringRef fn, ArrayRef<Value> args) {
  using PrimitiveBuilder = std::function<Value()>;
  auto builder =
      StringSwitch<PrimitiveBuilder>(fn)
          .Case("index", [this, args]() { return impl->makeIndexOp(args); })
          .Case("prng", [this, args]() { return impl->makePrngOp(args); })
          .Default([this, fn, args]() {
            auto abstractOp = impl->lookupOperation(fn);
            auto genericBuilder =
                abstractOp->getInterface<util::GenericBuilder>();
            if (!genericBuilder) {
              throw std::runtime_error("Unknown intrinsic: " + fn.str());
            }
            auto op = genericBuilder->create(&impl->builder, impl->loc, args);
            return op->getResult(0);
          });
  return builder();
}

Value TileBuilder::Clone(Value value) {
  IVLOG(5, "TileBuilder::Clone> " << mlir::debugString(value));
  return impl->builder.clone(*value.getDefiningOp())->getResult(0);
}

Value TileBuilder::MakeNoneOp() {
  IVLOG(5, "TileBuilder::MakeNoneOp>");
  if (!impl->noneOp) {
    auto type = impl->builder.getNoneType();
    impl->noneOp = impl->builder.create<NoneOp>(impl->loc, type);
  }
  return impl->noneOp.result();
}

Value TileBuilder::MakeStringOp(StringRef value) {
  IVLOG(5, "TileBuilder::MakeStringOp> " << value.str());
  auto type = StringType::get(&impl->context);
  auto attr = impl->builder.getStringAttr(value);
  return impl->builder.create<StringOp>(impl->loc, type, attr).result();
}

StringRef TileBuilder::GetStringValue(Value value) {
  if (auto op = llvm::dyn_cast_or_null<StringOp>(value.getDefiningOp())) {
    return op.getValue().getValue();
  }
  throw std::runtime_error("Expected StringOp");
}

Value TileBuilder::MakeTupleOp(ArrayRef<Value> elts) {
  IVLOG(5, "TileBuilder::MakeTupleOp> elts: " << elts.size());
  std::vector<Type> types;
  for (auto elt : elts) {
    types.push_back(elt.getType());
  }
  auto tupleType = impl->builder.getTupleType(types);
  return impl->builder.create<TupleOp>(impl->loc, tupleType, elts).result();
}

std::vector<Value> TileBuilder::GetTupleElements(Value value) {
  IVLOG(5, "TileBuilder::GetTupleElements> " << mlir::debugString(value));
  if (auto op = llvm::dyn_cast_or_null<TupleOp>(value.getDefiningOp())) {
    return std::vector<Value>(op.elts().begin(), op.elts().end());
  }
  throw std::runtime_error("Expected TupleOp");
}

Value TileBuilder::MakeScalarConstantOp(uint64_t value) {
  IVLOG(5, "TileBuilder::MakeScalarConstantOp> " << value);
  auto type = impl->builder.getType<ScalarType>(DataType::i32);
  return impl->builder.create<ScalarConstantOp>(impl->loc, type, value)
      .result();
}

Value TileBuilder::MakeScalarConstantOp(int64_t value) {
  IVLOG(5, "TileBuilder::MakeScalarConstantOp> " << value);
  return impl->makeScalarConstantOp(value);
}

int64_t TileBuilder::GetIntegerValue(Value value) {
  if (auto op =
          llvm::dyn_cast_or_null<ScalarConstantOp>(value.getDefiningOp())) {
    return op.getIntAttr().getInt();
  }
  throw std::runtime_error("Expected ScalarConstantOp");
}

Value TileBuilder::MakeScalarConstantOp(double value) {
  IVLOG(5, "TileBuilder::MakeScalarConstantOp> " << value);
  return impl->makeScalarConstantOp(value);
}

double TileBuilder::GetFloatValue(Value value) {
  if (auto op =
          llvm::dyn_cast_or_null<ScalarConstantOp>(value.getDefiningOp())) {
    return op.getFloatAttr().getValueAsDouble();
  }
  throw std::runtime_error("Expected ScalarConstantOp");
}

Value TileBuilder::MakeDimOp(Value tensor, unsigned dim) {
  IVLOG(5, "TileBuilder::MakeDimOp> tensor: " << mlir::debugString(tensor)
                                              << ", dim: " << dim);
  return impl->builder.create<DimOp>(impl->loc, tensor, dim).result();
}

RankedTensorType TileBuilder::MakeRankedTensorType(DataType dtype,
                                                   ArrayRef<int64_t> dims) {
  IVLOG(5, "TileBuilder::MakeRankedTensorType> "
               << stringifyDataType(dtype).str());
  auto elementType = impl->builder.getType<ScalarType>(dtype);
  // Convert dims: PlaidML semantics use 0 for unknown size, MLIR uses -1.
  SmallVector<int64_t, 4> shape(dims.begin(), dims.end());
  for (auto &dim : shape) {
    if (dim == 0) {
      dim = -1;
    }
  }
  return RankedTensorType::get(shape, elementType);
}

Value TileBuilder::MakePlaceholderOp(RankedTensorType type, BufferPtr buffer,
                                     StringRef name) {
  IVLOG(5, "TileBuilder::MakePlaceholderOp> " << name.str() << ": "
                                              << mlir::debugString(type));
  auto op = impl->builder.create<PlaceholderOp>(impl->loc, type);
  if (!name.empty()) {
    op.setAttr("name", impl->builder.getStringAttr(name));
  }
  if (buffer) {
    impl->implicitBindings[op.result()] = buffer;
  }
  return op.result();
}

Value TileBuilder::MakeAffineConstantOp(int64_t value) {
  IVLOG(5, "TileBuilder::MakeAffineConstantOp> " << value);
  return impl->builder.create<AffineConstantOp>(impl->loc, value).result();
}

Value TileBuilder::MakeAffineIndexOp(StringRef name) {
  IVLOG(5, "TileBuilder::MakeAffineIndexOp> " << name.str());
  return impl->builder
      .create<AffineIndexOp>(impl->loc, impl->idxCounter++, name)
      .result();
}

Value TileBuilder::MakeAffineAddOp(ArrayRef<Value> args) {
  IVLOG(5, "TileBuilder::MakeAffineAddOp>");
  return impl->builder.create<AffineAddOp>(impl->loc, args).result();
}

Value TileBuilder::MakeAffineSubOp(ArrayRef<Value> args) {
  IVLOG(5, "TileBuilder::MakeAffineSubOp>");
  return impl->builder.create<AffineSubOp>(impl->loc, args).result();
}

Value TileBuilder::MakeAffineMulOp(ArrayRef<Value> args) {
  IVLOG(5, "TileBuilder::MakeAffineMulOp>");
  return impl->builder.create<AffineMulOp>(impl->loc, args).result();
}

Value TileBuilder::MakeAffineDivOp(ArrayRef<Value> args) {
  IVLOG(5, "TileBuilder::MakeAffineDivOp>");
  return impl->builder.create<AffineDivOp>(impl->loc, args).result();
}

Value TileBuilder::MakeAffineNegOp(ArrayRef<Value> args) {
  IVLOG(5, "TileBuilder::MakeAffineNegOp>");
  return impl->builder.create<AffineNegOp>(impl->loc, args).result();
}

Value TileBuilder::MakeAffineMaxOp(ArrayRef<Value> args) {
  IVLOG(5, "TileBuilder::MakeAffineMaxOp>");
  return impl->builder.create<AffineMaxOp>(impl->loc, args).result();
}

Value TileBuilder::MakeAffineMinOp(ArrayRef<Value> args) {
  IVLOG(5, "TileBuilder::MakeAffineMinOp>");
  return impl->builder.create<AffineMinOp>(impl->loc, args).result();
}

Value TileBuilder::MakeAffineTensorMapOp(Value tensor, ArrayRef<Value> idxs) {
  IVLOG(5, "TileBuilder::MakeAffineTensorMapOp>");
  return impl->builder.create<AffineTensorMapOp>(impl->loc, tensor, idxs)
      .result();
}

Value TileBuilder::MakeAffineMapOp(ArrayRef<Value> idxs) {
  IVLOG(5, "TileBuilder::MakeAffineMapOp>");
  return impl->builder.create<AffineMapOp>(impl->loc, idxs).result();
}

void TileBuilder::AddConstraint(Value cion, Value lhs, Value rhs) {
  IVLOG(5, "TileBuilder::AddConstraint>");
  auto op = cion.getDefiningOp();
  auto cionOp = llvm::dyn_cast_or_null<SymbolicContractionOp>(op);
  if (!cionOp) {
    throw std::runtime_error(
        "add_constraint can only be specified on a contraction.");
  }

  auto consOp = llvm::cast<AffineConstraintsOp>(cionOp.cons().getDefiningOp());
  SmallVector<Value, 6> pairs{consOp.pairs()};
  pairs.emplace_back(lhs);
  pairs.emplace_back(rhs);
  consOp.getOperation()->setOperands(pairs);
}

void TileBuilder::SetUseDefault(Value cion, Value init) {
  IVLOG(2, "TileBuilder::SetUseDefault>");
  auto op = cion.getDefiningOp();
  auto cionOp = llvm::dyn_cast_or_null<SymbolicContractionOp>(op);
  if (!cionOp) {
    throw std::runtime_error(
        "no_reduce can only be specified on a contraction.");
  }
  cionOp.setOperand(0, init);
}

void TileBuilder::SetNoReduce(Value cion, bool no_reduce) {
  IVLOG(2, "TileBuilder::SetNoReduce> " << no_reduce);
  auto op = cion.getDefiningOp();
  auto cionOp = llvm::dyn_cast_or_null<SymbolicContractionOp>(op);
  if (!cionOp) {
    throw std::runtime_error(
        "no_reduce can only be specified on a contraction.");
  }
  if (no_reduce) {
    cionOp.setAttr("no_reduce", impl->builder.getUnitAttr());
  } else {
    cionOp.removeAttr("no_reduce");
  }
}

Value TileBuilder::MakeContractionOp(AggregationKind agg, CombinationKind combo,
                                     ArrayRef<Value> srcs, Value sink,
                                     Value sizes, StringRef name) {
  IVLOG(5, "TileBuilder::MakeContractionOp> "
               << stringifyAggregationKind(agg).str() << ":"
               << stringifyCombinationKind(combo).str()
               << ", name: " << name.str());
  IVLOG(5, "\n" << mlir::debugString(impl->module));
  // TODO: handle names (and idx_names)
  // Compute the sink shape of the contraction
  SmallVector<Type, 3> types;
  for (auto src : srcs) {
    auto mapOp = llvm::cast<AffineTensorMapOp>(src.getDefiningOp());
    types.push_back(mapOp.tensor().getType());
  }
  Type elementType;
  if (combo == CombinationKind::eq) {
    elementType = ScalarType::get(&impl->context, DataType::u1);
  } else if (combo == CombinationKind::cond) {
    auto rankedTensorType = eltwise::getRankedTensorType(types[2]);
    elementType = rankedTensorType.getElementType();
  } else {
    elementType = impl->inferElementType(types);
  }
  auto sizeMapOp = llvm::cast<AffineMapOp>(sizes.getDefiningOp());
  SmallVector<Value, 4> sizeDims(sizeMapOp.dims());
  auto shape = eltwise::ComputeShape(sizeDims);

  StringAttr nameAttr;
  if (name.size()) {
    nameAttr = impl->builder.getStringAttr(name);
  }
  Value ident = impl->makeIdentity(elementType.cast<ScalarType>(), agg);
  auto op = impl->builder.create<SymbolicContractionOp>(
      impl->loc, RankedTensorType::get(shape, elementType), ident,
      impl->builder.create<AffineConstraintsOp>(impl->loc), sizes, sink, srcs,
      impl->builder.getI64IntegerAttr(static_cast<int64_t>(agg)),
      impl->builder.getI64IntegerAttr(static_cast<int64_t>(combo)), UnitAttr{},
      nameAttr);
  return op.result();
}

class MakeProgramDriver : public PatternRewriter {
public:
  MakeProgramDriver(MLIRContext *ctx, const OwningRewritePatternList &patterns)
      : PatternRewriter(ctx), matcher(patterns), folder(ctx) {}

  void run(FuncOp funcOp) {
    funcOp.walk([&](Operation *op) {
      // Try to fold this op.
      if (succeeded(folder.tryToFold(op))) {
        return;
      }

      // Make sure that any new operations are inserted at this point.
      setInsertionPoint(op);

      // Try to match one of the patterns.
      matcher.matchAndRewrite(op, *this);
    });
  }

protected:
  Operation *insert(Operation *op) final { return OpBuilder::insert(op); }

private:
  RewritePatternMatcher matcher;
  OperationFolder folder;
};

struct MakeProgramPass : public mlir::FunctionPass<MakeProgramPass> {
  void runOnFunction() final {
    OwningRewritePatternList patterns;
    auto context = &getContext();
    for (auto op : context->getRegisteredOperations()) {
      op->getCanonicalizationPatterns(patterns, context);
    }

    MakeProgramDriver driver(context, patterns);
    driver.run(getFunction());
  }

  static std::unique_ptr<mlir::Pass> create() {
    return std::make_unique<MakeProgramPass>();
  }
};

std::shared_ptr<compiler::Program>
TileBuilder::MakeProgram(StringRef name, const ProgramMutations &mutations,
                         DataType floatx, DataType intx) {
  if (name.empty()) {
    name = "noname";
  }
  IVLOG(1, "TileBuilder::MakeProgram> " << name.str());
  IVLOG(6, "\n" << mlir::debugString(impl->module));
  // Wrap duplicate outputs and outputs that directly refer to inputs
  SetVector<Value> outputs;
  for (auto output : mutations.outputs) {
    if (!output) {
      throw std::runtime_error("Invalid output");
    }
    if (outputs.count(output) ||
        llvm::isa<PlaceholderOp>(output.getDefiningOp())) {
      outputs.insert(MakePrimitiveOp("ident", {output}));
    } else {
      outputs.insert(output);
    }
  }
  for (const auto &update : mutations.updates) {
    outputs.insert(update.source);
  }
  auto slice = util::getBackwardSlice(outputs, true);
  // Compute the input types
  std::vector<Type> inputTypes;
  for (auto value : slice) {
    auto op = value.getDefiningOp();
    if (auto placeholderOp = llvm::dyn_cast_or_null<PlaceholderOp>(op)) {
      inputTypes.push_back(placeholderOp.result().getType());
    }
  }
  // Construct a module
  auto loc = UnknownLoc::get(&impl->context);
  auto module = ModuleOp::create(loc);
  auto program = std::make_shared<compiler::Program>(module);
  program->entry = name;
  // Construct a function to represent the entire program
  auto initialFuncType = FunctionType::get(inputTypes, {}, &impl->context);
  auto funcOp = FuncOp::create(loc, name, initialFuncType, {});
  funcOp.addEntryBlock();
  OpBuilder builder(funcOp.getBody());
  std::set<std::string> names;
  auto attrName = Dialect::getDialectAttrName("name");
  unsigned argcnt = 0;
  std::map<Operation *, Operation *> opMap;
  BlockAndValueMapping mapper;
  for (auto value : slice) {
    auto op = value.getDefiningOp();
    // Only copy over top-level ops (those owned by the workspace module)
    if (op && op->getBlock() == impl->module.getBody()) {
      if (auto placeholderOp = llvm::dyn_cast<PlaceholderOp>(op)) {
        // Replace placeholders with block arguments
        auto blockArg = funcOp.getArgument(argcnt++);
        if (auto attr = placeholderOp.getAttrOfType<StringAttr>("name")) {
          auto uniqueName = util::getUniqueName(&names, attr.getValue());
          auto uniqueAttr = builder.getStringAttr(uniqueName);
          funcOp.setArgAttr(blockArg.getArgNumber(), attrName, uniqueAttr);
        }
        IVLOG(5, "BlockArgument mapping: " << value << " -> " << blockArg);
        mapper.map(value, blockArg);
        compiler::ProgramArgument programArg{
            true, value, value.getType().cast<RankedTensorType>()};
        auto itBinding = impl->implicitBindings.find(value);
        if (itBinding != impl->implicitBindings.end()) {
          programArg.buffer = itBinding->second;
        }
        program->arguments.emplace_back(programArg);
      } else {
        Operation *newOp;
        auto it = opMap.find(op);
        if (it != opMap.end()) {
          newOp = it->second;
        } else {
          newOp = builder.clone(*op, mapper);
          opMap.emplace(op, newOp);
        }
        for (unsigned i = 0; i < op->getNumResults(); i++) {
          auto oldResult = op->getResult(i);
          auto newResult = newOp->getResult(i);
          if (oldResult == value) {
            IVLOG(5, "mapping: " << value << " -> " << newResult);
            IVLOG(6, "value: " << mlir::debugString(value));
            IVLOG(6, "newResult: " << mlir::debugString(newResult));
            mapper.map(value, newResult);
          } else {
            auto itUpdate = impl->implicitUpdates.find(oldResult);
            if (itUpdate != impl->implicitUpdates.end()) {
              mapper.map(oldResult, newResult);
              outputs.insert(oldResult);
            }
          }
        }
      }
    }
  }
  // Add a final ReturnOp
  std::vector<Value> returnOperands;
  std::vector<Type> resultTypes;
  for (auto output : outputs) {
    auto value = mapper.lookupOrNull(output);
    if (!value) {
      throw std::runtime_error("Output not found in mapper");
    }
    resultTypes.emplace_back(value.getType());
    returnOperands.emplace_back(value);
  }
  auto returnOp = builder.create<mlir::ReturnOp>(loc, returnOperands);
  // compute final function type
  auto finalFuncType =
      FunctionType::get(inputTypes, resultTypes, &impl->context);
  funcOp.setType(finalFuncType);
  // Attach the function to the module
  module.push_back(funcOp);
  IVLOG(5, "\n" << mlir::debugString(module));
  if (failed(mlir::verify(module))) {
    throw std::runtime_error("Module verification error");
  }
  // Do some optimization passes
  mlir::PassManager pm(&impl->context);
  pm.addPass(createConstantTypesPass(floatx, intx));
  pm.addPass(MakeProgramPass::create());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  auto result = pm.run(module);
  if (failed(result)) {
    IVLOG(1, "\n" << mlir::debugString(module));
    throw std::runtime_error("Optimization passes failure");
  }
  for (unsigned i = 0; i < returnOp.getNumOperands(); i++) {
    auto userValue = outputs[i];
    auto finalValue = returnOp.getOperand(i);
    auto itUpdate = impl->implicitUpdates.find(outputs[i]);
    if (itUpdate != impl->implicitUpdates.end()) {
      userValue = itUpdate->second;
    }
    compiler::ProgramArgument programArg{
        false, userValue, finalValue.getType().cast<RankedTensorType>()};
    auto itBinding = impl->implicitBindings.find(finalValue);
    if (itBinding != impl->implicitBindings.end()) {
      programArg.buffer = itBinding->second;
    }
    program->arguments.emplace_back(programArg);
  }
  program->tileIR = mlir::debugString(module);
  // IVLOG(2, "TileBuilder::MakeProgram>\n" << mlir::debugString(module));
  return program;
}

std::vector<Value> TileBuilder::ComputeGradients(ArrayRef<Value> wrt,
                                                 Value loss) {
  IVLOG(2, "TileBuilder::ComputeGradients>");
  auto value = loss;
  auto ndims = ComputeShape(loss).getShape().size();
  if (ndims) {
    std::vector<Value> src_idxs;
    for (size_t i = 0; i < ndims; ++i) {
      src_idxs.emplace_back(MakeAffineIndexOp(""));
    }
    auto src = MakeAffineTensorMapOp(loss, src_idxs);
    auto sink = MakeAffineMapOp(ArrayRef<Value>{});
    auto sizes = MakeAffineMapOp(ArrayRef<Value>{});
    auto cion = MakeContractionOp(AggregationKind::add, CombinationKind::none,
                                  {src}, sink, sizes, "net_loss");
    value = cion;
  }
  Gradient grad(value, this);
  std::vector<Value> ret(wrt.size());
  for (size_t i = 0; i < wrt.size(); i++) {
    ret[i] = grad.GetDerivative(wrt[i]);
  }
  return ret;
}

void TileBuilder::Dump() { IVLOG(5, "\n" << mlir::debugString(impl->module)); }

} // namespace pmlc::dialect::tile
