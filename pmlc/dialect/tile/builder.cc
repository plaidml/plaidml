// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/builder.h"

#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/Passes.h"

#include "base/util/env.h"
#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/dialect.h"
#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/tile/dialect.h"
#include "pmlc/dialect/tile/ops.h"
#include "pmlc/dialect/tile/program.h"
#include "pmlc/util/slice.h"
#include "pmlc/util/util.h"
#include "tile/base/shape.h"

namespace pmlc::dialect::tile {

using eltwise::ScalarConstantOp;
using eltwise::ScalarType;
using llvm::ArrayRef;
using llvm::SmallVector;
using llvm::StringRef;
using mlir::Block;
using mlir::BlockAndValueMapping;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::RankedTensorType;
using mlir::Type;
using mlir::UnknownLoc;
using mlir::Value;
using util::AggregationKind;
using util::CombinationKind;
using vertexai::tile::BufferPtr;

struct DomainInfo {
  BlockAndValueMapping mapping;
};

using ContractionKey = std::pair<AggregationKind, CombinationKind>;

struct TileBuilder::Impl {
  MLIRContext context;
  ModuleOp module;
  OpBuilder builder;
  std::map<AffineDomainOp, DomainInfo> domains;
  IoMap ioMap;
  static std::map<ContractionKey, std::string> contractions;
  NoneOp noneOp;

  Impl()
      : module(ModuleOp::create(UnknownLoc::get(&context))),  //
        builder(module.getBody()) {
    builder.setInsertionPointToStart(module.getBody());
  }

  Type inferElementType(ArrayRef<Type> types) {
    DataType ret = DataType::INVALID;
    for (auto type : types) {
      auto rankedTensorType = eltwise::getRankedTensorType(type);
      auto dtype = rankedTensorType.getElementType().cast<ScalarType>().type();
      ret = CommonSupertype(ret, dtype);
    }
    return builder.getType<ScalarType>(ret);
  }

  const mlir::AbstractOperation* lookupOperation(StringRef op) {
    auto opName = eltwise::Dialect::getCanonicalOpName(op);
    auto abstractOp = mlir::AbstractOperation::lookup(opName, &context);
    if (!abstractOp) {
      opName = tile::Dialect::getCanonicalOpName(op);
      abstractOp = mlir::AbstractOperation::lookup(opName, &context);
      if (!abstractOp) {
        throw std::runtime_error("Unknown op: " + op.str());
      }
    }
    return abstractOp;
  }

  std::vector<mlir::Value*> getBackwardSliceOfAffine(const llvm::SetVector<mlir::Value*>& values) {
    return util::getBackwardSlice(values, false, [](Value* value) {
      if (auto scalarType = value->getType().dyn_cast<ScalarType>()) {
        return scalarType.type() == DataType::INTX;
      }
      return false;
    });
  }
};

std::map<ContractionKey, std::string> TileBuilder::Impl::contractions{
    // assign
    {std::make_pair(AggregationKind::assign, CombinationKind::none), "=(x)"},
    {std::make_pair(AggregationKind::assign, CombinationKind::add), "=(x+y)"},
    {std::make_pair(AggregationKind::assign, CombinationKind::cond), "=(x==y?z)"},
    {std::make_pair(AggregationKind::assign, CombinationKind::eq), "=(x==y)"},
    {std::make_pair(AggregationKind::assign, CombinationKind::mul), "=(x*y)"},
    // max
    {std::make_pair(AggregationKind::max, CombinationKind::none), ">(x)"},
    {std::make_pair(AggregationKind::max, CombinationKind::add), ">(x+y)"},
    {std::make_pair(AggregationKind::max, CombinationKind::cond), ">(x==y?z)"},
    {std::make_pair(AggregationKind::max, CombinationKind::eq), ">(x==y)"},
    {std::make_pair(AggregationKind::max, CombinationKind::mul), ">(x*y)"},
    // min
    {std::make_pair(AggregationKind::min, CombinationKind::none), "<(x)"},
    {std::make_pair(AggregationKind::min, CombinationKind::add), "<(x+y)"},
    {std::make_pair(AggregationKind::min, CombinationKind::cond), "<(x==y?z)"},
    {std::make_pair(AggregationKind::min, CombinationKind::eq), "<(x==y)"},
    {std::make_pair(AggregationKind::min, CombinationKind::mul), "<(x*y)"},
    // prod
    {std::make_pair(AggregationKind::mul, CombinationKind::none), "*(x)"},
    {std::make_pair(AggregationKind::mul, CombinationKind::add), "*(x+y)"},
    {std::make_pair(AggregationKind::mul, CombinationKind::cond), "*(x==y?z)"},
    {std::make_pair(AggregationKind::mul, CombinationKind::eq), "*(x==y)"},
    {std::make_pair(AggregationKind::mul, CombinationKind::mul), "*(x*y)"},
    // sum
    {std::make_pair(AggregationKind::add, CombinationKind::none), "+(x)"},
    {std::make_pair(AggregationKind::add, CombinationKind::add), "+(x+y)"},
    {std::make_pair(AggregationKind::add, CombinationKind::cond), "+(x==y?z)"},
    {std::make_pair(AggregationKind::add, CombinationKind::eq), "+(x==y)"},
    {std::make_pair(AggregationKind::add, CombinationKind::mul), "+(x*y)"},
};

TileBuilder::TileBuilder() : impl(new Impl) {}

TileBuilder::~TileBuilder() = default;

void TileBuilder::Destroy(Value* value) {
  IVLOG(5, "TileBuilder::Destroy> value");
  // impl->ioMap.erase(value);
  // TODO: fix memory mgmt issues, once purely MLIR path is complete
  // if (value && value->use_empty()) {
  //   auto op = value->getDefiningOp();
  //   if (op && op->use_empty()) {
  //     op->erase();
  //   }
  // }
}

stripe::TensorType TileBuilder::MakeTensorType(  //
    DataType dtype,                              //
    llvm::ArrayRef<int64_t> sizes,               //
    llvm::ArrayRef<int64_t> strides) {
  auto cls = mlir::Identifier::get(stripe::kAddressClassIdentifier, &impl->context);
  llvm::SmallVector<stripe::TensorDim, 4> dims;
  for (unsigned i = 0; i < sizes.size(); i++) {
    dims.emplace_back(stripe::TensorDim{sizes[i], strides[i], cls});
  }
  auto elementType = impl->builder.getType<ScalarType>(dtype);
  return stripe::TensorType::get(elementType, dims, stripe::OffsetsMap{}, false);
}

stripe::TensorType TileBuilder::IntoTensorType(mlir::RankedTensorType type) {
  auto shape = type.getShape();
  auto cls = mlir::Identifier::get(stripe::kAddressClassIdentifier, type.getContext());
  llvm::SmallVector<stripe::TensorDim, 4> newShape(shape.size(), stripe::TensorDim{0, 0, cls});
  int64_t stride = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    newShape[i].stride = stride;
    newShape[i].size = shape[i];
    stride *= shape[i];
  }
  return stripe::TensorType::get(type.getElementType(), newShape, stripe::OffsetsMap{}, false);
}

void TileBuilder::BindShape(mlir::Value* tensor, mlir::RankedTensorType type) {  //
  tensor->setType(type);
}

void TileBuilder::BindTensorDims(Value* from, ArrayRef<Value**> intos) {
  if (!from) {
    throw std::runtime_error("BindTensorDim: from == nullptr");
  }
  IVLOG(5, "TileBuilder::BindTensorDim> from: " << mlir::debugString(*from));
  for (unsigned i = 0; i < intos.size(); i++) {
    auto into = intos[i];
    if (!into) {
      throw std::runtime_error("BindTensorDim: into == nullptr");
    }
    if (*into) {
      IVLOG(6, "into: " << mlir::debugString(**into));
      auto fromType = from->getType().dyn_cast<RankedTensorType>();
      if (!fromType) {
        throw std::runtime_error("Unexpected type");
      }
      auto fromSize = fromType.getDimSize(i);
      if (!mlir::ShapedType::isDynamic(fromSize)) {
        auto op = (*into)->getDefiningOp();
        if (!op) {
          throw std::runtime_error("No defining op");
        }
        if (auto const_op = llvm::dyn_cast<AffineConstantOp>(op)) {
          auto attr = const_op.getValue().dyn_cast<IntegerAttr>();
          if (!attr) {
            throw std::runtime_error("Expected IntegerAttr for value of AffineConstantOp");
          }
          IVLOG(6, "dim: " << i << ", from: " << fromSize << ", into: " << attr.getInt());
          if (fromSize != attr.getInt()) {
            std::string str;
            llvm::raw_string_ostream os(str);
            os << llvm::formatv("bind_dims() mismatch on dim {0}. from: {1}, into: {2}", i, fromSize, attr.getInt());
            throw std::runtime_error(os.str());
          }
        }
      }
    }
    *into = MakeDimOp(from, i);
  }
}

RankedTensorType TileBuilder::ComputeShape(Value* tensor) {
  IVLOG(5, "TileBuilder::ComputeShape>");
  auto type = eltwise::getRankedTensorType(tensor->getType());
  if (type.hasStaticShape()) {
    return type;
  }
  SmallVector<Value*, 1> outputs{tensor};
  auto program = MakeProgram("compute_shape", outputs, outputs);
  return outputs[0]->getType().dyn_cast<RankedTensorType>();
}

Value* TileBuilder::MakeCastOp(Value* tensor, DataType dtype) {
  IVLOG(5, "TileBuilder::MakeCastOp> " << to_string(dtype));
  IVLOG(6, "  arg: " << mlir::debugString(*tensor));
  auto elementType = impl->builder.getType<ScalarType>(dtype);
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto resultType = RankedTensorType::get(tensorType.getShape(), elementType);
  return impl->builder.create<eltwise::CastOp>(impl->builder.getUnknownLoc(), resultType, tensor).result();
}

Value* TileBuilder::MakePrimitiveOp(StringRef fn, ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakePrimitiveOp> " << fn.str());
  for (auto arg : args) {
    IVLOG(6, "  arg: " << mlir::debugString(*arg));
  }
  if (fn == "index") {
    if (args.size() != 2) {
      throw std::runtime_error("index op expects 2 operands");
    }
    auto tensor = args[0];
    auto dim = args[1];
    auto resultType = IndexOp::getResultType(args.take_front());
    IntegerAttr dimAttr;
    if (!m_Constant(&dimAttr).match(dim->getDefiningOp())) {
      throw std::runtime_error("index op expect argument 2 to be a constant integer");
    }
    auto op = impl->builder.create<IndexOp>(impl->builder.getUnknownLoc(), resultType, tensor, dimAttr);
    return op.result();
  }
  auto abstractOp = impl->lookupOperation(fn);
  auto genericBuilder = abstractOp->getInterface<util::GenericBuilder>();
  if (!genericBuilder) {
    throw std::runtime_error("Unknown intrinsic: " + fn.str());
  }
  auto type = impl->builder.getType<ScalarType>(DataType::FLOAT32);  // TODO
  auto op = genericBuilder->create(&impl->builder, impl->builder.getUnknownLoc(), type, args);
  return op->getResult(0);
}

Value* TileBuilder::Clone(Value* value) {
  IVLOG(5, "TileBuilder::Clone> " << mlir::debugString(*value));
  return impl->builder.clone(*value->getDefiningOp())->getResult(0);
}

Value* TileBuilder::MakeNoneOp() {
  IVLOG(5, "TileBuilder::MakeNoneOp>");
  if (!impl->noneOp) {
    auto type = impl->builder.getNoneType();
    impl->noneOp = impl->builder.create<NoneOp>(impl->builder.getUnknownLoc(), type);
  }
  return impl->noneOp.result();
}

Value* TileBuilder::MakeStringOp(StringRef value) {
  IVLOG(5, "TileBuilder::MakeStringOp> " << value.str());
  auto type = StringType::get(&impl->context);
  auto attr = impl->builder.getStringAttr(value);
  return impl->builder.create<StringOp>(impl->builder.getUnknownLoc(), type, attr).result();
}

llvm::StringRef TileBuilder::GetStringValue(mlir::Value* value) {
  if (auto op = llvm::dyn_cast_or_null<StringOp>(value->getDefiningOp())) {
    return op.getValue().getValue();
  }
  throw std::runtime_error("Expected StringOp");
}

Value* TileBuilder::MakeTupleOp(ArrayRef<Value*> elts) {
  IVLOG(5, "TileBuilder::MakeTupleOp> elts: " << elts.size());
  std::vector<Type> types;
  for (auto elt : elts) {
    types.push_back(elt->getType());
  }
  auto tupleType = impl->builder.getTupleType(types);
  return impl->builder.create<TupleOp>(impl->builder.getUnknownLoc(), tupleType, elts).result();
}

std::vector<Value*> TileBuilder::GetTupleElements(Value* value) {
  IVLOG(5, "TileBuilder::GetTupleElements> " << mlir::debugString(*value));
  if (auto op = llvm::dyn_cast_or_null<TupleOp>(value->getDefiningOp())) {
    return std::vector<Value*>(op.elts().begin(), op.elts().end());
  }
  throw std::runtime_error("Expected TupleOp");
}

Value* TileBuilder::MakeScalarConstantOp(int64_t value) {
  IVLOG(5, "TileBuilder::MakeScalarConstantOp> " << value);
  auto type = impl->builder.getType<ScalarType>(DataType::INTX);
  return impl->builder.create<ScalarConstantOp>(impl->builder.getUnknownLoc(), type, value).result();
}

int64_t TileBuilder::GetIntegerValue(mlir::Value* value) {
  if (auto op = llvm::dyn_cast_or_null<ScalarConstantOp>(value->getDefiningOp())) {
    return op.getIntAttr().getInt();
  }
  throw std::runtime_error("Expected ScalarConstantOp");
}

Value* TileBuilder::MakeScalarConstantOp(double value) {
  IVLOG(5, "TileBuilder::MakeScalarConstantOp> " << value);
  auto type = impl->builder.getType<ScalarType>(DataType::FLOATX);
  return impl->builder.create<ScalarConstantOp>(impl->builder.getUnknownLoc(), type, value).result();
}

double TileBuilder::GetFloatValue(mlir::Value* value) {
  if (auto op = llvm::dyn_cast_or_null<ScalarConstantOp>(value->getDefiningOp())) {
    return op.getFloatAttr().getValueAsDouble();
  }
  throw std::runtime_error("Expected ScalarConstantOp");
}

Value* TileBuilder::MakeDimOp(Value* tensor, unsigned dim) {
  IVLOG(5, "TileBuilder::MakeDimOp> tensor: " << mlir::debugString(*tensor) << ", dim: " << dim);
  return impl->builder.create<DimOp>(impl->builder.getUnknownLoc(), tensor, dim).result();
}

RankedTensorType TileBuilder::MakeRankedTensorType(DataType dtype, ArrayRef<int64_t> dims) {
  IVLOG(5, "TileBuilder::MakeRankedTensorType> " << to_string(dtype));
  auto elementType = impl->builder.getType<ScalarType>(dtype);
  // Convert dims: PlaidML semantics use 0 for unknown size, MLIR uses -1.
  SmallVector<int64_t, 4> shape(dims.begin(), dims.end());
  for (auto& dim : shape) {
    if (dim == 0) {
      dim = -1;
    }
  }
  return RankedTensorType::get(shape, elementType);
}

Value* TileBuilder::MakePlaceholderOp(RankedTensorType type, BufferPtr buffer, StringRef name) {
  IVLOG(5, "TileBuilder::MakePlaceholderOp> " << name.str() << ": " << mlir::debugString(type));
  auto op = impl->builder.create<PlaceholderOp>(impl->builder.getUnknownLoc(), type);
  if (!name.empty()) {
    op.setAttr("name", impl->builder.getStringAttr(name));
  }
  if (buffer) {
    impl->ioMap.emplace(op.result(), buffer);
  }
  return op.result();
}

Value* TileBuilder::MakeAffineConstantOp(int64_t value) {
  IVLOG(5, "TileBuilder::MakeAffineConstantOp> " << value);
  return impl->builder.create<AffineConstantOp>(impl->builder.getUnknownLoc(), value).result();
}

Value* TileBuilder::MakeAffineIndexOp(StringRef name) {
  IVLOG(5, "TileBuilder::MakeAffineIndexOp> " << name.str());
  auto op = impl->builder.create<AffineIndexOp>(impl->builder.getUnknownLoc());
  if (!name.empty()) {
    op.setAttr("name", impl->builder.getStringAttr(name));
  }
  return op.result();
}

Value* TileBuilder::MakeAffineAddOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineAddOp>");
  return impl->builder.create<AffineAddOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineSubOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineSubOp>");
  return impl->builder.create<AffineSubOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineMulOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineMulOp>");
  return impl->builder.create<AffineMulOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineDivOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineDivOp>");
  return impl->builder.create<AffineDivOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineNegOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineNegOp>");
  return impl->builder.create<AffineNegOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineMaxOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineMaxOp>");
  return impl->builder.create<AffineMaxOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineMinOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineMinOp>");
  return impl->builder.create<AffineMinOp>(impl->builder.getUnknownLoc(), args).result();
}

Value* TileBuilder::MakeAffineSourceIndexMapOp(Value* tensor, ArrayRef<Value*> idxs) {
  IVLOG(5, "TileBuilder::MakeAffineSourceIndexMapOp>");
  return impl->builder.create<AffineSourceIndexMapOp>(impl->builder.getUnknownLoc(), tensor, idxs).result();
}

Value* TileBuilder::MakeAffineSinkIndexMapOp(ArrayRef<Value*> idxs) {
  IVLOG(5, "TileBuilder::MakeAffineSinkIndexMapOp>");
  return impl->builder.create<AffineSinkIndexMapOp>(impl->builder.getUnknownLoc(), idxs).result();
}

Value* TileBuilder::MakeAffineSizeMapOp(ArrayRef<Value*> sizes) {
  IVLOG(5, "TileBuilder::MakeAffineSizeMapOp>");
  return impl->builder.create<AffineSizeMapOp>(impl->builder.getUnknownLoc(), sizes).result();
}

void TileBuilder::AddConstraint(Value* cion, Value* lhs, Value* rhs) {
  IVLOG(5, "TileBuilder::AddConstraint>");
  auto op = cion->getDefiningOp();
  auto domainOp = llvm::dyn_cast_or_null<AffineDomainOp>(op);
  if (!domainOp) {
    throw std::runtime_error("add_constraint can only be specified on a contraction.");
  }

  auto& region = domainOp.body();
  auto src = &region.front();
  OpBuilder builder(src->getTerminator());

  // Get a backward slice to trace the transitive defs of the lhs and rhs.
  auto& info = impl->domains[domainOp];
  llvm::SetVector<Value*> values;
  values.insert(lhs);
  values.insert(rhs);
  auto slice = impl->getBackwardSliceOfAffine(values);

  // Previously, some values will have already been cloned into the AffineDomainOp
  // However, there might be other ops that this constraint introduced that needs
  // to be cloned into the AffineDomainOp.
  for (auto value : slice) {
    if (!info.mapping.contains(value)) {
      IVLOG(5, "clone: " << mlir::debugString(*value));
      auto op = value->getDefiningOp();
      auto newValue = builder.clone(*op, info.mapping)->getResult(0);
      info.mapping.map(value, newValue);
    }
  }

  // Create the ConstraintOp as a parent of the existing terminator.
  auto constraintOp = builder.create<ConstraintOp>(op->getLoc(), info.mapping.lookup(lhs), info.mapping.lookup(rhs));
  auto it = std::prev(src->end(), 1);
  auto block = builder.createBlock(&constraintOp.body());
  auto& dst = block->getOperations();
  dst.splice(dst.end(), src->getOperations(), it, src->end());
}

void TileBuilder::SetUseDefault(Value* cion, Value* defaultValue) {
  IVLOG(2, "TileBuilder::SetUseDefault>");
  auto op = cion->getDefiningOp();
  auto domainOp = llvm::dyn_cast_or_null<AffineDomainOp>(op);
  if (!domainOp) {
    throw std::runtime_error("use_default can only be specified on a contraction.");
  }
  auto terminator = domainOp.body().front().getTerminator();
  while (!llvm::isa<ContractionOp>(terminator)) {
    terminator = terminator->getRegion(0).front().getTerminator();
  }
  SmallVector<Value*, 6> operands{terminator->getOperands()};
  operands.emplace_back(defaultValue);
  terminator->setOperands(operands);
}

void TileBuilder::SetNoReduce(mlir::Value* cion, bool no_reduce) {
  IVLOG(2, "TileBuilder::SetNoReduce> " << no_reduce);
  auto op = cion->getDefiningOp();
  auto domainOp = llvm::dyn_cast_or_null<AffineDomainOp>(op);
  if (!domainOp) {
    throw std::runtime_error("no_reduce can only be specified on a contraction.");
  }
  domainOp.setAttr("no_reduce", impl->builder.getBoolAttr(no_reduce));
}

Value* TileBuilder::MakeContractionOp(  //
    util::AggregationKind agg,          //
    util::CombinationKind combo,        //
    ArrayRef<Value*> srcs,              //
    Value* sink,                        //
    Value* sizes,                       //
    StringRef name) {
  IVLOG(5, "TileBuilder::MakeContractionOp> " << util::stringifyAggregationKind(agg).str() << ":"
                                              << util::stringifyCombinationKind(combo).str()
                                              << ", name: " << name.str());
  IVLOG(5, mlir::debugString(impl->module));
  auto it = Impl::contractions.find(std::make_pair(agg, combo));
  if (it == Impl::contractions.end()) {
    throw std::runtime_error("Unsupported contraction");
  }
  auto abstractOp = impl->lookupOperation(it->second);
  auto contractionBuilder = abstractOp->getInterface<tile::ContractionOp>();
  if (!contractionBuilder) {
    throw std::runtime_error("Unsupported contraction");
  }
  // Compute the sink shape of the contraction
  SmallVector<Type, 3> types;
  for (auto src : srcs) {
    IVLOG(6, "  src: " << mlir::debugString(*src));
    auto map_op = llvm::cast<AffineSourceIndexMapOp>(src->getDefiningOp());
    types.push_back(map_op.tensor()->getType());
  }
  IVLOG(6, "  sink: " << mlir::debugString(*sink));
  IVLOG(6, "  sizes: " << mlir::debugString(*sizes));
  Type elementType;
  if (combo == CombinationKind::eq) {
    elementType = ScalarType::get(&impl->context, DataType::BOOLEAN);
  } else if (combo == CombinationKind::cond) {
    auto rankedTensorType = eltwise::getRankedTensorType(types[2]);
    elementType = rankedTensorType.getElementType();
  } else {
    elementType = impl->inferElementType(types);
  }
  auto size_map_op = llvm::cast<AffineSizeMapOp>(sizes->getDefiningOp());
  SmallVector<Value*, 4> size_map_sizes(size_map_op.sizes());
  auto shape = eltwise::ComputeShape(size_map_sizes);
  auto tensorType = RankedTensorType::get(shape, elementType);
  auto domainOp = impl->builder.create<AffineDomainOp>(impl->builder.getUnknownLoc(), tensorType, BoolAttr{});
  auto& info = impl->domains[domainOp];
  auto body = new Block();
  domainOp.body().push_back(body);
  llvm::SetVector<Value*> values;
  values.insert(srcs.begin(), srcs.end());
  values.insert(sink);
  values.insert(sizes);
  auto slice = impl->getBackwardSliceOfAffine(values);
  // Find and replace each AffineIndexOp with a BlockArgument of the domain op
  SmallVector<Attribute, 8> idxNames;
  std::queue<Value*> worklist;
  for (auto value : slice) {
    auto op = value->getDefiningOp();
    if (auto indexOp = llvm::dyn_cast_or_null<AffineIndexOp>(op)) {
      auto arg = body->addArgument(indexOp.getType());
      info.mapping.map(value, arg);
      worklist.push(value);
      if (auto attr = indexOp.getAttrOfType<StringAttr>("name")) {
        idxNames.emplace_back(attr);
      } else {
        auto name = llvm::formatv("x{0}", arg->getArgNumber());
        idxNames.emplace_back(impl->builder.getStringAttr(name.str()));
      }
    }
  }
  if (!name.empty()) {
    domainOp.setAttr("name", impl->builder.getStringAttr(name));
  }
  domainOp.setAttr("idx_names", mlir::ArrayAttr::get(idxNames, &impl->context));
  // Move across only the values/ops that depend on AffineIndexOps
  // First determine the transitive users of AffineIndexOps
  std::set<Value*> belong;
  while (worklist.size()) {
    auto value = worklist.front();
    worklist.pop();
    for (auto user : value->getUsers()) {
      auto user_value = user->getResult(0);
      if (!belong.count(user_value)) {
        belong.insert(user_value);
        worklist.push(user_value);
      }
    }
  }
  // Now move across ops but do so in topologically sorted order
  OpBuilder domainBuilder(body);
  for (auto value : slice) {
    auto op = value->getDefiningOp();
    if (belong.count(value) ||                    //
        llvm::isa<AffineSourceIndexMapOp>(op) ||  //
        llvm::isa<AffineSinkIndexMapOp>(op) ||    //
        llvm::isa<AffineSizeMapOp>(op)) {
      auto new_value = domainBuilder.clone(*op, info.mapping)->getResult(0);
      info.mapping.map(value, new_value);
    }
  }
  auto new_sizes = info.mapping.lookup(sizes);
  auto new_sink = info.mapping.lookup(sink);
  SmallVector<Value*, 3> new_srcs;
  for (auto src : srcs) {
    new_srcs.emplace_back(info.mapping.lookup(src));
  }
  contractionBuilder->create(&domainBuilder, impl->builder.getUnknownLoc(), new_sizes, new_srcs, new_sink);
  IVLOG(5, mlir::debugString(domainOp));
  return domainOp.result();
}

std::shared_ptr<TileProgram> TileBuilder::MakeProgram(  //
    StringRef name,                                     //
    ArrayRef<Value*> outputs,                           //
    llvm::MutableArrayRef<Value*> new_outputs) {
  IVLOG(5, "TileBuilder::MakeProgram> " << name.str());
  IVLOG(6, mlir::debugString(impl->module));
  // Compute the result types
  std::vector<Type> resultTypes(outputs.size());
  llvm::SetVector<Value*> values;
  for (unsigned i = 0; i < outputs.size(); i++) {
    if (!outputs[i]) {
      throw std::runtime_error("Invalid output");
    }
    resultTypes[i] = outputs[i]->getType();
    if (values.count(outputs[i]) || llvm::isa<PlaceholderOp>(outputs[i]->getDefiningOp())) {
      values.insert(MakePrimitiveOp("ident", {outputs[i]}));
    } else {
      values.insert(outputs[i]);
    }
  }
  auto slice = util::getBackwardSlice(values, true);
  // Compute the input types
  std::vector<Type> inputTypes;
  for (auto value : slice) {
    auto op = value->getDefiningOp();
    if (auto placeholderOp = llvm::dyn_cast_or_null<PlaceholderOp>(op)) {
      inputTypes.push_back(placeholderOp.result()->getType());
    }
  }
  // Construct a module
  auto loc = mlir::UnknownLoc::get(&impl->context);
  auto module = ModuleOp::create(loc);
  auto program = std::make_shared<TileProgram>(module);
  // Construct a function to represent the entire program
  auto funcType = mlir::FunctionType::get(inputTypes, resultTypes, &impl->context);
  auto funcOp = mlir::FuncOp::create(loc, name, funcType, {});
  funcOp.addEntryBlock();
  OpBuilder builder(funcOp.getBody());
  std::set<std::string> names;
  auto attrName = Dialect::getDialectAttrName("name");
  unsigned argcnt = 0;
  for (auto value : slice) {
    auto it = impl->ioMap.find(value);
    if (it != impl->ioMap.end()) {
      program->ioMap.emplace(it->first, it->second);
    }
    auto op = value->getDefiningOp();
    // Only copy over top-level ops (those owned by the workspace module)
    if (op && op->getBlock() == impl->module.getBody()) {
      if (auto placeholderOp = llvm::dyn_cast<PlaceholderOp>(op)) {
        // Replace placeholders with block arguments
        auto new_value = funcOp.getArgument(argcnt++);
        if (auto attr = placeholderOp.getAttrOfType<StringAttr>("name")) {
          auto uniqueName = util::getUniqueName(&names, attr.getValue());
          auto uniqueAttr = builder.getStringAttr(uniqueName);
          funcOp.setArgAttr(new_value->getArgNumber(), attrName, uniqueAttr);
        }
        IVLOG(5, "BlockArgument mapping: " << value << " -> " << new_value);
        program->mapper.map(value, new_value);
      } else {
        auto new_value = builder.clone(*op, program->mapper)->getResult(0);
        IVLOG(5, "mapping: " << value << " -> " << new_value);
        program->mapper.map(value, new_value);
      }
    }
  }
  // Add a final ReturnOp
  std::vector<Value*> rets;
  for (unsigned i = 0; i < values.size(); i++) {
    rets.emplace_back(program->mapper.lookup(values[i]));
  }
  auto returnOp = builder.create<mlir::ReturnOp>(loc, rets);
  // Attach the function to the module
  module.push_back(funcOp);
  IVLOG(5, mlir::debugString(module));
  if (failed(mlir::verify(module))) {
    throw std::runtime_error("Module verification error");
  }
  // Do some optimization passes
  mlir::PassManager pm(&impl->context);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  auto result = pm.run(module);
  if (failed(result)) {
    IVLOG(1, mlir::debugString(module));
    throw std::runtime_error("Optimization passes failure");
  }
  for (unsigned i = 0; i < returnOp.getNumOperands(); i++) {
    new_outputs[i] = returnOp.getOperand(i);
  }
  IVLOG(2, "TileBuilder::MakeProgram>" << mlir::debugString(module));
  return program;
}

std::vector<Value*> TileBuilder::ComputeGradients(ArrayRef<Value*> wrt, Value* loss) {
  // TODO
  return wrt;
}

}  // namespace pmlc::dialect::tile
