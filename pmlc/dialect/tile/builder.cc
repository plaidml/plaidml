// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/builder.h"

#include <map>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
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

struct TileBuilder::Impl {
  MLIRContext context;
  ModuleOp module;
  OpBuilder builder;
  std::unordered_map<Value*, Value*> implicitUpdates;
  std::unordered_map<Value*, BufferPtr> implicitBindings;
  std::unordered_map<Value*, RankedTensorType> shapeCache;
  NoneOp noneOp;
  Value* defaultInit;
  Location loc;
  unsigned idxCounter = 0;

  Impl()
      : module(ModuleOp::create(UnknownLoc::get(&context))),  //
        builder(module.getBody()),
        loc(builder.getUnknownLoc()) {
    builder.setInsertionPointToStart(module.getBody());
    defaultInit = makeScalarConstantOp(0.0);
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

  std::vector<Value*> getBackwardSliceOfAffine(const llvm::SetVector<Value*>& values) {
    return util::getBackwardSlice(values, false, [](Value* value) {
      if (auto scalarType = value->getType().dyn_cast<ScalarType>()) {
        return scalarType.type() == DataType::INT32;
      }
      return false;
    });
  }

  Value* makeIndexOp(StringRef fn, ArrayRef<Value*> args) {
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
    auto op = builder.create<IndexOp>(loc, resultType, tensor, dimAttr);
    return op.result();
  }

  Value* makePrngOp(StringRef fn, ArrayRef<Value*> args) {
    if (args.size() < 1) {
      throw std::runtime_error("prng op expects at least one operand");
    }
    auto state = args.front();
    auto dims = args.drop_front();
    auto resultType = PrngOp::getResultType(args);
    auto elementType = builder.getType<ScalarType>(DataType::UINT32);
    auto stateType = RankedTensorType::get({3, 2048}, elementType);
    auto op = builder.create<PrngOp>(loc, resultType, stateType, state, dims);
    implicitUpdates.emplace(op.new_state(), op.state());
    return op.result();
  }

  Value* makeScalarConstantOp(double value) {
    auto type = builder.getType<ScalarType>(DataType::FLOAT32);
    return builder.create<ScalarConstantOp>(loc, type, value).result();
  }
};

TileBuilder::TileBuilder() : impl(new Impl) {}

TileBuilder::~TileBuilder() = default;

void TileBuilder::Destroy(Value* value) {
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

stripe::TensorType TileBuilder::IntoTensorType(RankedTensorType type) {
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

void TileBuilder::BindShape(Value* tensor, RankedTensorType type) {
  IVLOG(5, "TileBuilder::BindShape>");
  tensor->setType(type);
}

void TileBuilder::BindBuffer(Value* tensor, BufferPtr buffer) {
  IVLOG(5, "TileBuilder::BindBuffer>");
  impl->implicitBindings[tensor] = buffer;
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
  auto it = impl->shapeCache.find(tensor);
  if (it != impl->shapeCache.end()) {
    return it->second;
  }
  ProgramMutations mutations;
  mutations.outputs.emplace_back(tensor);
  auto program = MakeProgram("compute_shape", mutations);
  auto shape = program->outputs[0]->getType().dyn_cast<RankedTensorType>();
  impl->shapeCache.emplace(tensor, shape);
  return shape;
}

Value* TileBuilder::MakeCastOp(Value* tensor, DataType dtype) {
  IVLOG(5, "TileBuilder::MakeCastOp> " << to_string(dtype));
  IVLOG(6, "  arg: " << mlir::debugString(*tensor));
  auto elementType = impl->builder.getType<ScalarType>(dtype);
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto resultType = RankedTensorType::get(tensorType.getShape(), elementType);
  return impl->builder.create<eltwise::CastOp>(impl->loc, resultType, tensor).result();
}

Value* TileBuilder::MakePrimitiveOp(StringRef fn, ArrayRef<Value*> args) {
  using PrimitiveBuilder = Value* (Impl::*)(StringRef fn, ArrayRef<Value*> args);
  static std::map<StringRef, PrimitiveBuilder> primitives = {
      {"index", &Impl::makeIndexOp},
      {"prng", &Impl::makePrngOp},
  };
  IVLOG(5, "TileBuilder::MakePrimitiveOp> " << fn.str());
  for (auto arg : args) {
    IVLOG(6, "  arg: " << mlir::debugString(*arg));
  }
  auto it = primitives.find(fn);
  if (it != primitives.end()) {
    return (impl.get()->*it->second)(fn, args);
  }
  auto abstractOp = impl->lookupOperation(fn);
  auto genericBuilder = abstractOp->getInterface<util::GenericBuilder>();
  if (!genericBuilder) {
    throw std::runtime_error("Unknown intrinsic: " + fn.str());
  }
  auto type = impl->builder.getType<ScalarType>(DataType::FLOAT32);  // TODO
  auto op = genericBuilder->create(&impl->builder, impl->loc, type, args);
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
    impl->noneOp = impl->builder.create<NoneOp>(impl->loc, type);
  }
  return impl->noneOp.result();
}

Value* TileBuilder::MakeStringOp(StringRef value) {
  IVLOG(5, "TileBuilder::MakeStringOp> " << value.str());
  auto type = StringType::get(&impl->context);
  auto attr = impl->builder.getStringAttr(value);
  return impl->builder.create<StringOp>(impl->loc, type, attr).result();
}

llvm::StringRef TileBuilder::GetStringValue(Value* value) {
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
  return impl->builder.create<TupleOp>(impl->loc, tupleType, elts).result();
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
  auto type = impl->builder.getType<ScalarType>(DataType::INT32);
  return impl->builder.create<ScalarConstantOp>(impl->loc, type, value).result();
}

int64_t TileBuilder::GetIntegerValue(Value* value) {
  if (auto op = llvm::dyn_cast_or_null<ScalarConstantOp>(value->getDefiningOp())) {
    return op.getIntAttr().getInt();
  }
  throw std::runtime_error("Expected ScalarConstantOp");
}

Value* TileBuilder::MakeScalarConstantOp(double value) {
  IVLOG(5, "TileBuilder::MakeScalarConstantOp> " << value);
  return impl->makeScalarConstantOp(value);
}

double TileBuilder::GetFloatValue(Value* value) {
  if (auto op = llvm::dyn_cast_or_null<ScalarConstantOp>(value->getDefiningOp())) {
    return op.getFloatAttr().getValueAsDouble();
  }
  throw std::runtime_error("Expected ScalarConstantOp");
}

Value* TileBuilder::MakeDimOp(Value* tensor, unsigned dim) {
  IVLOG(5, "TileBuilder::MakeDimOp> tensor: " << mlir::debugString(*tensor) << ", dim: " << dim);
  return impl->builder.create<DimOp>(impl->loc, tensor, dim).result();
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
  auto op = impl->builder.create<PlaceholderOp>(impl->loc, type);
  if (!name.empty()) {
    op.setAttr("name", impl->builder.getStringAttr(name));
  }
  if (buffer) {
    impl->implicitBindings[op.result()] = buffer;
  }
  return op.result();
}

Value* TileBuilder::MakeAffineConstantOp(int64_t value) {
  IVLOG(5, "TileBuilder::MakeAffineConstantOp> " << value);
  return impl->builder.create<AffineConstantOp>(impl->loc, value).result();
}

Value* TileBuilder::MakeAffineIndexOp(StringRef name) {
  IVLOG(5, "TileBuilder::MakeAffineIndexOp> " << name.str());
  return impl->builder.create<AffineIndexOp>(impl->loc, impl->idxCounter++, name).result();
}

Value* TileBuilder::MakeAffineAddOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineAddOp>");
  return impl->builder.create<AffineAddOp>(impl->loc, args).result();
}

Value* TileBuilder::MakeAffineSubOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineSubOp>");
  return impl->builder.create<AffineSubOp>(impl->loc, args).result();
}

Value* TileBuilder::MakeAffineMulOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineMulOp>");
  return impl->builder.create<AffineMulOp>(impl->loc, args).result();
}

Value* TileBuilder::MakeAffineDivOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineDivOp>");
  return impl->builder.create<AffineDivOp>(impl->loc, args).result();
}

Value* TileBuilder::MakeAffineNegOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineNegOp>");
  return impl->builder.create<AffineNegOp>(impl->loc, args).result();
}

Value* TileBuilder::MakeAffineMaxOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineMaxOp>");
  return impl->builder.create<AffineMaxOp>(impl->loc, args).result();
}

Value* TileBuilder::MakeAffineMinOp(ArrayRef<Value*> args) {
  IVLOG(5, "TileBuilder::MakeAffineMinOp>");
  return impl->builder.create<AffineMinOp>(impl->loc, args).result();
}

Value* TileBuilder::MakeAffineSourceIndexMapOp(Value* tensor, ArrayRef<Value*> idxs) {
  IVLOG(5, "TileBuilder::MakeAffineSourceIndexMapOp>");
  return impl->builder.create<AffineTensorMapOp>(impl->loc, tensor, idxs).result();
}

Value* TileBuilder::MakeAffineSinkIndexMapOp(ArrayRef<Value*> idxs) {
  IVLOG(5, "TileBuilder::MakeAffineSinkIndexMapOp>");
  return impl->builder.create<AffineMapOp>(impl->loc, idxs).result();
}

Value* TileBuilder::MakeAffineSizeMapOp(ArrayRef<Value*> sizes) {
  IVLOG(5, "TileBuilder::MakeAffineSizeMapOp>");
  return impl->builder.create<AffineMapOp>(impl->loc, sizes).result();
}

void TileBuilder::AddConstraint(Value* cion, Value* lhs, Value* rhs) {
  IVLOG(5, "TileBuilder::AddConstraint>");
  auto op = cion->getDefiningOp();
  auto cionOp = llvm::dyn_cast_or_null<SymbolicContractionOp>(op);
  if (!cionOp) {
    throw std::runtime_error("add_constraint can only be specified on a contraction.");
  }

  auto consOp = llvm::cast<AffineConstraintsOp>(cionOp.cons()->getDefiningOp());
  SmallVector<Value*, 6> pairs{consOp.pairs()};
  pairs.emplace_back(lhs);
  pairs.emplace_back(rhs);
  consOp.getOperation()->setOperands(pairs);
}

void TileBuilder::SetUseDefault(Value* cion, Value* init) {
  IVLOG(2, "TileBuilder::SetUseDefault>");
  auto op = cion->getDefiningOp();
  auto cionOp = llvm::dyn_cast_or_null<SymbolicContractionOp>(op);
  if (!cionOp) {
    throw std::runtime_error("no_reduce can only be specified on a contraction.");
  }
  cionOp.setOperand(0, init);
}

void TileBuilder::SetNoReduce(Value* cion, bool no_reduce) {
  IVLOG(2, "TileBuilder::SetNoReduce> " << no_reduce);
  auto op = cion->getDefiningOp();
  auto cionOp = llvm::dyn_cast_or_null<SymbolicContractionOp>(op);
  if (!cionOp) {
    throw std::runtime_error("no_reduce can only be specified on a contraction.");
  }
  if (no_reduce) {
    cionOp.setAttr("no_reduce", impl->builder.getUnitAttr());
  } else {
    cionOp.removeAttr("no_reduce");
  }
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
  // TODO: handle names (and idx_names)
  // Compute the sink shape of the contraction
  SmallVector<Type, 3> types;
  for (auto src : srcs) {
    auto mapOp = llvm::cast<AffineTensorMapOp>(src->getDefiningOp());
    types.push_back(mapOp.tensor()->getType());
  }
  Type elementType;
  if (combo == CombinationKind::eq) {
    elementType = ScalarType::get(&impl->context, DataType::BOOLEAN);
  } else if (combo == CombinationKind::cond) {
    auto rankedTensorType = eltwise::getRankedTensorType(types[2]);
    elementType = rankedTensorType.getElementType();
  } else {
    elementType = impl->inferElementType(types);
  }
  auto sizeMapOp = llvm::cast<AffineMapOp>(sizes->getDefiningOp());
  SmallVector<Value*, 4> sizeDims(sizeMapOp.dims());
  auto shape = eltwise::ComputeShape(sizeDims);

  StringAttr nameAttr;
  if (name.size()) {
    nameAttr = impl->builder.getStringAttr(name);
  }
  auto op = impl->builder.create<SymbolicContractionOp>(             //
      impl->loc,                                                     //
      RankedTensorType::get(shape, elementType),                     //
      impl->defaultInit,                                             //
      impl->builder.create<AffineConstraintsOp>(impl->loc),          //
      sizes,                                                         //
      sink,                                                          //
      srcs,                                                          //
      impl->builder.getI64IntegerAttr(static_cast<int64_t>(agg)),    //
      impl->builder.getI64IntegerAttr(static_cast<int64_t>(combo)),  //
      UnitAttr{},                                                    //
      nameAttr);
  return op.result();
}

std::shared_ptr<TileProgram> TileBuilder::MakeProgram(StringRef name, const ProgramMutations& mutations) {
  if (name.empty()) {
    name = "noname";
  }
  IVLOG(1, "TileBuilder::MakeProgram> " << name.str());
  IVLOG(6, mlir::debugString(impl->module));
  // Wrap duplicate outputs and outputs that directly refer to inputs
  llvm::SetVector<Value*> outputs;
  for (auto output : mutations.outputs) {
    if (!output) {
      throw std::runtime_error("Invalid output");
    }
    if (outputs.count(output) || llvm::isa<PlaceholderOp>(output->getDefiningOp())) {
      outputs.insert(MakePrimitiveOp("ident", {output}));
    } else {
      outputs.insert(output);
    }
  }
  for (const auto& update : mutations.updates) {
    outputs.insert(update.source);
  }
  auto slice = util::getBackwardSlice(outputs, true);
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
  auto initialFuncType = mlir::FunctionType::get(inputTypes, {}, &impl->context);
  auto funcOp = mlir::FuncOp::create(loc, name, initialFuncType, {});
  funcOp.addEntryBlock();
  OpBuilder builder(funcOp.getBody());
  std::set<std::string> names;
  auto attrName = Dialect::getDialectAttrName("name");
  unsigned argcnt = 0;
  std::map<Operation*, Operation*> opMap;
  BlockAndValueMapping mapper;
  for (auto value : slice) {
    auto op = value->getDefiningOp();
    // Only copy over top-level ops (those owned by the workspace module)
    if (op && op->getBlock() == impl->module.getBody()) {
      if (auto placeholderOp = llvm::dyn_cast<PlaceholderOp>(op)) {
        // Replace placeholders with block arguments
        auto blockArg = funcOp.getArgument(argcnt++);
        if (auto attr = placeholderOp.getAttrOfType<StringAttr>("name")) {
          auto uniqueName = util::getUniqueName(&names, attr.getValue());
          auto uniqueAttr = builder.getStringAttr(uniqueName);
          funcOp.setArgAttr(blockArg->getArgNumber(), attrName, uniqueAttr);
        }
        IVLOG(5, "BlockArgument mapping: " << value << " -> " << blockArg);
        mapper.map(value, blockArg);
        ProgramArgument programArg{true, value, value->getType().cast<RankedTensorType>()};
        auto itBinding = impl->implicitBindings.find(value);
        if (itBinding != impl->implicitBindings.end()) {
          programArg.buffer = itBinding->second;
        }
        program->arguments.emplace_back(programArg);
      } else {
        Operation* newOp;
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
            IVLOG(6, "value: " << mlir::debugString(*value));
            IVLOG(6, "newResult: " << mlir::debugString(*newResult));
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
  std::vector<Value*> returnOperands;
  std::vector<Type> resultTypes;
  for (auto output : outputs) {
    auto value = mapper.lookupOrNull(output);
    if (!value) {
      throw std::runtime_error("Output not found in mapper");
    }
    resultTypes.emplace_back(value->getType());
    returnOperands.emplace_back(value);
  }
  auto returnOp = builder.create<mlir::ReturnOp>(loc, returnOperands);
  // compute final function type
  auto finalFuncType = mlir::FunctionType::get(inputTypes, resultTypes, &impl->context);
  funcOp.setType(finalFuncType);
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
    auto userValue = outputs[i];
    auto finalValue = returnOp.getOperand(i);
    auto itUpdate = impl->implicitUpdates.find(outputs[i]);
    if (itUpdate != impl->implicitUpdates.end()) {
      userValue = itUpdate->second;
    }
    ProgramArgument programArg{false, userValue, finalValue->getType().cast<RankedTensorType>()};
    auto itBinding = impl->implicitBindings.find(finalValue);
    if (itBinding != impl->implicitBindings.end()) {
      programArg.buffer = itBinding->second;
    }
    program->arguments.emplace_back(programArg);
    program->outputs.emplace_back(finalValue);
  }
  IVLOG(2, "TileBuilder::MakeProgram>" << mlir::debugString(module));
  return program;
}

std::vector<Value*> TileBuilder::ComputeGradients(ArrayRef<Value*> wrt, Value* loss) {
  // TODO
  return wrt;
}

}  // namespace pmlc::dialect::tile
