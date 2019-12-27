// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/util.h"

#include "mlir/IR/Builders.h"

#include "base/util/logging.h"

using mlir::NamedAttribute;
using mlir::OpBuilder;
using pmlc::dialect::stripe::ParallelForOp;
using pmlc::dialect::stripe::TerminateOp;

namespace pmlc {
namespace dialect {
namespace stripe {

void createMainParallelFor(mlir::FuncOp funcOp) {
  auto& region = funcOp.getBody();
  OpBuilder builder(region);
  auto src = &region.front();
  auto it = src->begin();
  auto forOp = builder.create<ParallelForOp>(funcOp.getLoc(), builder.getI64ArrayAttr({}));
  auto attrs = llvm::SmallVector<NamedAttribute, 1>{
      {builder.getIdentifier("main"), builder.getUnitAttr()},
  };
  forOp.setAttr(dialect::stripe::Dialect::getStripeAttrsName(), builder.getDictionaryAttr(attrs));
  forOp.setAttr("name", builder.getStringAttr("main"));
  auto block = builder.createBlock(&forOp.inner());
  auto& dst = block->getOperations();
  dst.splice(dst.end(), src->getOperations(), it, src->end());

  builder.setInsertionPointToEnd(src);
  builder.create<TerminateOp>(funcOp.getLoc());
}

bool hasAttr(Operation* op, const std::string& attr) {
  std::set<std::string> op_attrs_set;
  DictionaryAttr dict_attr = op->getAttrOfType<DictionaryAttr>(Dialect::getStripeAttrsName());
  if (!dict_attr) {
    return false;
  }
  ArrayRef<NamedAttribute> op_attrs = dict_attr.getValue();
  for (const auto& [key, value] : op_attrs) {
    auto name = key.strref();
    op_attrs_set.insert(name);
  }
  return op_attrs_set.find(attr) != op_attrs_set.end();
}

bool hasAttrs(Operation* op, const std::set<std::string>& attrs) {
  std::set<std::string> op_attrs_set;
  DictionaryAttr dict_attr = op->getAttrOfType<DictionaryAttr>(Dialect::getStripeAttrsName());
  if (!dict_attr) {
    return false;
  }
  ArrayRef<NamedAttribute> op_attrs = dict_attr.getValue();
  for (const auto& [key, value] : op_attrs) {
    auto name = key.strref();
    op_attrs_set.insert(name);
  }
  for (const auto& attr : attrs) {
    if (op_attrs_set.find(attr) == op_attrs_set.end()) {
      return false;
    }
  }
  return true;
}

DictionaryAttr addAttrInDictionary(DictionaryAttr old_dict, OpBuilder builder, NamedAttribute elem) {
  llvm::SmallVector<NamedAttribute, 8> new_array;
  if (old_dict) {
    ArrayRef<NamedAttribute> old_array = old_dict.getValue();
    new_array.insert(new_array.begin(), old_array.begin(), old_array.end());
  }
  new_array.emplace_back(elem);
  return builder.getDictionaryAttr(new_array);
}

ArrayAttr addAttrInArray(ArrayAttr old_array, OpBuilder builder, Attribute elem) {
  llvm::SmallVector<Attribute, 8> new_array;
  if (old_array) {
    ArrayRef<Attribute> elements = old_array.getValue();
    new_array.insert(new_array.begin(), elements.begin(), elements.end());
  }
  new_array.emplace_back(elem);
  ArrayRef new_array_ref(new_array.begin(), new_array.end());
  return builder.getArrayAttr(new_array_ref);
}

DictionaryAttr replaceAttrInDictionary(DictionaryAttr old_dict, OpBuilder builder, unsigned n, NamedAttribute elem) {
  llvm::SmallVector<NamedAttribute, 8> new_array;
  if (old_dict) {
    ArrayRef<NamedAttribute> old_array = old_dict.getValue();
    new_array.insert(new_array.begin(), old_array.begin(), old_array.end());
  }
  while (n >= new_array.size()) {
    new_array.emplace_back(builder.getIdentifier(""), builder.getUnitAttr());
  }
  new_array[n] = elem;
  return builder.getDictionaryAttr(new_array);
}

ArrayAttr replaceAttrInArray(ArrayAttr old_array, OpBuilder builder, unsigned n, Attribute elem) {
  llvm::SmallVector<Attribute, 8> new_array;
  if (old_array) {
    ArrayRef<Attribute> elements = old_array.getValue();
    new_array.insert(new_array.begin(), elements.begin(), elements.end());
  }
  while (n >= new_array.size()) {
    new_array.emplace_back(builder.getUnitAttr());
  }
  new_array[n] = elem;
  ArrayRef new_array_ref(new_array.begin(), new_array.end());
  return builder.getArrayAttr(new_array_ref);
}

void setOpAttrUnit(Operation* op, OpBuilder builder, const std::string& attr_name) {
  if (!op) {
    throw std::runtime_error("setUnitAttr: op is null");
  }
  auto old_attrs_dict = op->getAttrOfType<DictionaryAttr>(Dialect::getStripeAttrsName());
  NamedAttribute new_elem = {builder.getIdentifier(attr_name), builder.getUnitAttr()};
  auto new_attrs_dict = addAttrInDictionary(old_attrs_dict, builder, new_elem);
  op->setAttr(Dialect::getStripeAttrsName(), new_attrs_dict);
}

void setIdxAttrUnit(ParallelForOp op, StringRef target_idx, const std::string& attr_name) {
  auto idx_names = op.getAttrOfType<ArrayAttr>(mlir::Identifier::get("idx_names", op.getContext()));
  if (!idx_names) {
    return;
  }
  auto old_idx_attrs = op.getAttrOfType<ArrayAttr>(mlir::Identifier::get("idx_attrs", op.getContext()));
  ArrayAttr new_idx_attrs;
  auto builder = op.getBodyBuilder();
  for (unsigned i = 0; i < op.ranges().size(); i++) {
    StringRef idx_name;
    if (i < idx_names.size()) {
      if (auto str_attr = idx_names.getValue()[i].dyn_cast<StringAttr>()) {
        idx_name = str_attr.getValue();
      }
    }
    if (idx_name == target_idx) {
      DictionaryAttr old_dict;
      if (old_idx_attrs && i < old_idx_attrs.size()) {
        old_dict = old_idx_attrs.getValue()[i].dyn_cast<DictionaryAttr>();
      }
      NamedAttribute new_elem = {builder.getIdentifier(attr_name), builder.getUnitAttr()};
      DictionaryAttr new_dict = addAttrInDictionary(old_dict, builder, new_elem);
      new_idx_attrs = replaceAttrInArray(old_idx_attrs, builder, i, new_dict);
      break;
    }
  }
  op.setAttr(mlir::Identifier::get("idx_attrs", op.getContext()), new_idx_attrs);
}

int64_t idxRange(BlockArgument* idx) {
  auto pf = mlir::cast<ParallelForOp>(idx->getOwner()->getParentOp());
  return pf.getRange(idx->getArgNumber());
}

StringRef idxName(BlockArgument* idx) {
  auto pf = mlir::cast<ParallelForOp>(idx->getOwner()->getParentOp());
  auto names = pf.getAttrOfType<ArrayAttr>(mlir::Identifier::get("idx_names", pf.getContext()));
  auto n = idx->getArgNumber();
  auto idx_name = StringAttr::get("", pf.getContext());
  if (names && n < names.size()) {
    if (auto str_attr = names.getValue()[n].template dyn_cast<StringAttr>()) {
      idx_name = str_attr;
    }
  }
  return idx_name.getValue();
}

std::pair<StringRef, unsigned> getSingleIndex(ParallelForOp op, unsigned n) {
  auto names = op.getAttrOfType<ArrayAttr>(mlir::Identifier::get("idx_names", op.getContext()));
  auto ranges = op.ranges();
  auto idx_name = StringAttr::get("", op.getContext());
  if (names && n < names.size()) {
    if (auto str_attr = names.getValue()[n].template dyn_cast<StringAttr>()) {
      idx_name = str_attr;
    }
  }
  unsigned range = ranges.getValue()[n].cast<IntegerAttr>().getInt();
  return std::make_pair(idx_name.getValue(), range);
}

llvm::SmallVector<std::pair<StringRef, unsigned>, kIndexLimit> getAllIndex(ParallelForOp op) {
  llvm::SmallVector<std::pair<StringRef, unsigned>, kIndexLimit> idxs;
  auto names = op.getAttrOfType<ArrayAttr>(mlir::Identifier::get("idx_names", op.getContext()));
  auto ranges = op.ranges();
  for (unsigned i = 0; i < op.ranges().size(); i++) {
    auto idx_name = StringAttr::get("", op.getContext());
    if (names && i < names.size()) {
      if (auto str_attr = names.getValue()[i].template dyn_cast<StringAttr>()) {
        idx_name = str_attr;
      }
    }
    unsigned range = ranges.getValue()[i].cast<IntegerAttr>().getInt();
    idxs.push_back(std::make_pair(idx_name.getValue(), range));
  }
  return idxs;
}

TensorType baseType(Value* value) {
  Value *cur_value = value;
  do {
    if (auto def = cur_value->getDefiningOp()) {
      if (auto aop = mlir::dyn_cast<AllocateOp>(def)) {
        return aop.layout();
      }
      auto rop = mlir::dyn_cast<RefineOp>(def);
      cur_value = rop.in();
    }
    else if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(cur_value)) {
      auto parentOp = arg->getOwner()->getParentOp();
      auto funcOp = mlir::dyn_cast<mlir::FuncOp>(parentOp);
      if (!funcOp) {
        throw std::runtime_error("Invalid tensor value: block argument not contained by FuncOp");
      }
      auto attrName = stripe::Dialect::getDialectAttrName("layout");
      auto attr = funcOp.getArgAttrOfType<mlir::TypeAttr>(arg->getArgNumber(), attrName);
      assert(attr && "Expected 'layout' attribute in TensorRefType function argument");
      return attr.getValue().cast<TensorType>();
    }
    else {
      throw std::runtime_error("Invalid tensor value");
    }
  } while (cur_value);
  throw std::runtime_error("Base type not found for the operation.");
}

llvm::SmallVector<mlir::BlockArgument*, kIndexLimit> strideOneIdxs(Value* value) {
  llvm::SmallVector<mlir::BlockArgument*, kIndexLimit> idxs;
  auto ref_op = mlir::dyn_cast<RefineOp>(value->getDefiningOp());
  TensorType base_type = baseType(value);
  auto shape = base_type.getShape();
  for (unsigned i = 0; i < shape.size(); i++) {
    if (shape[i].stride != 1) {
      continue;
    }
    auto access = AffinePolynomial(ref_op.getOffset(i));
    for (auto [arg, scale] : access.terms) {
      if (scale == 1) {
        idxs.push_back(arg);
      }
    }
  }
  return idxs;
}

StringRef tensorName(Value* tensor) {
  if (auto op = tensor->getDefiningOp()) {
    auto nameAttr = op->getAttrOfType<StringAttr>("name");
    if (nameAttr) {
      return nameAttr.getValue();
    }
  }
  return StringRef();
}

DataType tensorElementType(Value* tensor) {
  auto tensor_type = tensor->getType().cast<TensorRefType>();
  auto elt_type = tensor_type.getElementType().cast<eltwise::ScalarType>();
  return elt_type.type();
}

eltwise::ScalarConstantOp initialValue(OpBuilder* builder, DataType type,
                                       const std::string& agg_name,
                                       const std::string& var_name) {
  if (agg_name == "assign") {
    return eltwise::ScalarConstantOp();
  }
  eltwise::ScalarConstantOp op;
  auto unknownLoc = builder->getUnknownLoc();

  #define BUILD_CONST_OP(ivalue, fvalue)                                    \
    if (IsIntegerDataType(type)) {                                          \
      op = builder->create<eltwise::ScalarConstantOp>(                      \
        unknownLoc, ScalarType::get(builder->getContext(), type), ivalue);  \
    }                                                                       \
    else if (IsFloatDataType(type)) {                                       \
      op = builder->create<eltwise::ScalarConstantOp>(                      \
        unknownLoc, ScalarType::get(builder->getContext(), type), fvalue);  \
    }

  if (agg_name == "add") {
    BUILD_CONST_OP((int64_t)0, (double)0.0);
  }
  else if (agg_name == "mul") {
    BUILD_CONST_OP((int64_t)1, (double)1.0);
  }
  else if (agg_name == "max") {
    BUILD_CONST_OP(IntegerMin(type), FloatMin(type));
  }
  else if (agg_name == "min") {
    BUILD_CONST_OP(IntegerMax(type), FloatMax(type));
  }
  else {
    throw std::runtime_error("Unsupported aggregate op.");
  }
  op.setAttr("scalar_name", builder->getStringAttr(var_name == "" ? "cst" : var_name));
  return op;
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
