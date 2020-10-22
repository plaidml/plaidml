
#include "pmlc/util/tags.h"

#include "mlir/IR/StandardTypes.h"

namespace pmlc {

using namespace llvm; // NOLINT
using namespace mlir; // NOLINT

static StringRef tagDict() { return "tags"; }

bool hasTags(mlir::Operation *op) {
  return static_cast<bool>(op->getAttrOfType<DictionaryAttr>(tagDict()));
}
void copyTags(mlir::Operation *dst, mlir::Operation *src) {
  auto dict = src->getAttrOfType<DictionaryAttr>(tagDict());
  if (dict) {
    dst->setAttr(tagDict(), dict);
  }
}

void clearTags(mlir::Operation *op) { op->removeAttr(tagDict()); }

void clearTag(mlir::Operation *op, llvm::StringRef name) {
  MutableDictionaryAttr dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  auto ident = Identifier::get(name, op->getContext());
  dict.remove(ident);
  if (dict.empty()) {
    op->removeAttr(tagDict());
  } else {
    op->setAttr(tagDict(), dict.getDictionary(op->getContext()));
  }
}
void setUnitTag(mlir::Operation *op, llvm::StringRef name) {
  auto ident = Identifier::get(name, op->getContext());
  MutableDictionaryAttr dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  dict.set(ident, UnitAttr::get(op->getContext()));
  op->setAttr(tagDict(), dict.getDictionary(op->getContext()));
}

void setIntegerTag(mlir::Operation *op, llvm::StringRef name, int64_t val) {
  auto ident = Identifier::get(name, op->getContext());
  MutableDictionaryAttr dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  auto type = IntegerType::get(64, op->getContext());
  dict.set(ident, IntegerAttr::get(type, val));
  op->setAttr(tagDict(), dict.getDictionary(op->getContext()));
}

bool hasUnitTag(mlir::Operation *op, llvm::StringRef name) {
  auto dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  if (!dict) {
    return false;
  }
  auto attr = dict.get(name).dyn_cast_or_null<UnitAttr>();
  return static_cast<bool>(attr);
}

bool hasIntegerTag(mlir::Operation *op, llvm::StringRef name) {
  auto dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  if (!dict) {
    return false;
  }
  auto attr = dict.get(name).dyn_cast_or_null<IntegerAttr>();
  return static_cast<bool>(attr);
}

int64_t getIntegerTag(mlir::Operation *op, llvm::StringRef name,
                      int64_t defaultVal) {
  auto dict = op->getAttrOfType<DictionaryAttr>(tagDict());
  if (!dict) {
    return defaultVal;
  }
  auto attr = dict.get(name).dyn_cast_or_null<IntegerAttr>();
  if (attr) {
    return attr.getInt();
  }
  return defaultVal;
}

} // namespace pmlc
