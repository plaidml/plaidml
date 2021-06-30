#include "pmlc/util/tags.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace llvm; // NOLINT
using namespace mlir; // NOLINT

namespace pmlc {

static constexpr StringLiteral kTagAttribute = "tags";

bool hasTags(Operation *op) {
  return static_cast<bool>(op->getAttrOfType<DictionaryAttr>(kTagAttribute));
}

void copyTags(Operation *dst, Operation *src) {
  NamedAttrList srcDict = src->getAttrOfType<DictionaryAttr>(kTagAttribute);
  NamedAttrList dstDict = dst->getAttrOfType<DictionaryAttr>(kTagAttribute);
  dstDict.append(srcDict.begin(), srcDict.end());
  dst->setAttr(kTagAttribute, dstDict.getDictionary(dst->getContext()));
}

void clearTags(Operation *op) { op->removeAttr(kTagAttribute); }

void clearTag(Operation *op, StringRef name) {
  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  dict.erase(name);
  if (dict.empty())
    op->removeAttr(kTagAttribute);
  else
    op->setAttr(kTagAttribute, dict.getDictionary(op->getContext()));
}

void setUnitTag(Operation *op, StringRef name) {
  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  dict.set(name, UnitAttr::get(op->getContext()));
  op->setAttr(kTagAttribute, dict.getDictionary(op->getContext()));
}

void setIntegerTag(Operation *op, StringRef name, int64_t val) {
  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  auto type = IntegerType::get(op->getContext(), 64);
  dict.set(name, IntegerAttr::get(type, val));
  op->setAttr(kTagAttribute, dict.getDictionary(op->getContext()));
}

void setTags(Operation *op, ArrayRef<StringRef> tags) {
  if (tags.empty())
    return;

  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  for (StringRef tag : tags) {
    dict.set(tag, UnitAttr::get(op->getContext()));
  }
  op->setAttr(kTagAttribute, dict.getDictionary(op->getContext()));
}

bool hasUnitTag(Operation *op, StringRef name) {
  auto dict = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  if (!dict)
    return false;

  auto attr = dict.get(name).dyn_cast_or_null<UnitAttr>();
  return attr != nullptr;
}

bool hasIntegerTag(Operation *op, StringRef name) {
  auto dict = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  if (!dict)
    return false;

  auto attr = dict.get(name).dyn_cast_or_null<IntegerAttr>();
  return attr != nullptr;
}

int64_t getIntegerTag(Operation *op, StringRef name, int64_t defaultVal) {
  auto dict = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  if (!dict)
    return defaultVal;

  auto attr = dict.get(name).dyn_cast_or_null<IntegerAttr>();
  if (attr)
    return attr.getInt();

  return defaultVal;
}

bool hasAllTags(Operation *op, ArrayRef<StringRef> tags) {
  if (tags.empty())
    return true;

  DictionaryAttr dict = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  if (!dict)
    return false;

  for (StringRef tag : tags) {
    if (!dict.get(tag))
      return false;
  }

  return true;
}

bool hasTag(Operation *op, StringRef tag) {
  DictionaryAttr dict = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  if (!dict)
    return false;

  return dict.get(tag) != nullptr;
}

} // namespace pmlc
