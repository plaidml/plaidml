#include "pmlc/util/tags.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace pmlc {

using namespace mlir; // NOLINT

static constexpr StringLiteral kTagsAttrName = "tags";

bool hasTags(Operation *op) {
  return op->hasAttrOfType<DictionaryAttr>(kTagsAttrName);
}

void copyTags(Operation *dst, Operation *src) {
  if (auto dict = src->getAttrOfType<DictionaryAttr>(kTagsAttrName))
    dst->setAttr(kTagsAttrName, dict);
}

void clearTags(Operation *op) { op->removeAttr(kTagsAttrName); }

void clearTag(Operation *op, StringRef name) {
  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(kTagsAttrName);
  dict.erase(name);
  if (dict.empty())
    op->removeAttr(kTagsAttrName);
  else
    op->setAttr(kTagsAttrName, dict.getDictionary(op->getContext()));
}

void setUnitTag(Operation *op, StringRef name) {
  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(kTagsAttrName);
  dict.set(name, UnitAttr::get(op->getContext()));
  op->setAttr(kTagsAttrName, dict.getDictionary(op->getContext()));
}

void setIntegerTag(Operation *op, StringRef name, int64_t val) {
  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(kTagsAttrName);
  auto type = IntegerType::get(op->getContext(), 64);
  dict.set(name, IntegerAttr::get(type, val));
  op->setAttr(kTagsAttrName, dict.getDictionary(op->getContext()));
}

bool hasUnitTag(Operation *op, StringRef name) {
  auto dict = op->getAttrOfType<DictionaryAttr>(kTagsAttrName);
  if (!dict)
    return false;
  return !!dict.get(name).dyn_cast_or_null<UnitAttr>();
}

bool hasIntegerTag(Operation *op, StringRef name) {
  auto dict = op->getAttrOfType<DictionaryAttr>(kTagsAttrName);
  if (!dict)
    return false;
  return !!dict.get(name).dyn_cast_or_null<IntegerAttr>();
}

int64_t getIntegerTag(Operation *op, StringRef name, int64_t defaultVal) {
  if (auto dict = op->getAttrOfType<DictionaryAttr>(kTagsAttrName))
    if (auto attr = dict.get(name).dyn_cast_or_null<IntegerAttr>())
      return attr.getInt();
  return defaultVal;
}

// Check if all tags exist
bool hasAllTags(Operation *op, ArrayRef<StringRef> tags) {
  if (tags.empty())
    return true;

  auto opTagsAttr = op->getAttrOfType<DictionaryAttr>(kTagsAttrName);
  if (!opTagsAttr)
    return false;

  return llvm::all_of(tags,
                      [&](StringRef tag) { return !!opTagsAttr.get(tag); });
}

bool hasTag(Operation *op, StringRef tag) {
  auto opTagsAttr = op->getAttrOfType<DictionaryAttr>(kTagsAttrName);
  if (!opTagsAttr)
    return false;
  return !!opTagsAttr.get(tag);
}

// Set tags in op
void setTags(Operation *op, ArrayRef<StringRef> tags) {
  for (StringRef tag : tags)
    setUnitTag(op, tag);
}

void setIntegerArrayTag(Operation *op, StringRef name,
                        ArrayRef<int64_t> values) {
  Builder builder(op->getContext());
  NamedAttrList dict = op->getAttrOfType<DictionaryAttr>(kTagsAttrName);
  dict.set(name, builder.getI64ArrayAttr(values));
  op->setAttr(kTagsAttrName, dict.getDictionary(op->getContext()));
}

bool getIntegerArrayTag(Operation *op, StringRef name,
                        SmallVectorImpl<int64_t> &out) {
  if (auto dict = op->getAttrOfType<DictionaryAttr>(kTagsAttrName)) {
    if (auto attr = dict.get(name).dyn_cast_or_null<ArrayAttr>()) {
      out.clear();
      for (auto value : attr.getAsValueRange<IntegerAttr>()) {
        out.push_back(value.getZExtValue());
      }
      return true;
    }
  }
  return false;
}

} // namespace pmlc
