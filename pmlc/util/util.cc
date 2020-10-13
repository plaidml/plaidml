// Copyright 2019, Intel Corporation

#include "pmlc/util/util.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "llvm/Support/Process.h"

using namespace mlir; // NOLINT

namespace pmlc::util {

uint64_t getByteSize(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t, 8> strides;
  if (failed(mlir::getStridesAndOffset(type, strides, offset))) {
    throw std::runtime_error("Could not retrieve strides");
  }
  auto sizes = type.getShape();
  uint64_t total = 0;
  for (unsigned i = 0; i < type.getRank(); i++) {
    if (!sizes[i]) {
      return 0;
    }
    if (strides[i] > 0) {
      total += (sizes[i] - 1) * strides[i];
    }
  }
  unsigned elem_bytes = llvm::divideCeil(type.getElementTypeBitWidth(), 8);
  return (total + 1) * elem_bytes;
}

// Check if all tags exist
bool hasAllTags(Operation *op, ArrayRef<StringRef> tags) {
  if (tags.empty()) {
    return true;
  }
  DictionaryAttr opTagsAttr = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  if (!opTagsAttr) {
    return false;
  }
  for (StringRef tag : tags) {
    if (!opTagsAttr.get(tag)) {
      return false;
    }
  }
  return true;
}

bool hasTag(Operation *op, StringRef tag) {
  DictionaryAttr opTagsAttr = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  if (!opTagsAttr) {
    return false;
  }
  return opTagsAttr.get(tag) != nullptr;
}

// Set tags in op
void setTags(Operation *op, ArrayRef<StringRef> tags) {
  if (tags.empty()) {
    return;
  }
  OpBuilder builder(op);
  DictionaryAttr opTagsAttr = op->getAttrOfType<DictionaryAttr>(kTagAttribute);
  SmallVector<NamedAttribute, 4> newTags;
  if (opTagsAttr) {
    newTags.append(opTagsAttr.begin(), opTagsAttr.end());
  }
  for (StringRef tag : tags) {
    if (!opTagsAttr || !opTagsAttr.get(tag)) {
      newTags.emplace_back(builder.getNamedAttr(tag, builder.getUnitAttr()));
    }
  }
  op->setAttr(kTagAttribute, builder.getDictionaryAttr(newTags));
}

DiagnosticCounter::DiagnosticCounter() : counter(0), threshold(0) {
  auto env = llvm::sys::Process::GetEnv("PLAIDML_COUNTER");
  if (env) {
    threshold = std::atoi(env->c_str());
  }
}

DiagnosticCounter::Result DiagnosticCounter::next() {
  if (!threshold) {
    return Result::Continue;
  }
  ++counter;
  if (counter < threshold) {
    return Result::Continue;
  }
  if (counter > threshold) {
    return Result::Break;
  }
  return Result::Match;
}

} // namespace pmlc::util
