// Copyright 2019, Intel Corporation
#include "pmlc/dialect/stripe/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc {
namespace dialect {
namespace stripe {

#include "pmlc/dialect/stripe/ops_interfaces.cpp.inc"

void PrintSimple(Operation* op, OpAsmPrinter* p, ArrayRef<StringRef> fixed, Type otype = Type()) {
  *p << op->getName() << " ";
  bool first = true;
  for (size_t i = 0; i < op->getNumOperands(); i++) {
    if (!first) {
      *p << ", ";
    }
    first = false;
    p->printOperand(op->getOperand(i));
  }
  for (StringRef name : fixed) {
    if (!first) {
      *p << ", ";
    }
    first = false;
    *p << op->getAttrOfType<IntegerAttr>(name).getValue();
  }
  if (otype) {
    *p << " : " << otype;
  }
  llvm::SmallVector<StringRef, 4> ignore(fixed.begin(), fixed.end());
  // ignore.push_back("name");
  // ignore.push_back("scalar_name");
  p->printOptionalAttrDict(op->getAttrs(), ignore);
}

template <typename T, size_t N>
bool ParseSimple(OpAsmParser* p, OperationState* res, std::array<OpAsmParser::OperandType, N>* ops,
                 ArrayRef<StringRef> fixed, T* out_type) {
  bool first = true;
  bool r = false;
  for (size_t i = 0; i < N; i++) {
    if (!first) {
      r = r || p->parseComma();
    }
    first = false;
    r = r || p->parseOperand((*ops)[i]);
  }
  for (StringRef name : fixed) {
    if (!first) {
      r = r || p->parseComma();
      Attribute dont_care;
      r = r || p->parseAttribute(dont_care, name, res->attributes);
    }
    first = false;
  }
  if (out_type) {
    r = r || p->parseColon();
    r = r || p->parseType(*out_type);
  }
  r = r || p->parseOptionalAttributeDict(res->attributes);
  return r;
}

#define GET_OP_CLASSES
#include "pmlc/dialect/stripe/ops.cpp.inc"

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
