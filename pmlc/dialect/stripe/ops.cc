// Copyright 2019, Intel Corporation
#include "pmlc/dialect/stripe/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc {
namespace dialect {
namespace stripe {

#include "pmlc/dialect/stripe/ops_interfaces.cpp.inc"

/*
void PrintSimple(Operation* op, OpAsmPrinter* p, ArrayRef<StringRef> fixed, Type otype) {
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
    *p << " : ";
    p->printType(otype);
  }
  p->printOptionalAttrDict(op->getAttrs(), fixed);
}

Type ParseSimple(OpAsmParser* p, OperationState *res, ArrayRef<StringRef> fixed, bool with_type) {
  bool first = true;
  Type out_type;
  ParseResult r;
  for (size_t i = 0; i < op->getNumOperands(); i++) {
    if (!first) {
      r = r || p->parseComma();
    }
    first = false;
    OperandType op;
    r = r || p->parseOperand(op);
    r = r || p->resolveOperand(op, Type(), res->operands);
  }
  for (StringRef name : fixed) {
    if (!first) {
      r = r || p->parseComma();
    }
    first = false;
    r->attributes.
    *p << op->getAttrOfType<IntegerAttr>(name).getValue();
  }
  if (with_type) {
    r = r || p->parseColon();
    r = r || p->parseType(out_type);
  }
  r = r || p->parseOptionalAttributeDict(res->attributes);
  return r;
}
*/

#define GET_OP_CLASSES
#include "pmlc/dialect/stripe/ops.cpp.inc"

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
