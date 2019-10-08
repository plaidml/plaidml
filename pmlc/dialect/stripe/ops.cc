// Copyright 2019, Intel Corporation
#include "pmlc/dialect/stripe/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc {
namespace dialect {
namespace stripe {

#include "pmlc/dialect/stripe/ops_interfaces.cpp.inc"

void PrintSimple(Operation* op, OpAsmPrinter* p, size_t count, ArrayRef<StringRef> fixed, Type otype, bool vararg) {
  // Print the op name
  *p << op->getName() << " ";
  // Pring the normal (fixed) operands
  p->printOperands(op->operand_begin(), op->operand_begin() + count);
  // Print the fixed attributes (which are always integers in our case)
  bool first = (count == 0);
  for (StringRef name : fixed) {
    if (!first) {
      *p << ", ";
    }
    first = false;
    *p << op->getAttrOfType<IntegerAttr>(name).getValue();
  }
  // If we can have varargs, print them wrapped in ()'s
  if (vararg) {
    if (count > 0) {
      *p << " ";
    }
    *p << "(";
    p->printOperands(op->operand_begin() + count, op->operand_end());
    *p << ")";
  }
  // Print a type (if needed)
  if (otype) {
    *p << " : " << otype;
  }
  // Print any additional attributes
  p->printOptionalAttrDict(op->getAttrs(), fixed);
}

template <typename T>
bool ParseSimple(OpAsmParser* p, OperationState* res, llvm::SmallVectorImpl<OpAsmParser::OperandType>* ops,
                 size_t count, ArrayRef<StringRef> fixed, T* out_type, bool vararg) {
  bool r = false;
  // Parse the normal operands, annoyingly parseOperandList doesn't
  // have an option to read exactly N operands, only to read all and verify
  bool first = true;
  for (size_t i = 0; i < count; i++) {
    if (!first) {
      r = r || p->parseComma();
    }
    first = false;
    OpAsmParser::OperandType op;
    r = r || p->parseOperand(op);
    ops->push_back(op);
  }
  // Parse the fixed attributes
  for (StringRef name : fixed) {
    if (!first) {
      r = r || p->parseComma();
    }
    Attribute dont_care;
    r = r || p->parseAttribute(dont_care, name, res->attributes);
    first = false;
  }
  // If we can have varargs, parse them wrapped in ()'s
  if (vararg) {
    r = r || p->parseOperandList(*ops, -1, OpAsmParser::Delimiter::Paren);
  }
  // Parse a type if needed
  if (out_type) {
    r = r || p->parseColon();
    r = r || p->parseType(*out_type);
  }
  // Parse any additional attributes
  r = r || p->parseOptionalAttributeDict(res->attributes);
  return r;
}

#define GET_OP_CLASSES
#include "pmlc/dialect/stripe/ops.cpp.inc"

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
