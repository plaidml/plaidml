// Copyright 2021, Intel Corporation

#include "pmlc/dialect/pml/ir/dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "pmlc/dialect/pml/ir/attrdef.cc.inc"

#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pml {

void PMLDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "pmlc/dialect/pml/ir/attrdef.cc.inc" // NOLINT
      >();
}

Attribute PMLDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return Attribute();
  Attribute attr;
  auto parseResult =
      generatedAttributeParser(parser, attrTag, type, attr);
  if (parseResult.hasValue())
    return attr;
  parser.emitError(parser.getNameLoc(), "unknown schedule attribute");
  return Attribute();
}

void PMLDialect::printAttribute(Attribute attr,
                                DialectAsmPrinter &printer) const {
  if (succeeded(generatedAttributePrinter(attr, printer)))
    return;
}

static ParseResult parseAxisAttr(MLIRContext *context, DialectAsmParser &parser,
                                 AxisAttr &attr) {
  StringRef typeStr;
  IntegerAttr range;
  if (parser.parseKeyword(&typeStr) || //
      parser.parseColon() ||           //
      parser.parseAttribute(range)) {
    parser.emitError(parser.getNameLoc(), "expected '$type:$range'");
    return failure();
  }

  auto typeAttr = StringAttr::get(context, typeStr);
  attr = parser.getChecked<AxisAttr>(context, typeAttr, range.getInt());

  return success();
}

static void printAxisAttr(DialectAsmPrinter &printer, AxisAttr axis) {
  printer << axis.getName().getValue() << ':' << axis.getRange();
}

Attribute AxisAttr::parse(DialectAsmParser &parser,
                          Type type) {
  AxisAttr attr;
  if (parser.parseLess() ||                   //
      parseAxisAttr(parser.getContext(), parser, attr) || //
      parser.parseGreater())
    return {};
  return attr;
}

void AxisAttr::print(DialectAsmPrinter &printer) const {
  printer << "axis<" << getName().getValue() << ':' << getRange() << '>';
}

Attribute ScheduleAttr::parse(DialectAsmParser &parser,
                              Type type) {
  AffineMap map;
  if (parser.parseLess() ||         //
      parser.parseAffineMap(map) || //
      parser.parseComma() ||        //
      parser.parseLSquare()) {
    parser.emitError(parser.getNameLoc(), "expected '$map, $axes'");
    return {};
  }

  SmallVector<AxisAttr> axes;
  do {
    AxisAttr axis;
    if (parseAxisAttr(parser.getContext(), parser, axis))
      return {};
    axes.push_back(axis);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRSquare() || //
      parser.parseGreater())
    return {};

  return parser.getChecked<ScheduleAttr>(parser.getContext(), map, axes);
}

void ScheduleAttr::print(DialectAsmPrinter &printer) const {
  printer << "schedule<" << getMap() << ", [";
  llvm::interleaveComma(getAxes(), printer,
                        [&](AxisAttr axis) { printAxisAttr(printer, axis); });
  printer << "]>";
}

Optional<AxisDim> ScheduleAttr::getAxisInputDim(StringRef name) {
  for (auto it : llvm::enumerate(getAxes())) {
    AxisAttr axis = it.value();
    if (axis.getName().getValue() == name)
      return AxisDim{axis, it.index()};
  }
  return None;
}

Optional<AxisDim> ScheduleAttr::getAxisResultDim(StringRef name) {
  Optional<AxisDim> axisInputDim = getAxisInputDim(name);
  if (!axisInputDim)
    return None;

  AffineMap map = getMap();
  SmallVector<int64_t> values(map.getNumInputs());
  values[axisInputDim->dim] = 1;

  SmallVector<int64_t, 4> results = map.compose(values);

  SmallVector<size_t> dims;
  for (auto it : llvm::enumerate(results)) {
    if (it.value()) {
      if (!dims.empty())
        return None;
      dims.push_back(it.index());
    }
  }

  if (dims.empty())
    return None;

  return AxisDim{axisInputDim->axis, dims.front()};
}

ScheduleAttr ScheduleAttr::removeAxes(DenseSet<StringRef> names) {
  SmallVector<unsigned> toKeep;
  SmallVector<AxisAttr> axes;
  for (auto it : llvm::enumerate(getAxes())) {
    AxisAttr axis = it.value();
    if (!names.contains(axis.getName().getValue())) {
      toKeep.push_back(it.index());
      axes.push_back(axis);
    }
  }

  if (axes.empty())
    return {};

  AffineMap map = getMap().getSubMap(toKeep);
  return ScheduleAttr::get(getContext(), map, axes);
}

Attribute PatternAttr::parse(DialectAsmParser &parser,
                             Type type) {
  StringAttr op;
  DictionaryAttr dict;
  if (parser.parseLess() ||          //
      parser.parseAttribute(op) ||   //
      parser.parseComma() ||         //
      parser.parseAttribute(dict) || //
      parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected '$op, $dict'");
    return {};
  }
  return parser.getChecked<PatternAttr>(parser.getContext(), op, dict);
}

void PatternAttr::print(DialectAsmPrinter &printer) const {
  printer << "pattern<" << getOp() << ", " << getDict() << '>';
}

Attribute ApplyAttr::parse(DialectAsmParser &parser,
                           Type type) {
  PatternAttr pattern;
  DictionaryAttr dict;
  if (parser.parseLess() ||             //
      parser.parseAttribute(pattern) || //
      parser.parseComma() ||            //
      parser.parseAttribute(dict) ||    //
      parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected '$pattern, $dict'");
    return {};
  }
  return parser.getChecked<ApplyAttr>(parser.getContext(), pattern, dict);
}

void ApplyAttr::print(DialectAsmPrinter &printer) const {
  printer << "apply<" << getPattern() << ", " << getDict() << '>';
}

} // namespace pmlc::dialect::pml

#include "pmlc/dialect/pml/ir/dialect.cc.inc" // NOLINT
