// Copyright 2021, Intel Corporation

#include "pmlc/util/schedule.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "pmlc/util/schedule_attrdef.cc.inc"

using namespace mlir; // NOLINT

namespace pmlc::util {

void PMLDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "pmlc/util/schedule_attrdef.cc.inc" // NOLINT
      >();
}

Attribute PMLDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return Attribute();
  Attribute attr;
  auto parseResult =
      generatedAttributeParser(getContext(), parser, attrTag, type, attr);
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

Attribute AxisAttr::parse(MLIRContext *context, DialectAsmParser &parser,
                          Type type) {
  StringRef typeStr;
  IntegerAttr range;
  if (parser.parseKeyword(&typeStr) || //
      parser.parseColon() ||           //
      parser.parseAttribute(range)) {
    parser.emitError(parser.getNameLoc(), "expected 'type:range'");
    return {};
  }

  auto typeAttr = StringAttr::get(context, typeStr);
  return parser.getChecked<AxisAttr>(context, typeAttr, range.getInt());
}

void AxisAttr::print(DialectAsmPrinter &printer) const {
  printer << getName().getValue() << ':' << getRange();
}

Attribute ScheduleAttr::parse(MLIRContext *context, DialectAsmParser &parser,
                              Type type) {
  if (failed(parser.parseLess()))
    return {};

  AffineMap map;
  if (failed(parser.parseAffineMap(map))) {
    parser.emitError(parser.getNameLoc(), "expected an affine map");
    return {};
  }

  if (parser.parseComma() || parser.parseLSquare())
    return {};

  SmallVector<AxisAttr> axes;
  do {
    Attribute attr = AxisAttr::parse(context, parser, type);
    axes.push_back(attr.cast<AxisAttr>());
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRSquare() || parser.parseGreater())
    return {};

  return parser.getChecked<ScheduleAttr>(context, map, axes);
}

void ScheduleAttr::print(DialectAsmPrinter &printer) const {
  printer << "schedule<" << getMap() << ", [";
  llvm::interleaveComma(getAxes(), printer,
                        [&](AxisAttr axis) { axis.print(printer); });
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

} // namespace pmlc::util
