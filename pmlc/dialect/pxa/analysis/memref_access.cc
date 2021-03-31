// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/analysis/memref_access.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "pmlc/dialect/pxa/ir/ops.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

void MemRefAccess::getAccessMap(AffineMap map, SmallVector<Value, 8> operands,
                                AffineValueMap *accessMap) {
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  accessMap->reset(map, operands);
}

} // namespace pmlc::dialect::pxa
