#include <gtest/gtest.h>

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;  // NOLINT

namespace {

// This test is for a crash discovered under VS 2019.
// When /std:c++14 and /std:c++17 compiler flags in libraries are mixed,
// the following test will crash during the dtor of the ConversionTarget.
// However, if the "Control" test is un-commented, the crash will disappear.
TEST(InconsistentCppStd, CanCrash) {
  MLIRContext context;
  ConversionTarget target(context);
  OperationName op("func", &context);
  target.setOpAction(op, ConversionTarget::LegalizationAction::Illegal);
}

// If the above test is crashing, un-comment this test to see it succeed.
// TEST(InconsistentCppStd, Control) {
//   MLIRContext context;
//   llvm::MapVector<OperationName, std::string> map;
//   OperationName op("func", &context);
//   map[op] = "";
// }

}  // end namespace
