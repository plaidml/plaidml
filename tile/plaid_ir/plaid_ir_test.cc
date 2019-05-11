
#include <iostream>

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"

int main() {
  mlir::MLIRContext context;
  std::unique_ptr<mlir::Module> module;
  printf("Hello IR\n");
  module = std::unique_ptr<mlir::Module>(mlir::parseSourceFile("woot.mlir", &context));
}
