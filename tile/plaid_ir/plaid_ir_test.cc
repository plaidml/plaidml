
#include <iostream>

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"

int main() {
  mlir::MLIRContext context;
  printf("Hello IR\n");
  auto module = std::make_unique<mlir::Module>(&context);
}
