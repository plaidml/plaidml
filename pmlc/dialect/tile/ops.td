// Copyright 2019, Intel Corporation

include "mlir/IR/OpBase.td"

def TileDialect : Dialect {
  let name = "tile";
  let cppNamespace = "pmlc::dialect::tile";
}

// def TensorType :
//     Type<
//         CPred<"$_self.isa<TensorType>()">,
//         "Plaid AST Tensor Type"
//     >;

class TileOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<TileDialect, mnemonic, traits> {}

def ContractionOp : TileOp<"contraction"> {
    let summary = "Contraction operation";
}

// class PlaidUnaryOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
//     PlaidAstOp<mnemonic, traits>,
//     Arguments<(ins TensorType:$arg)>,
//     Results<(outs TensorType:$res)>
// {
// //   // TODO: Implement
// //   let parser = [{ NGRAPH_CHECK(false, "No parser support"); return mlir::failure(); }];

// //   let verifier = [{ return verifyUnaryArithOp(this); }];
// }

// def CallOp : ExprOp<"call"> {
//   let summary = "call operation";
//   let arguments = (Variadic<AnyType>:$args);
//   let results = (outs AnyType);
// }

// def IntConst : Expr<"int"> {

// }



// def AbsOp : PlaidUnaryOp<"abs">;

