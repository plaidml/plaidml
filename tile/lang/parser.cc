#include <utility>

#include "base/util/catch.h"
#include "tile/lang/parser.h"
#include "tile/lang/parser.y.h"
#include "tile/lang/parser_lex.h"
#include "tile/lang/type.h"

namespace vertexai {
namespace tile {
namespace lang {

Context parse_helper(const std::string& code, int64_t start_tmp, const std::string& id = "") {
  try {
    Context ctx;
    ctx.program.next_tmp = start_tmp;
    ctx.id = id;
    yyscan_t scanner;
    yylex_init(&scanner);
    YY_BUFFER_STATE state = yy_scan_string(code.c_str(), scanner);
    yyparse(scanner, ctx);
    yy_delete_buffer(state, scanner);
    yylex_destroy(scanner);
    return ctx;
  } catch (std::invalid_argument& e) {
    std::string err = std::string(e.what()) + " : " + code;
    LOG(ERROR) << err;
    throw std::invalid_argument(err);
  }
}

Program Parser::Parse(const std::string& code, const std::string& id) const {
  return parse_helper(code, 0, id).program;
}

Program Parser::ParseExpr(const std::string& code, int64_t start_tmp) const {
  return parse_helper("expression " + code + ";", start_tmp).program;
}

Polynomial<Rational> Parser::ParsePolynomial(const std::string& code) const {
  Bindings empty;
  SymbolicPolynomialPtr sp = parse_helper("polynomial " + code + ";", 0).polynomial;
  Polynomial<Rational> p = sp->Evaluate(empty);
  return p;
}

Contraction Parser::ParseContraction(const std::string& code) const {
  Bindings iconsts;
  Context ctx = parse_helper("contraction " + code + ";", 0);
  for (const auto& op : ctx.program.ops) {
    if (op.tag == Op::CONSTANT && op.f.fn == "iconst") {
      iconsts.emplace(op.output, Binding(int64_t(std::stoll(op.inputs[0].c_str()))));
    }
  }
  Contraction c = ctx.program.ops[ctx.program.ops.size() - 1].c;
  for (auto& ts : c.specs) {
    for (auto& ss : ts.sspec) {
      ts.spec.push_back(ss->Evaluate(iconsts));
    }
    ts.sspec.clear();
  }
  for (auto& con : c.constraints) {
    con.bound = RangeConstraint(con.poly->Evaluate(iconsts), iconsts.at(con.range).iconst);
    con.poly = SymbolicPolynomialPtr();
    con.range = "";
  }
  IVLOG(1, to_string(c));
  return c;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
