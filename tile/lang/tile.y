%output "parser.y.cc"
%defines "parser.y.h"
%define api.pure full
%error-verbose
 
%code requires {
#pragma once

#include <string>

#include "tile/lang/ops.h"
#include "tile/lang/sym_poly.h"
#include "base/util/logging.h"

using vertexai::tile::lang::Integer;
using vertexai::tile::lang::Rational;

struct Context {
  std::string id;
  vertexai::tile::lang::Program program;
  vertexai::tile::lang::SymbolicPolynomialPtr polynomial;
  std::vector<vertexai::tile::lang::SymbolicConstraint> constraints;
  vertexai::tile::lang::SymbolicSpec index_spec;
  std::vector<std::string> out_spec;

  inline std::string apply(const std::string& func, const std::vector<std::string>& inputs) {
    vertexai::tile::lang::Op op;
    op.tag = vertexai::tile::lang::Op::FUNCTION;
    op.f.fn = func;
    op.output = std::string("_T") + std::to_string(program.next_tmp++);
    op.inputs = inputs;
    program.ops.emplace_back(op);
    return op.output;
  }
  void finish_stmt() {
    if (id != "") {
      vertexai::tile::lang::Attribute attr;
      attr.name = "pid";
      attr.params.push_back(id);
      program.ops.back().attributes.emplace_back(attr);
    }
  }
};

struct Value {
  std::string s;
  int64_t i;
  vertexai::tile::lang::TensorSpec tensor_spec;
  vertexai::tile::lang::SymbolicPolynomialPtr poly;
  vertexai::tile::lang::CombinationOp comb_op;
  vertexai::tile::lang::AggregationOp agg_op;
  std::vector<std::string> expr_list;
  vertexai::tile::lang::Op op;
  vertexai::tile::lang::Attribute attr;
  std::vector<vertexai::tile::lang::Attribute> attr_list;
};

#define YYSTYPE Value

typedef void* yyscan_t;

}

%{
#include "tile/lang/parser.y.h"
#include "tile/lang/parser_lex.h"
using namespace vertexai::tile::lang;

int yyerror(yyscan_t s, Context& context, const char* message) {
  throw std::invalid_argument(message);
};
%}

%param { yyscan_t scanner }
%parse-param { Context &context }

%token FUNCTION "function"
%token POLYNOMIAL "polynomial" EXPRESSION "expression" CONTRACTION "contraction"
%token NO_DEFRACT "no_defract"
%token QUESTION "?" COLON ":"
%left QUESTION COLON
%token BIT_XOR "^"
%left BIT_XOR
%token BIT_OR "|"
%left BIT_OR
%token BIT_AND "&"
%left BIT_AND
%token EQ "==" NE "!=" 
%nonassoc EQ NE
%token GT ">" GE ">=" LT "<" LE "<=" 
%nonassoc GT GE LT LE
%token BIT_LEFT "<<" BIT_RIGHT ">>"
%left BIT_LEFT BIT_RIGHT
%token PLUS "+" MINUS "-"
%left PLUS MINUS
%token left MULT "*" DIV "/"
%left MULT DIV
%precedence UNARY
%token BIT_NOT "~"
%token COMMA "," ASSIGN "=" SEMICOLON ";" ARROW "->"
%token LBRACE "[" RBRACE "]" LPAREN "(" RPAREN ")" LCURLY "{" RCURLY "}" DOT "." 
%token LLBRACE "[[" RRBRACE "]]"
%token <s> ID IDXID
%token <i> INT_LITERAL
%token <s> FLOAT_LITERAL
%type <tensor_spec> tensor_spec out_spec
%type <poly> polynomial
%type <comb_op> comb_op
%type <agg_op> agg_op
%type <s> expr 
%type <expr_list> expr_list name_list
%type <op> unary_con binary_con ternary_con
%type <s> attribute_parameter
%type <expr_list> attribute_parameter_list attribute_parameters
%type <attr> attribute
%type <attr_list> attribute_list

%%

input
  : "polynomial" polynomial ";" { context.polynomial = $2; }
  | "expression" expr ";" {}
  | "contraction" contract ";" {}
  | function
;

stmt
  : contract { context.constraints.clear(); }
  | contract "no_defract" { context.program.ops.back().c.no_defract = true; context.constraints.clear(); }
  | assign
;

function
  : "function" "(" inputs ")" "->" "(" outputs ")" "{" body "}" {}
  | "function" "(" ")" "->" "(" outputs ")" "{" body "}" {}
;

attributed_stmt
  : stmt
  | attribute_list stmt { context.program.ops.back().attributes = std::move($1); }
;

attribute_list
  : attribute { $$ = std::vector<vertexai::tile::lang::Attribute>{$1}; }
  | attribute_list attribute { $1.emplace_back(std::move($2)); $$ = std::move($1); }
;

attribute
  : "[[" IDXID attribute_parameters "]]" { $$ = vertexai::tile::lang::Attribute{$2, std::move($3)}; }
;

attribute_parameters
  : /* empty */ { $$ = std::vector<std::string>(); }
  | "(" attribute_parameter_list ")" { $$ = std::move($2); }
;

attribute_parameter_list
  : attribute_parameter { $$ = std::vector<std::string>{std::move($1)}; }
  | attribute_parameter_list "," attribute_parameter { $1.emplace_back(std::move($3)); $$ = std::move($1); }
;

attribute_parameter:
  IDXID
;

body
  : attributed_stmt ";" { context.finish_stmt(); }
  | body attributed_stmt ";" {  context.finish_stmt(); }
;

name_list
  : ID { $$ = {$1}; }
  | name_list "," ID { $$ = std::move($1); $$.push_back($3); }
;

inputs 
  : input 
  | inputs "," input 
;

input
  : ID { context.program.inputs.push_back(Input{Input::VARIABLE, $1, {}}); }
  | ID "[" "]" { context.program.inputs.push_back(Input{Input::FIXED, $1, {}}); }
  | ID "[" name_list "]" { context.program.inputs.push_back(Input{Input::FIXED, $1, $3}); }
;

outputs
  : output 
  | outputs "," output 
;

output
  : ID { context.program.outputs.push_back($1); }
;
  
contract
  : unary_con { context.program.ops.push_back($1); }
  | unary_con constraints { context.program.ops.push_back($1); context.program.ops.back().c.constraints = context.constraints; }
  | binary_con { context.program.ops.push_back($1); }
  | binary_con constraints { context.program.ops.push_back($1); context.program.ops.back().c.constraints = context.constraints; }
  | ternary_con { context.program.ops.push_back($1); }
  | ternary_con constraints { context.program.ops.push_back($1); context.program.ops.back().c.constraints = context.constraints; }
;

assign
  : ID "=" expr {
      if (context.program.ops.size() == 0 || context.program.ops.back().output != $3) {
        context.apply("ident", {$3});
      }
      context.program.ops.back().output = $1; 
    }
;

expr_list
  : expr { $$ = {$1}; } 
  | expr_list "," expr { $$ = std::move($1); $$.push_back($3); }
;

expr
  : INT_LITERAL { $$ = context.apply("iconst", {std::to_string($1)}); context.program.ops.back().tag = Op::CONSTANT; }
  | FLOAT_LITERAL { $$ = context.apply("fconst", {$1}); context.program.ops.back().tag = Op::CONSTANT; }
  | ID { $$ = $1; }
  | ID "(" expr_list ")" { $$ = context.apply($1, $3); }
  | IDXID "(" expr_list ")" { $$ = context.apply($1, $3); }
  | expr "+" expr  { $$ = context.apply("add",  {$1, $3}); }
  | expr "-" expr  { $$ = context.apply("sub",  {$1, $3}); }
  | expr "*" expr  { $$ = context.apply("mul",  {$1, $3}); }
  | expr "/" expr  { $$ = context.apply("div",  {$1, $3}); }
  | expr "==" expr { $$ = context.apply("cmp_eq", {$1, $3}); }
  | expr "!=" expr { $$ = context.apply("cmp_ne", {$1, $3}); }
  | expr "<" expr  { $$ = context.apply("cmp_lt", {$1, $3}); }
  | expr ">" expr  { $$ = context.apply("cmp_gt", {$1, $3}); }
  | expr "<=" expr { $$ = context.apply("cmp_le", {$1, $3}); }
  | expr ">=" expr { $$ = context.apply("cmp_ge", {$1, $3}); }
  | "-" expr %prec UNARY { $$ = context.apply("neg", {$2}); }
  | expr "<<" expr { $$ = context.apply("bit_left", {$1, $3}); }
  | expr ">>" expr { $$ = context.apply("bit_right", {$1, $3}); }
  | expr "&" expr  { $$ = context.apply("bit_and", {$1, $3}); }
  | expr "|" expr  { $$ = context.apply("bit_or", {$1, $3}); }
  | expr "^" expr  { $$ = context.apply("bit_xor", {$1, $3}); }
  | "~" expr %prec UNARY { $$ = context.apply("bit_not", {$2}); }
  | "(" expr ")" { $$ = $2; }
  | expr "?" expr ":" expr { $$ = context.apply("cond", {$1, $3, $5}); }
;

unary_con
  : out_spec "=" agg_op "(" tensor_spec ")" {
      Contraction c; c.specs = {$1, $5}; c.agg_op = $3;
      c.output_size = context.out_spec;
      context.out_spec.clear();
      Op op;
      op.tag = Op::CONTRACTION;
      op.c = std::move(c);
      op.output = $1.id;
      op.inputs.push_back($5.id);
      $$ = op;
    }
;

binary_con
  : out_spec "=" agg_op "(" tensor_spec comb_op tensor_spec ")" {
      Contraction c; c.specs = {$1, $5, $7}; c.agg_op = $3; c.comb_op = $6; 
      c.output_size = context.out_spec;
      context.out_spec.clear();
      Op op;
      op.tag = Op::CONTRACTION;
      op.c = std::move(c);
      op.output = $1.id;
      op.inputs.push_back($5.id);
      op.inputs.push_back($7.id);
      $$ = op;
    }
;

ternary_con
  : out_spec "=" agg_op "(" tensor_spec comb_op tensor_spec "?" tensor_spec ")" {
      Contraction c; c.specs = {$1, $5, $7, $9}; 
      c.agg_op = $3; c.comb_op = CombinationOp::COND; 
      c.output_size = context.out_spec;
      context.out_spec.clear();
      Op op;
      op.tag = Op::CONTRACTION;
      op.c = std::move(c);
      op.output = $1.id;
      op.inputs.push_back($5.id);
      op.inputs.push_back($7.id);
      op.inputs.push_back($9.id);
      $$ = op;
    }
;

tensor_spec
  : ID "[" index_spec "]" { $$ = { $1, context.index_spec }; context.index_spec = SymbolicSpec(); }
  | ID "[" "]" { $$ = { $1, context.index_spec }; context.index_spec = SymbolicSpec(); }
;

out_spec
  : ID "[" index_spec "]" { $$ = { $1, context.index_spec }; context.index_spec = SymbolicSpec(); }
  | ID "[" index_spec ":" expr_list "]" { $$ = { $1, context.index_spec }; context.index_spec = SymbolicSpec(); context.out_spec = $5; }
  | ID "[" ":" "]" { $$ = { $1, context.index_spec }; context.index_spec = SymbolicSpec(); }
  | ID "[" "]" { $$ = { $1, context.index_spec }; context.index_spec = SymbolicSpec(); }
;

index_spec
  : polynomial                { context.index_spec.push_back($1); }
  | index_spec "," polynomial { context.index_spec.push_back($3); }
;

constraints
  : "," constraint
  | constraints "," constraint
;

constraint
  : polynomial "<" expr { context.constraints.emplace_back($1, $3); }
;

polynomial
  : INT_LITERAL               { $$ = SymbolicPolynomial::MakeLiteral($1); }
  | ID                        { $$ = SymbolicPolynomial::MakeSymbol($1); }
  | IDXID                     { $$ = SymbolicPolynomial::MakeIndex($1); }
  | INT_LITERAL IDXID         { 
    $$ = SymbolicPolynomial::MakeBinaryOp("*", SymbolicPolynomial::MakeLiteral($1), SymbolicPolynomial::MakeIndex($2)); 
  }
  | "-" polynomial %prec UNARY { $$ = SymbolicPolynomial::MakeUnaryOp("-", $2); }
  | polynomial "+" polynomial { $$ = SymbolicPolynomial::MakeBinaryOp("+", $1, $3); }
  | polynomial "-" polynomial { $$ = SymbolicPolynomial::MakeBinaryOp("-", $1, $3); }
  | polynomial "*" polynomial { $$ = SymbolicPolynomial::MakeBinaryOp("*", $1, $3); }
  | polynomial "/" polynomial { $$ = SymbolicPolynomial::MakeBinaryOp("/", $1, $3); }
  | "(" polynomial ")"        { $$ = $2; }
;
  
agg_op
  : "+" { $$ = AggregationOp::SUM; }
  | "*" { $$ = AggregationOp::PROD; }
  | ">" { $$ = AggregationOp::MAX; }
;

comb_op
  : "+"  { $$ = CombinationOp::PLUS; }
  | "*"  { $$ = CombinationOp::MULTIPLY; }
  | "==" { $$ = CombinationOp::EQ; }
;
%%
