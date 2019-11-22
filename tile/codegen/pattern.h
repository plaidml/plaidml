// Copyright 2019, Intel Corp.

#pragma once

#include <list>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace pattern {

using Atom = std::string;

using Number = int64_t;

struct Variable {
  std::string name;
};

struct Struct;
struct List;
struct Set;

using Term = std::variant<   //
    Atom,                    //
    Number,                  //
    Variable,                //
    std::shared_ptr<List>,   //
    std::shared_ptr<Set>,    //
    std::shared_ptr<Struct>  //
    >;

struct Struct {
  explicit Struct(const Atom& functor) : functor(functor) {}
  Atom functor;
  std::vector<Term> args;
};

// An ordered list of terms.
struct List {
  std::vector<Term> elts;
};

// A permutable list of terms.
struct Set {
  std::vector<Term> elts;
};

struct MatchResult {
  std::map<std::string, Term> vars;
};

// Attempts to find the first permutation of 'value' that matches the supplied 'pattern'.
std::optional<MatchResult> MatchFirst(const Term& pattern, const Term& value);

// Attempts to find all permutations of 'value' that matches the supplied 'pattern'.
std::list<MatchResult> MatchAll(const Term& pattern, const Term& value);

// For testing the lexer
std::vector<std::string> GetTokens(const std::string& code);

// Parse a prolog-like string into a generic term.
Term Parse(const std::string& code);

// Transform a stripe block into a generic term with the following form:
// block([
//   ref(<dir>, [
//     dim(
//       <access.offset>,
//       [term(<factor>, <idx>), ...],
//       <shape.size>,
//       <shape.stride>
//     ),
//     ...
//   ]),
//   ...
// ])
Term IntoTerm(const stripe::Block& block);

std::ostream& operator<<(std::ostream& os, const Term& term);

inline std::string to_string(const Term& term) {
  std::stringstream ss;
  ss << term;
  return ss.str();
}

std::string to_string(const MatchResult& result);
std::vector<std::string> to_string(const std::list<MatchResult>& results);

}  // namespace pattern

class PatternPass final : public CompilePass {
 public:
  explicit PatternPass(const proto::PatternPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::PatternPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
