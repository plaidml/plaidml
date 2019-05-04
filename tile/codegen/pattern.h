// Copyright 2019, Intel Corp.

#pragma once

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <boost/variant.hpp>

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

struct Predicate;
struct List;
struct Set;

using Term = boost::variant<    //
    Atom,                       //
    Number,                     //
    Variable,                   //
    std::shared_ptr<List>,      //
    std::shared_ptr<Set>,       //
    std::shared_ptr<Predicate>  //
    >;

struct Predicate {
  explicit Predicate(const Atom& functor) : functor(functor) {}
  Atom functor;
  std::vector<Term> args;
};

// An ordered list of terms.
struct List {
  std::vector<Term> elts;
};

// An unordered list of terms.
struct Set {
  std::vector<Term> elts;
};

struct MatchResult {
  bool matched = false;
  std::map<std::string, Term> vars;
};

// Attempts to find the first permutation of 'value' that matches the supplied 'pattern'.
MatchResult Match(const Term& pattern, const Term& value);

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

}  // namespace pattern
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
