// Copyright 2019, Intel Corp.

#include <memory>
#include <string>
#include <vector>

#include <boost/format.hpp>

#include "tile/codegen/pattern.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace pattern {

using namespace stripe;  // NOLINT

static std::locale kLocale{"en_US.UTF-8"};

class Lexer {
 public:
  explicit Lexer(const std::string& code) : ss_(code) {}

  const std::string& cur() const { return cur_; }
  bool done() { return ss_.eof() || ss_.peek() == -1; }

  std::string next() {
    // tokens:
    // ( ) , . [ ] { } atom number
    cur_.clear();
    bool first = true;
    while (!done()) {
      char ch = ss_.peek();
      if (std::isspace(ch, kLocale)) {
        if (first) {
          ss_.get();
          continue;
        }
        return cur_;
      }
      switch (ch) {
        case '(':
        case ')':
        case '[':
        case ']':
        case ',':
        case '.':
        case '{':
        case '}':
        case '_':
          if (first) {
            ss_.get();
            cur_.push_back(ch);
          }
          return cur_;
      }
      ss_.get();
      cur_.push_back(ch);
      first = false;
    }
    return cur_;
  }

 private:
  std::istringstream ss_;
  std::string cur_;
};

std::vector<std::string> GetTokens(const std::string& code) {
  Lexer lexer(code);
  std::vector<std::string> tokens;
  while (!lexer.done()) {
    tokens.push_back(lexer.next());
  }
  return tokens;
}

class Parser {
 public:
  explicit Parser(const std::string& code) : lexer_(code) { lexer_.next(); }

  Term parse_term() {
    if (lexer_.cur() == "[") {
      lexer_.next();  // eat '['
      auto ret = std::make_shared<List>();
      ret->elts = parse_terms("]");
      return std::move(ret);
    }
    if (lexer_.cur() == "{") {
      lexer_.next();  // eat '{'
      auto ret = std::make_shared<Set>();
      ret->elts = parse_terms("}");
      return std::move(ret);
    }
    auto token = lexer_.cur();
    lexer_.next();
    if (token == "_" || std::isupper(token[0], kLocale)) {
      return Variable{token};
    }
    if (lexer_.cur() != "(") {
      try {
        return boost::lexical_cast<Number>(token);
      } catch (boost::bad_lexical_cast&) {
        return token;
      }
    }
    lexer_.next();  // eat '('
    auto ret = std::make_shared<Predicate>(token);
    ret->args = parse_terms(")");
    return std::move(ret);
  }

 private:
  std::vector<Term> parse_terms(const std::string& delimiter) {
    std::vector<Term> ret;
    while (lexer_.cur() != delimiter) {
      ret.emplace_back(parse_term());
      if (lexer_.cur() != "," && lexer_.cur() != delimiter) {
        throw std::runtime_error(
            str(boost::format("Expected: ',' or '%1%' in term, but got: '%2%'") % delimiter % lexer_.cur()));
      }
      if (lexer_.cur() == ",") {
        lexer_.next();  // eat ','
      }
    }
    lexer_.next();  // eat delimiter
    return ret;
  }

 private:
  Lexer lexer_;
};

Term Parse(const std::string& code) {
  Parser parser(code);
  return parser.parse_term();
}

class TermPrinter : public boost::static_visitor<void> {
 public:
  explicit TermPrinter(std::ostream& os) : os_(os) {}

  void operator()(const Atom& x) { os_ << x; }

  void operator()(const Number& x) { os_ << x; }

  void operator()(const Variable& x) { os_ << x.name; }

  void operator()(const std::shared_ptr<List>& x) {
    os_ << "[";
    print_terms(x->elts);
    os_ << "]";
  }

  void operator()(const std::shared_ptr<Set>& x) {
    os_ << "{";
    print_terms(x->elts);
    os_ << "}";
  }

  void operator()(const std::shared_ptr<Predicate>& x) {
    os_ << x->functor;
    os_ << "(";
    print_terms(x->args);
    os_ << ")";
  }

  void print_terms(const std::vector<Term>& terms) {
    for (size_t i = 0; i < terms.size(); i++) {
      if (i) {
        os_ << ", ";
      }
      boost::apply_visitor(*this, terms[i]);
    }
  }

 private:
  std::ostream& os_;
};

std::ostream& operator<<(std::ostream& os, const Term& term) {
  TermPrinter printer(os);
  boost::apply_visitor(printer, term);
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<Term>& terms) {
  TermPrinter printer(os);
  printer.print_terms(terms);
  return os;
}

Term IntoTerm(const Block& block) {
  auto ct_block = std::make_shared<Predicate>("block");
  auto refs_list = std::make_shared<List>();
  for (const auto& ref : block.refs) {
    auto ct_ref = std::make_shared<Predicate>("ref");
    switch (ref.dir) {
      case RefDir::In:
        ct_ref->args.emplace_back("in");
        break;
      case RefDir::Out:
        ct_ref->args.emplace_back("out");
        break;
      case RefDir::InOut:
        ct_ref->args.emplace_back("inout");
        break;
      default:
        throw std::runtime_error("Invalid dir");
    }
    auto dims_list = std::make_shared<List>();
    for (size_t i = 0; i < ref.access.size(); i++) {
      auto ct_dim = std::make_shared<Predicate>("dim");
      auto terms_list = std::make_shared<List>();
      int64_t offset = 0;
      for (const auto& term : ref.access[i].getMap()) {
        if (term.first.empty()) {
          offset = term.second;
          continue;
        }
        auto ct_term = std::make_shared<Predicate>("term");
        ct_term->args.emplace_back(term.second);
        ct_term->args.emplace_back(term.first);
        terms_list->elts.emplace_back(std::move(ct_term));
      }
      ct_dim->args.emplace_back(offset);
      ct_dim->args.emplace_back(std::move(terms_list));
      ct_dim->args.emplace_back(ref.interior_shape.dims[i].size);
      ct_dim->args.emplace_back(ref.interior_shape.dims[i].stride);
      dims_list->elts.emplace_back(std::move(ct_dim));
    }
    ct_ref->args.emplace_back(std::move(dims_list));
    refs_list->elts.emplace_back(std::move(ct_ref));
  }
  ct_block->args.emplace_back(std::move(refs_list));
  return std::move(ct_block);
}

struct TermCompare {
  bool operator()(const Term& lhs, const Term& rhs) {
    // N.B.: This isn't the best way to do this but it should work for now.
    return to_string(lhs) < to_string(rhs);
  }
};

// lhs is always the pattern.
// rhs is always the value to match against.
class MatchVisitor : public boost::static_visitor<bool> {
 public:
  const std::map<std::string, Term>& vars() const { return vars_; }

  template <typename L, typename R>
  bool operator()(const L& lhs, const R& rhs) {
    return false;
  }

  bool operator()(const Atom& lhs, const Atom& rhs) { return lhs == rhs; }

  bool operator()(const Number& lhs, const Number& rhs) { return lhs == rhs; }

  bool operator()(const Variable& lhs, const Atom& rhs) {
    if (lhs.name == "_") {
      return true;
    }
    std::map<std::string, Term>::iterator it;
    bool is_new;
    std::tie(it, is_new) = vars_.emplace(lhs.name, rhs);
    return is_new || to_string(it->second) == rhs;
  }

  bool operator()(const Variable& lhs, const Number& rhs) {
    if (lhs.name == "_") {
      return true;
    }
    std::map<std::string, Term>::iterator it;
    bool is_new;
    std::tie(it, is_new) = vars_.emplace(lhs.name, rhs);
    return is_new || to_string(it->second) == to_string(rhs);
  }

  bool operator()(const std::shared_ptr<List>& lhs, const std::shared_ptr<List>& rhs) {
    return compare_terms(this, lhs->elts, rhs->elts);
  }

  bool operator()(const std::shared_ptr<Set>& lhs, const std::shared_ptr<List>& rhs) {
    std::sort(lhs->elts.begin(), lhs->elts.end(), TermCompare());
    do {
      MatchVisitor branch(*this);
      if (compare_terms(&branch, lhs->elts, rhs->elts)) {
        vars_ = branch.vars();
        return true;
      }
    } while (std::next_permutation(lhs->elts.begin(), lhs->elts.end(), TermCompare()));
    return false;
  }

  bool operator()(const std::shared_ptr<Predicate>& lhs, const std::shared_ptr<Predicate>& rhs) {
    if (lhs->functor != rhs->functor) {
      return false;
    }
    return compare_terms(this, lhs->args, rhs->args);
  }

 private:
  bool compare_terms(MatchVisitor* visitor, const std::vector<Term>& lhs, const std::vector<Term>& rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
      if (!boost::apply_visitor(*visitor, lhs[i], rhs[i])) {
        return false;
      }
    }
    return true;
  }

 private:
  std::map<std::string, Term> vars_;
};

MatchResult Match(const Term& pattern, const Term& value) {
  MatchResult result;
  MatchVisitor visitor;
  result.matched = boost::apply_visitor(visitor, pattern, value);
  result.vars = visitor.vars();
  return result;
}

}  // namespace pattern
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
