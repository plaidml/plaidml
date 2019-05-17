// Copyright 2019, Intel Corp.

#include <memory>
#include <regex>
#include <string>
#include <vector>

#include <boost/format.hpp>

#include "base/util/stream_container.h"
#include "tile/codegen/alias.h"
#include "tile/codegen/pattern.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace pattern {

enum class TokenType {
  None,
  Atom,
  Number,
  Variable,
  Punctuation,
  Whitespace,
};

struct Token {
  TokenType type;
  std::string value;
};

static const std::vector<Token> kTokens{
    Token{TokenType::Whitespace, "//.*\\n|[[:space:]]+"},     //
    Token{TokenType::Number, "[+\\-]?[[:digit:]]+"},          //
    Token{TokenType::Atom, "[[:lower:]][_[:alnum:]]*"},       //
    Token{TokenType::Variable, "[_[:upper:]][_[:alnum:]]*"},  //
    Token{TokenType::Punctuation, "[(){}[\\],.]"},            //
};

std::string MakeTokensRegex() {
  std::stringstream ss;
  for (size_t i = 0; i < kTokens.size(); i++) {
    if (i) {
      ss << "|";
    }
    ss << "(" << kTokens[i].value << ")";
  }
  return ss.str();
}

const std::regex* TokensRegex() {
  static std::regex re{MakeTokensRegex()};
  return &re;
}

class Lexer {
 public:
  explicit Lexer(const std::string& code) : end_{TokenType::None} {
    auto tokens_begin = std::sregex_iterator(code.begin(), code.end(), *TokensRegex());
    auto tokens_end = std::sregex_iterator{};
    for (auto it = tokens_begin; it != tokens_end; ++it) {
      if (it->prefix().length()) {
        throw std::runtime_error(str(boost::format("Invalid token found: %1%") % it->prefix()));
      }
      for (size_t i = 0; i < it->size(); i++) {
        // Use (i + 1) since the 0 submatch is the whole result
        if (it->str(i + 1).size()) {
          if (kTokens[i].type != TokenType::Whitespace) {  // skip whitespace & comments
            tokens_.emplace_back(Token{kTokens[i].type, it->str()});
          }
          break;
        }
      }
    }
  }

  const Token& cur() const {
    if (it_ < tokens_.size()) {
      return tokens_[it_];
    }
    return end_;
  }

  bool done() { return it_ == tokens_.size(); }

  Token next() {
    if (it_ < tokens_.size()) {
      it_++;
    }
    return cur();
  }

 private:
  size_t it_ = 0;
  std::vector<Token> tokens_;
  Token end_;
};

std::vector<std::string> GetTokens(const std::string& code) {
  std::vector<std::string> tokens;
  for (Lexer lexer(code); !lexer.done(); lexer.next()) {
    tokens.push_back(lexer.cur().value);
  }
  return tokens;
}

class Parser {
 public:
  explicit Parser(const std::string& code) : lexer_(code) {}

  Term parse_term() {
    if (cur() == "[") {
      lexer_.next();  // eat '['
      auto ret = std::make_shared<List>();
      ret->elts = parse_terms("]");
      return std::move(ret);
    }
    if (cur() == "{") {
      lexer_.next();  // eat '{'
      auto ret = std::make_shared<Set>();
      ret->elts = parse_terms("}");
      return std::move(ret);
    }
    auto token = lexer_.cur();
    lexer_.next();
    if (token.type == TokenType::Variable) {
      return Variable{token.value};
    }
    if (cur() != "(") {
      if (token.type == TokenType::Number) {
        return boost::lexical_cast<Number>(token.value);
      }
      return token.value;
    }
    lexer_.next();  // eat '('
    auto ret = std::make_shared<Struct>(token.value);
    ret->args = parse_terms(")");
    return std::move(ret);
  }

 private:
  std::vector<Term> parse_terms(const std::string& delimiter) {
    std::vector<Term> ret;
    while (cur() != delimiter) {
      ret.emplace_back(parse_term());
      if (cur() != "," && cur() != delimiter) {
        throw std::runtime_error(
            str(boost::format("Expected: ',' or '%1%' in term, but got: '%2%'") % delimiter % cur()));
      }
      if (cur() == ",") {
        lexer_.next();  // eat ','
      }
    }
    lexer_.next();  // eat delimiter
    return ret;
  }

  const std::string& cur() const { return lexer_.cur().value; }

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

  void operator()(const std::shared_ptr<Struct>& x) {
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
  auto p_block = std::make_shared<Struct>("block");
  auto refs_list = std::make_shared<List>();
  for (const auto& ref : block.refs) {
    auto p_ref = std::make_shared<Struct>("ref");
    switch (ref.dir) {
      case RefDir::In:
        p_ref->args.emplace_back("in");
        break;
      case RefDir::Out:
        p_ref->args.emplace_back("out");
        break;
      case RefDir::InOut:
        p_ref->args.emplace_back("inout");
        break;
      default:
        throw std::runtime_error("Invalid dir");
    }
    auto dims_list = std::make_shared<List>();
    for (size_t i = 0; i < ref.access.size(); i++) {
      auto p_dim = std::make_shared<Struct>("dim");
      auto terms_list = std::make_shared<List>();
      int64_t offset = 0;
      for (const auto& term : ref.access[i].getMap()) {
        if (term.first.empty()) {
          offset = term.second;
          continue;
        }
        auto p_term = std::make_shared<Struct>("term");
        p_term->args.emplace_back(term.second);
        p_term->args.emplace_back(term.first);
        terms_list->elts.emplace_back(std::move(p_term));
      }
      p_dim->args.emplace_back(offset);
      p_dim->args.emplace_back(std::move(terms_list));
      p_dim->args.emplace_back(ref.interior_shape.dims[i].size);
      p_dim->args.emplace_back(ref.interior_shape.dims[i].stride);
      dims_list->elts.emplace_back(std::move(p_dim));
    }
    p_ref->args.emplace_back(std::move(dims_list));
    refs_list->elts.emplace_back(std::move(p_ref));
  }
  p_block->args.emplace_back(std::move(refs_list));
  auto idxs_list = std::make_shared<List>();
  for (const auto& idx : block.idxs) {
    if (idx.affine != Affine{}) {
      continue;
    }
    auto p_idx = std::make_shared<Struct>("idx");
    p_idx->args.emplace_back(idx.name);
    p_idx->args.emplace_back(idx.range);
    idxs_list->elts.emplace_back(std::move(p_idx));
  }
  p_block->args.emplace_back(std::move(idxs_list));
  return std::move(p_block);
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
  MatchVisitor() : choices_{MatchResult{}} {}
  explicit MatchVisitor(const MatchVisitor& rhs) : choices_{rhs.choices_} {}

  const std::list<MatchResult>& matches() const { return choices_; }

  template <typename L, typename R>
  bool operator()(const L& lhs, const R& rhs) {
    return false;
  }

  bool operator()(const Atom& lhs, const Atom& rhs) { return lhs == rhs; }

  bool operator()(const Number& lhs, const Number& rhs) { return lhs == rhs; }

  bool operator()(const Variable& lhs, const Atom& rhs) { return compare_var(lhs, rhs); }

  bool operator()(const Variable& lhs, const Number& rhs) { return compare_var(lhs, rhs); }

  bool operator()(const std::shared_ptr<List>& lhs, const std::shared_ptr<List>& rhs) {
    return compare_terms(this, lhs->elts, rhs->elts);
  }

  bool operator()(const std::shared_ptr<Set>& lhs, const std::shared_ptr<List>& rhs) {
    std::sort(rhs->elts.begin(), rhs->elts.end(), TermCompare());
    std::list<MatchResult> merged;
    do {
      MatchVisitor branch(*this);
      if (compare_terms(&branch, lhs->elts, rhs->elts)) {
        // TODO: handle duplicates when merging branches
        for (const auto& match : branch.matches()) {
          merged.emplace_back(match);
        }
      }
    } while (std::next_permutation(rhs->elts.begin(), rhs->elts.end(), TermCompare()));
    choices_ = merged;
    return choices_.size();
  }

  bool operator()(const std::shared_ptr<Struct>& lhs, const std::shared_ptr<Struct>& rhs) {
    if (lhs->functor != rhs->functor) {
      return false;
    }
    return compare_terms(this, lhs->args, rhs->args);
  }

 private:
  bool compare_var(const Variable& lhs, const Term& rhs) {
    if (lhs.name[0] == '_') {
      return true;
    }
    auto it_choices = choices_.begin();
    while (it_choices != choices_.end()) {
      bool is_new;
      std::map<std::string, Term>::iterator it;
      std::tie(it, is_new) = it_choices->vars.emplace(lhs.name, rhs);
      if (!is_new && to_string(it->second) != to_string(rhs)) {
        it_choices = choices_.erase(it_choices);
      } else {
        ++it_choices;
      }
    }
    return choices_.size();
  }

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
  std::list<MatchResult> choices_;
};

boost::optional<MatchResult> MatchFirst(const Term& pattern, const Term& value) {
  MatchVisitor visitor;
  if (boost::apply_visitor(visitor, pattern, value)) {
    return visitor.matches().front();
  }
  return boost::none;
}

std::list<MatchResult> MatchAll(const Term& pattern, const Term& value) {
  MatchVisitor visitor;
  if (boost::apply_visitor(visitor, pattern, value)) {
    return visitor.matches();
  }
  return std::list<MatchResult>{};
}

std::string to_string(const MatchResult& result) {
  std::stringstream ss;
  ss << StreamContainer(result.vars);
  return ss.str();
}

std::vector<std::string> to_string(const std::list<MatchResult>& results) {
  std::vector<std::string> ret;
  for (const auto& result : results) {
    ret.push_back(to_string(result));
  }
  return ret;
}

}  // namespace pattern

void PatternPass::Apply(Block* block) const {
  auto reqs = FromProto(options_.reqs());
  auto pattern = pattern::Parse(options_.pattern());
  RunOnBlocks(block, reqs, [&](const AliasMap& map, Block* block) {
    auto term = pattern::IntoTerm(*block);
    auto match = pattern::MatchFirst(pattern, term);
    if (match) {
      IVLOG(2, "PatternPass> block: " << block->name);
      for (const auto& kvp : options_.set_vars()) {
        auto value = boost::get<pattern::Number>(safe_at(match->vars, kvp.second));
        IVLOG(2, "  " << kvp.first << " = " << value);
        block->set_attr(kvp.first, value);
      }
    }
  });
}

namespace {

[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<PatternPass, proto::PatternPass>::Register();
  return 0;
}();

}  // namespace

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
