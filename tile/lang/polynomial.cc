#include "tile/lang/polynomial.h"

namespace vertexai {
namespace tile {
namespace lang {

Polynomial::Polynomial() {}

Polynomial::Polynomial(const Rational& c) : Polynomial("", c) {}

Polynomial::Polynomial(const std::string& i, const Rational& c) {
  if (c) {
    map_[i] = c;
  }
}

Rational Polynomial::eval(const std::map<std::string, Rational>& values) const {
  Rational res = 0;
  for (const auto& kvp : map_) {
    if (kvp.first == "") {
      res += kvp.second;
    } else if (values.find(kvp.first) != values.end()) {
      res += kvp.second * values.at(kvp.first);
    } else {
      throw std::runtime_error(
          printstring("Failed to find value for %s, when evaluating %s", kvp.first.c_str(), this->toString().c_str()));
    }
  }
  return res;
}

Rational Polynomial::operator[](const std::string& var) const {
  auto it = map_.find(var);
  if (it == map_.end()) {
    return 0;
  }
  return it->second;
}
const std::map<std::string, Rational>& Polynomial::getMap() const { return map_; }

Polynomial& Polynomial::operator+=(const Polynomial& rhs) {
  for (const auto& kvp : rhs.map_) {
    Rational new_val = (map_[kvp.first] += kvp.second);
    if (new_val == 0) {
      map_.erase(kvp.first);
    }
  }
  return *this;
}

bool Polynomial::operator==(const Polynomial& rhs) const { return map_ == rhs.map_; }

bool Polynomial::operator<(const Polynomial& rhs) const { return map_ < rhs.map_; }

Polynomial& Polynomial::operator-=(const Polynomial& rhs) { return *this += -1 * rhs; }

Polynomial Polynomial::operator-() const { return -1 * (*this); }

Polynomial& Polynomial::operator*=(const Rational& rhs) {
  if (rhs == 0) {
    map_.clear();
  } else {
    for (auto& kvp : map_) {
      kvp.second *= rhs;
    }
  }
  return *this;
}

Polynomial& Polynomial::operator/=(const Rational& rhs) { return *this *= (1 / rhs); }

Rational Polynomial::constant() const {
  auto it = map_.find("");
  return (it == map_.end() ? 0 : it->second);
}

void Polynomial::setConstant(Rational value) { map_[""] = value; }

Rational Polynomial::tryDivide(const Polynomial& p, bool ignoreConst) const {
  auto it = p.map_.begin();
  if (ignoreConst && it != p.map_.end() && it->first == "") {
    it++;
  }
  Rational val = 0;
  for (const auto& kvp : map_) {
    if (ignoreConst && kvp.first == "") {
      continue;
    }
    if (it == p.map_.end() || it->first != kvp.first) {
      return 0;  // Indexes don't exactly line up, fail
    }
    Rational div = kvp.second / it->second;
    if (val != 0 && div != val) {
      return 0;  // They don't all divide by the same number
    }
    val = div;
    it++;
  }
  if (it != p.map_.end()) {
    return 0;
  }
  return val;
}

std::string Polynomial::GetNonzeroIndex() const {
  // Returns a nonconstant nonzero index, if one exists; otherwise returns empty string
  for (const auto& kvp : map_) {
    if (!(kvp.first.empty()) && kvp.second != 0) return kvp.first;
  }

  // No nonconstant index has a nonzero coefficient
  return std::string();
}

std::string Polynomial::toString() const {
  std::string r;
  if (map_.size() == 0) {
    return "0";
  }
  for (const auto& kvp : map_) {
    if (r != "") {
      if (kvp.second > 0) {
        r += " + ";
      } else {
        r += " - ";
      }
    } else if (r == "" && kvp.second < 0) {
      r += "-";
    }
    if (abs(kvp.second) != 1 || kvp.first == "") {
      r += to_string(abs(numerator(kvp.second)));
      if (denominator(kvp.second) != 1) {
        r += "/" + to_string(denominator(kvp.second));
      }
      if (kvp.first != "") {
        r += "*";
      }
    }
    r += kvp.first;
  }
  return r;
}

SimpleConstraint::SimpleConstraint(const Polynomial& _poly, int64_t _rhs) : poly(_poly), rhs(_rhs) {}

RangeConstraint::RangeConstraint(const Polynomial& _poly, int64_t _range) : poly(_poly), range(_range) {}

bool RangeConstraint::IsParallel(const RangeConstraint& c) {
  if (this->poly.tryDivide(c.poly, true) != 0) return true;
  return false;
}

SimpleConstraint RangeConstraint::lowerBound() const { return SimpleConstraint(-poly, 0); }

SimpleConstraint RangeConstraint::upperBound() const { return SimpleConstraint(poly, range - 1); }

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
