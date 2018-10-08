#include "tile/math/polynomial.h"

namespace vertexai {
namespace tile {
namespace math {

template <typename T>
Polynomial<T>::Polynomial() {}

template <typename T>
Polynomial<T>::Polynomial(const T& c) : Polynomial<T>("", c) {}

template <typename T>
Polynomial<T>::Polynomial(const std::string& i, const T& c) {
  if (c) {
    map_[i] = c;
  }
}

template <typename T>
T Polynomial<T>::eval(const std::map<std::string, T>& values) const {
  T res = 0;
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

template <typename T>
T Polynomial<T>::operator[](const std::string& var) const {
  auto it = map_.find(var);
  if (it == map_.end()) {
    return 0;
  }
  return it->second;
}

template <typename T>
const std::map<std::string, T>& Polynomial<T>::getMap() const {
  return map_;
}

template <typename T>
std::map<std::string, T>& Polynomial<T>::mutateMap() {
  return map_;
}

template <typename T>
Polynomial<T>& Polynomial<T>::operator+=(const Polynomial<T>& rhs) {
  for (const auto& kvp : rhs.map_) {
    T new_val = (map_[kvp.first] += kvp.second);
    if (new_val == 0) {
      map_.erase(kvp.first);
    }
  }
  return *this;
}

template <typename T>
bool Polynomial<T>::operator==(const Polynomial<T>& rhs) const {
  return map_ == rhs.map_;
}

template <typename T>
bool Polynomial<T>::operator<(const Polynomial<T>& rhs) const {
  return map_ < rhs.map_;
}

template <typename T>
Polynomial<T>& Polynomial<T>::operator-=(const Polynomial<T>& rhs) {
  return *this += -1 * rhs;
}

template <typename T>
Polynomial<T> Polynomial<T>::operator-() const {
  return -1 * (*this);
}

template <typename T>
Polynomial<T>& Polynomial<T>::operator*=(const T& rhs) {
  if (rhs == 0) {
    map_.clear();
  } else {
    for (auto& kvp : map_) {
      kvp.second *= rhs;
    }
  }
  return *this;
}

template <typename T>
Polynomial<T>& Polynomial<T>::operator/=(const T& rhs) {
  return *this *= (1 / rhs);
}

template <typename T>
T Polynomial<T>::constant() const {
  auto it = map_.find("");
  return (it == map_.end() ? 0 : it->second);
}

template <typename T>
void Polynomial<T>::setConstant(T value) {
  map_[""] = value;
}

template <typename T>
T Polynomial<T>::tryDivide(const Polynomial<T>& p, bool ignoreConst) const {
  auto it = p.map_.begin();
  if (ignoreConst && it != p.map_.end() && it->first == "") {
    it++;
  }
  T val = 0;
  for (const auto& kvp : map_) {
    if (ignoreConst && kvp.first == "") {
      continue;
    }
    if (it == p.map_.end() || it->first != kvp.first) {
      return 0;  // Indexes don't exactly line up, fail
    }
    T div = kvp.second / it->second;
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

template <typename T>
void Polynomial<T>::substitute(const std::string& var, const Polynomial<T>& replacement) {
  if (map_.count(var) == 0) {
    // If var isn't in this polynomial, nothing needs to be done
    return;
  }
  T coeff = map_.at(var);
  map_.erase(var);
  (*this) += coeff * replacement;
}

template <typename T>
void Polynomial<T>::substitute(const std::string& var, const T& replacement) {
  substitute(var, Polynomial<T>(replacement));
}

template <typename T>
std::string Polynomial<T>::GetNonzeroIndex() const {
  // Returns a nonconstant nonzero index, if one exists; otherwise returns empty string
  for (const auto& kvp : map_) {
    if (!(kvp.first.empty()) && kvp.second != 0) return kvp.first;
  }

  // No nonconstant index has a nonzero coefficient
  return std::string();
}

template <typename T>
T Polynomial<T>::get(const std::string& name) const {
  auto it = map_.find(name);
  if (it == map_.end()) {
    return T();
  }
  return it->second;
}

using std::to_string;

template <typename T>
std::string Polynomial<T>::toString() const {
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
      /*
      r += to_string(abs(numerator(kvp.second)));
      if (denominator(kvp.second) != 1) {
        r += "/" + to_string(denominator(kvp.second));
      }
      */
      r += to_string(abs(kvp.second));
      if (kvp.first != "") {
        r += "*";
      }
    }
    r += kvp.first;
  }
  return r;
}

template class Polynomial<Rational>;
template class Polynomial<int64_t>;

SimpleConstraint::SimpleConstraint(const Polynomial<Rational>& _poly, int64_t _rhs) : poly(_poly), rhs(_rhs) {}

RangeConstraint::RangeConstraint(const Polynomial<Rational>& _poly, int64_t _range) : poly(_poly), range(_range) {}

bool RangeConstraint::IsParallel(const RangeConstraint& c) {
  if (this->poly.tryDivide(c.poly, true) != 0) return true;
  return false;
}

SimpleConstraint RangeConstraint::lowerBound() const { return SimpleConstraint(-poly, 0); }

SimpleConstraint RangeConstraint::upperBound() const { return SimpleConstraint(poly, range - 1); }

}  // namespace math
}  // namespace tile
}  // namespace vertexai
