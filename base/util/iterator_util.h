#pragma once

namespace vertexai {

template <typename T>
class iterate_backwards {
 public:
  explicit iterate_backwards(const T& t) : t(t) {}

  typename T::const_reverse_iterator begin() const { return t.rbegin(); }

  typename T::const_reverse_iterator end() const { return t.rend(); }

 private:
  const T& t;
};

template <typename T>
iterate_backwards<T> backwards(const T& t) {
  return iterate_backwards<T>(t);
}

}  // namespace vertexai
