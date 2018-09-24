// Copyright 2018, Intel Corp.

#pragma once

#include <iostream>
#include <string>

namespace vertexai {

template <typename T>
struct StreamContainerHolder {
  const T& container;
  bool multiline;
  size_t indent;
  StreamContainerHolder(const T& _container, bool _multiline, size_t _indent)
      : container(_container), multiline(_multiline), indent(_indent) {}
};

template <typename T>
StreamContainerHolder<T> StreamContainer(const T& container, bool multiline = false, size_t indent = 0) {
  return StreamContainerHolder<T>(container, multiline, indent);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const StreamContainerHolder<T>& x) {
  std::string istr(2 * x.indent, ' ');
  if (x.multiline) {
    os << istr << "{\n";
    for (const auto& v : x.container) {
      os << istr << "  " << v << ",\n";
    }
    os << istr << "}\n";
  } else {
    size_t size = x.container.size();
    size_t cur = 0;
    os << istr << "{";
    for (const auto& v : x.container) {
      os << " " << v;
      if (cur++ != size - 1) {
        os << ",";
      }
    }
    os << " }";
  }
  return os;
}

}  // namespace vertexai
