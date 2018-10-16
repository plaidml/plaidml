// Copyright 2018, Intel Corp.

#pragma once

#include <iostream>
#include <string>
#include <utility>

namespace vertexai {

template <typename T>
struct StreamContainerContext {
  const T& container;
  bool multiline;
  bool outer;
  size_t indent;
};

template <typename T>
StreamContainerContext<T> StreamContainer(const T& container,      //
                                          bool multiline = false,  //
                                          bool outer = true,       //
                                          size_t indent = 0) {
  return StreamContainerContext<T>{container, multiline, outer, indent};
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const StreamContainerContext<T>& holder) {
  std::string indent(2 * holder.indent, ' ');
  if (holder.multiline) {
    os << indent;
    if (holder.outer) {
      os << "{";
    }
    os << "\n";
    for (const auto& item : holder.container) {
      os << indent << "  " << item << ",\n";
    }
    os << indent;
    if (holder.outer) {
      os << "}";
    }
    os << "\n";
  } else {
    size_t size = holder.container.size();
    size_t cur = 0;
    os << indent;
    if (holder.outer) {
      os << "{";
    }
    for (const auto& item : holder.container) {
      os << item;
      if (cur++ != size - 1) {
        os << ", ";
      }
    }
    if (holder.outer) {
      os << "}";
    }
  }
  return os;
}

}  // namespace vertexai

namespace std {

// Overload operator<< for pairs to make maps stream nicely
// TODO: Should we 'wrap' pairs/element similarly to containers to avoid
// overloading in the general case?
template <typename A, typename B>
std::ostream& operator<<(std::ostream& os, const pair<A, B>& value) {
  os << "(" << value.first << "->" << value.second << ")";
  return os;
}

}  // namespace std
