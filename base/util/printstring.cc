#include "base/util/printstring.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <stdexcept>

std::string printstring(const char* format, ...) {
  va_list vl;
  va_list vl2;
  va_start(vl, format);
  va_copy(vl2, vl);

  /* Get Length */
  char safety[1];
  int len = vsnprintf(safety, 0, format, vl);
  if (len < 0) {
    va_end(vl);
    va_end(vl2);
    throw std::runtime_error("vasprintf failure.");
  }
  /* +1 for \0 terminator. */
  char* buf = reinterpret_cast<char*>(malloc(len + 1));
  len = vsnprintf(buf, len + 1, format, vl2);
  if (len < 0) {
    va_end(vl);
    va_end(vl2);
    throw std::runtime_error("vasprintf failure.");
  }
  std::string r(buf, len);
  free(buf);
  va_end(vl);
  va_end(vl2);

  return r;
}
