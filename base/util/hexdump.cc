#include "base/util/hexdump.h"

#include <iomanip>
#include <sstream>

#include "base/util/logging.h"

namespace vertexai {

void hexdump(int log_level, void* buf, size_t len) {
  const size_t LINE_SIZE = 16;

  char* line = static_cast<char*>(buf);
  size_t offset = 0;
  size_t remain = len;
  size_t lines = (len / LINE_SIZE) + 1;

  for (size_t j = 0; j < lines; j++) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');

    size_t line_remain = std::min(LINE_SIZE, remain);
    ss << std::setw(8) << offset;

    for (size_t i = 0; i < LINE_SIZE; i++) {
      if (i % 8 == 0) {
        ss << ' ';
      }
      if (i < line_remain) {
        ss << ' ' << std::setw(2) << std::hex << (static_cast<int>(line[i]) & 0xff);
      } else {
        ss << "   ";
      }
    }

    ss << "  ";
    for (size_t i = 0; i < line_remain; i++) {
      if (std::isprint(line[i])) {
        ss << line[i];
      } else {
        ss << '.';
      }
    }
    VLOG(log_level) << ss.str();

    line += LINE_SIZE;
    offset += LINE_SIZE;
    remain -= LINE_SIZE;
  }
}

}  // namespace vertexai
