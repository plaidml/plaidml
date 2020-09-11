#include "plaidml/bridge/tensorflow/tests/utils.h"

#include <fstream>

#include "plaidml/bridge/tensorflow/tests/archive_generated.h"
#include "plaidml/bridge/tensorflow/tests/codegen_test.h"

using plaidml::edsl::MultiBuffer;
namespace zoo = plaidml::zoo;

namespace xla {
namespace plaidml {
MultiBuffer convertBuffer(const zoo::DataUnion& data) {
  switch (data.type) {
    case zoo::Data_I8Data:
      return data.AsI8Data()->data;
    case zoo::Data_I16Data:
      return data.AsI16Data()->data;
    case zoo::Data_I32Data:
      return data.AsI32Data()->data;
    case zoo::Data_I64Data:
      return data.AsI64Data()->data;
    case zoo::Data_U8Data:
      return data.AsU8Data()->data;
    case zoo::Data_U16Data:
      return data.AsU16Data()->data;
    case zoo::Data_U32Data:
      return data.AsU32Data()->data;
    case zoo::Data_U64Data:
      return data.AsU64Data()->data;
    case zoo::Data_F16Data:
      return data.AsF16Data()->data;
    case zoo::Data_F32Data:
      return data.AsF32Data()->data;
    case zoo::Data_F64Data:
      return data.AsF64Data()->data;
    default:
      break;
  }
  throw std::runtime_error("Invalid data_type");
}

std::vector<char> ReadFile(const std::string& path) {
  std::ifstream fs;
  fs.open(path, std::ios::binary | std::ios::in);
  fs.seekg(0, std::ios::end);
  int length = fs.tellg();
  fs.seekg(0, std::ios::beg);
  std::vector<char> buf(length);
  fs.read(buf.data(), buf.size());
  fs.close();
  return buf;
}
}  // namespace plaidml
}  // namespace xla
