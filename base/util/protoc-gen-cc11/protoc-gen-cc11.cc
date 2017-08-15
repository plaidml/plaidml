#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/compiler/plugin.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>

#include <memory>
#include <string>

namespace gp = google::protobuf;
namespace gpc = google::protobuf::compiler;
namespace gpi = google::protobuf::io;

namespace vertexai {

class Cc11CodeGenerator final : public gpc::CodeGenerator {
 public:
  bool Generate(const gp::FileDescriptor* file, const std::string& parameter, gpc::GeneratorContext* generator_context,
                std::string* error) const override {
    // Determine the file name base.
    auto slash = file->name().rfind('/');
    auto dot = file->name().rfind('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
      *error = "unable to determine base name of proto file ";
      error->append(file->name());
      return false;
    }
    auto basename = file->name().substr(0, dot);

    for (int idx = 0; idx < file->message_type_count(); ++idx) {
      DoInsertions("", basename, generator_context, file->message_type(idx));
    }
    return true;
  };

 private:
  static void DoInsertions(const std::string& name_prefix, const std::string& basename,
                           gpc::GeneratorContext* generator_context, const gp::Descriptor* desc) {
    static const std::string class_scope = "class_scope:";

    std::string name = name_prefix + desc->name();

    for (int idx = 0; idx < desc->nested_type_count(); ++idx) {
      DoInsertions(name + "_", basename, generator_context, desc->nested_type(idx));
    }

    if (desc->options().map_entry()) {
      return;
    }

    std::unique_ptr<gpi::ZeroCopyOutputStream> raw{
        generator_context->OpenForInsert(basename + ".pb.h", class_scope + desc->full_name())};
    gpi::CodedOutputStream out{raw.get()};

    static const std::string header{"// C++11 rvalue move operations\n"};
    out.WriteString(header);

    // Name(Name&& rvref) noexcept
    //   : ::google::protobuf::Message(), _internal_metadata_(nullptr) {
    //   SharedCtor();
    //   rvref.Swap(this);
    // }
    out.WriteString(name);
    out.WriteRaw("(", 1);
    out.WriteString(name);

    static const std::string move_ctor_body{
        "&& rvref) noexcept\n"
        "  : ::google::protobuf::Message(), _internal_metadata_(nullptr) {\n"
        "  SharedCtor();\n"
        "  rvref.Swap(this);\n"
        "}\n"
        "\n"};
    out.WriteString(move_ctor_body);

    // Name& operator=(Name&& rvref) noexcept {
    //   Clear();
    //   rvref.Swap(this);
    //   return *this;
    // }
    out.WriteString(name);
    static const std::string opeq = "& operator=(";
    out.WriteString(opeq);
    out.WriteString(name);
    static const std::string opeq_body{
        "&& rvref) noexcept {\n"
        "  Clear();\n"
        "  rvref.Swap(this);\n"
        "  return *this;\n"
        "}\n\n"};
    out.WriteString(opeq_body);
  }
};

}  // namespace vertexai

int main(int argc, char* argv[]) {
  vertexai::Cc11CodeGenerator generator;
  return gpc::PluginMain(argc, argv, &generator);
}
