// Copyright 2018, Intel Corporation

#include "tile/codegen/emitc.h"

namespace vertexai {
namespace tile {
namespace codegen {

using boost::format;
using namespace std::placeholders;  // NOLINT
using namespace stripe;             // NOLINT

namespace {

class CodeGenerator {
 public:
  CodeGenerator() {}

  std::string EmitProgram(const Block& program) {
    EmitLine("#include <stdint.h>");
    EmitLine("#include <stdlib.h>");
    EmitLine(R"(
int min(int x, int y) {
  return (x < y) ? x : y;
}

int max(int x, int y) {
  return (x < y) ? y : x;
}

float softmax(float x) {
  return x;
}
)");
    EmitLine("int main(int argc, char** argv) {");
    PushTab();
    EmitBlock(program);
    EmitLine("return 0;");
    PopTab();
    EmitLine("}");
    return oss_.str();
  }

 private:
  void EmitLoad(const Block& block, const Load& load) {
    auto ref = block.ref_by_into(load.from);
    EmitLine(format("%1% %2% = %3%[%4%];")      //
             % IntoC(ref->interior_shape.type)  //
             % ScalarName(load.into)            //
             % UniqueName(load.from)            //
             % UniqueResolve(ref->FlatAccess()));
  }

  void EmitStore(const Block& block, const Store& store) {
    auto ref = block.ref_by_into(store.into);
    auto into = UniqueName(ref->into);
    auto access = UniqueResolve(ref->FlatAccess());
    if (ref->agg_op == Intrinsic::SUM) {
      EmitLine(format("%1%[%2%] += %3%;") % into % access % ScalarName(store.from));
    } else if (ref->agg_op == Intrinsic::PROD) {
      EmitLine(format("%1%[%2%] *= %3%;") % into % access % ScalarName(store.from));
    } else if (ref->agg_op == Intrinsic::MIN) {
      EmitLine(format("%1%[%2%] = min(%1%[%2%], %3%);") % into % access % ScalarName(store.from));
    } else if (ref->agg_op == Intrinsic::MAX) {
      EmitLine(format("%1%[%2%] = max(%1%[%2%], %3%);") % into % access % ScalarName(store.from));
    } else {
      EmitLine(format("%1%[%2%] = %3%;") % into % access % ScalarName(store.from));
    }
  }

  void EmitIntrinsic(const Block& block, const Intrinsic& intrinsic) {
    if (intrinsic.outputs.size() > 1) {
      throw std::runtime_error("Only a single output is supported for intrinsics");
    }
    auto output = ScalarName(intrinsic.outputs[0]);
    if (intrinsic.name == Intrinsic::MUL) {
      EmitLine(format("%1% %2% = %3% * %4%;")     //
               % IntoC(intrinsic.type)            //
               % output                           //
               % ScalarName(intrinsic.inputs[0])  //
               % ScalarName(intrinsic.inputs[1]));
    } else if (intrinsic.name == Intrinsic::ADD) {
      EmitLine(format("%1% %2% = %3% + %4%;")     //
               % IntoC(intrinsic.type)            //
               % output                           //
               % ScalarName(intrinsic.inputs[0])  //
               % ScalarName(intrinsic.inputs[1]));
    } else if (intrinsic.name == "bit_right") {
      EmitLine(format("%1% %2% = %3% >> %4%;")    //
               % IntoC(intrinsic.type)            //
               % output                           //
               % ScalarName(intrinsic.inputs[0])  //
               % ScalarName(intrinsic.inputs[1]));
    } else if (intrinsic.name == Intrinsic::ASSIGN) {
      EmitLine(format("%1% %2% = %3%;")  //
               % IntoC(intrinsic.type)   //
               % output                  //
               % ScalarName(intrinsic.inputs[0]));
    } else if (intrinsic.name == "zelu") {
      EmitLine(format("%1% %2% = %3% < 0 ? 0 : %3%;")  //
               % IntoC(intrinsic.type)                 //
               % output                                //
               % ScalarName(intrinsic.inputs[0]));
    } else {
      std::stringstream inputs;
      for (size_t i = 0; i < intrinsic.inputs.size(); i++) {
        if (i) {
          inputs << ", ";
        }
        inputs << ScalarName(intrinsic.inputs[i]);
      }
      EmitLine(format("%1% %2% = %3%(%4%);")  //
               % IntoC(intrinsic.type)        //
               % output                       //
               % intrinsic.name               //
               % inputs.str());
    }
  }

  void EmitConstant(const Constant& constant) {
    switch (constant.type) {
      case ConstType::Integer:
        EmitLine(format("int %1% = %2%;") % ScalarName(constant.name) % constant.iconst);
        break;
      case ConstType::Float:
        EmitLine(format("double %1% = %2%;") % ScalarName(constant.name) % constant.fconst);
        break;
    }
  }

  void EmitSpecial(const Special& special) {  //
    EmitLine(format("// TODO: %1%") % special);
  }

  void EmitBlock(const Block& block) {
    Push();

    EmitLine(format("{ // block: %1%") % block.name);
    std::stringstream ss(block.comments);
    std::string line;
    for (std::string line; std::getline(ss, line, '\n');) {
      EmitLine(format("// %1%") % line);
    }
    PushTab();

    for (const auto& idx : block.idxs) {
      auto idx_name = UniqueName(idx.name);
      if (idx.range == 1) {
        EmitLine(format("int %1% = %2%;") % idx_name % ParentResolve(idx.affine));
      } else {
        EmitLine(format("for (int %1% = 0; %1% < %2%; %1%++) {") % idx_name % idx.range);
        PushTab();
      }
    }

    for (const auto& constraint : block.constraints) {
      EmitLine(format("if ((%1%) < 0) {") % UniqueResolve(constraint));
      PushTab();
      EmitLine("continue;");
      PopTab();
      EmitLine("}");
    }

    for (const auto& ref : block.refs) {
      auto type = IntoC(ref.interior_shape.type);
      auto into = UniqueName(ref.into);
      if (ref.from.empty()) {
        EmitLine(format("%1%* %2% = malloc(%3% * sizeof(%1%));") % type % into % ref.interior_shape.elem_size());
      } else {
        EmitLine(format("%1%* %2% = %3% + %4%;") % type % into % ParentName(ref.from) %
                 UniqueResolve(ref.FlatAccess()));
      }
    }

    for (const auto& stmt : block.stmts) {
      switch (stmt->kind()) {
        case StmtKind::Load:
          EmitLoad(block, *Load::Downcast(stmt));
          break;
        case StmtKind::Store:
          EmitStore(block, *Store::Downcast(stmt));
          break;
        case StmtKind::Intrinsic:
          EmitIntrinsic(block, *Intrinsic::Downcast(stmt));
          break;
        case StmtKind::Constant:
          EmitConstant(*Constant::Downcast(stmt));
          break;
        case StmtKind::Special:
          EmitSpecial(*Special::Downcast(stmt));
          break;
        case StmtKind::Block:
          EmitBlock(*Block::Downcast(stmt));
          break;
      }
    }
    for (const auto& ref : block.refs) {
      if (ref.from.empty()) {
        EmitLine(format("free(%1%);") % UniqueName(ref.into));
      }
    }

    for (const auto& idx : block.idxs) {
      if (idx.range != 1) {
        PopTab();
        EmitLine("}");
      }
    }

    PopTab();
    EmitLine("}");

    Pop();
  }

  std::string IntoC(const DataType& type) {
    switch (type) {
      case DataType::BOOLEAN:
        return "bool";
      case DataType::INT8:
        return "int8_t";
      case DataType::INT16:
        return "int16_t";
      case DataType::INT32:
        return "int32_t";
      case DataType::INT64:
        return "int64_t";
      case DataType::INT128:
        return "int128_t";
      case DataType::UINT8:
        return "uint8_t";
      case DataType::UINT16:
        return "uint16_t";
      case DataType::UINT32:
        return "uint32_t";
      case DataType::UINT64:
        return "uint64_t";
      case DataType::FLOAT16:
        return "half";
      case DataType::FLOAT32:
        return "float";
      case DataType::FLOAT64:
        return "double";
      default:
        throw std::runtime_error("Invalid tile type");
    }
  }

  void EmitTab() { oss_ << std::string(indent_ << 1, ' '); }

  void EmitLine(const std::string& str) {
    EmitTab();
    oss_ << str << '\n';
  }

  void EmitLine(const format& fmt) {
    EmitTab();
    oss_ << fmt << '\n';
  }

  void Push() { depth_++; }
  void Pop() { depth_--; }

  void PushTab() { indent_++; }
  void PopTab() { indent_--; }

  std::string ScalarName(std::string str) {
    std::replace(str.begin(), str.end(), '$', '_');
    return UniqueName(str);
  }

  Affine Resolve(Affine affine, const std::function<std::string(const std::string&)>& resolver) {
    std::vector<std::string> names;
    for (const auto& kvp : affine.getMap()) {
      if (!kvp.first.empty()) {
        names.push_back(kvp.first);
      }
    }
    for (const auto& name : names) {
      affine.substitute(name, Affine{resolver(name)});
    }
    return affine;
  }

  std::string ParentName(const std::string& name) { return str(format("d%1%_%2%") % (depth_ - 1) % name); }
  std::string UniqueName(const std::string& name) { return str(format("d%1%_%2%") % depth_ % name); }

  Affine ParentResolve(const Affine& affine) {
    return Resolve(affine, std::bind(&CodeGenerator::ParentName, this, _1));
  }

  Affine UniqueResolve(const Affine& affine) {
    return Resolve(affine, std::bind(&CodeGenerator::UniqueName, this, _1));
  }

  std::ostringstream oss_;
  size_t indent_ = 0;
  size_t depth_ = 0;
};

}  // namespace

std::string EmitC(const Block& program) {
  CodeGenerator gen;
  return gen.EmitProgram(program);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
