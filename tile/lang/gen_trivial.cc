#include "tile/lang/gen_trivial.h"

#include <memory>
#include <string>

#include "base/util/logging.h"
#include "tile/lang/sembuilder.h"

namespace vertexai {
namespace tile {
namespace lang {

KernelInfo GenCopy(const TensorShape& shape, const std::string& oname, const std::string& iname,
                   const std::string& kname) {
  using namespace sem::builder;  // NOLINT
  uint64_t size = shape.elem_size();
  IVLOG(2, "Making a copy for " << iname.c_str() << " -> " << oname.c_str() << ", of size " << size);
  sem::Function::params_t params;
  sem::Type oparamtype{sem::Type::POINTER_MUT, shape.type};
  sem::Type iparamtype{sem::Type::POINTER_CONST, shape.type};
  iparamtype.region = sem::Type::GLOBAL;
  oparamtype.region = sem::Type::GLOBAL;
  params.push_back(std::make_pair(oparamtype, "out"));
  params.push_back(std::make_pair(iparamtype, "in"));
  sem::Type voidret{sem::Type::TVOID};
  sem::StmtPtr body =
      _Block({_("out")[_Index(sem::IndexExpr::GLOBAL, 0)] = _("in")[_Index(sem::IndexExpr::GLOBAL, 0)]});
  auto func = std::make_shared<sem::Function>(kname, voidret, params, body);

  KernelInfo ki;
  ki.kname = kname;
  ki.kfunc = func;
  ki.outputs.push_back(oname);
  ki.inputs.push_back(iname);
  ki.gwork[0] = size;
  ki.gwork[1] = 1;
  ki.gwork[2] = 1;
  ki.lwork[0] = ki.lwork[1] = ki.lwork[2] = 0;
  ki.tot_bytes = size * ((bit_width(shape.type) + 7) / 8);
  ki.tot_flops = size;
  auto pb = ki.info.mutable_zero();
  pb->set_copy(true);
  ki.ktype = KernelType::kCopy;
  return ki;
}

KernelInfo GenZero(const TensorShape& shape, const std::string& bname, const std::string& kname) {
  using namespace sem::builder;  // NOLINT
  uint64_t size = shape.elem_size();
  IVLOG(2, "Making a zero for " << bname.c_str() << ", of size " << size);
  sem::Function::params_t params;
  sem::Type paramtype{sem::Type::POINTER_MUT, shape.type};
  paramtype.region = sem::Type::GLOBAL;
  params.push_back(std::make_pair(paramtype, "out"));
  sem::Type voidret{sem::Type::TVOID};
  sem::StmtPtr body = _Block({_("out")[_Index(sem::IndexExpr::GLOBAL, 0)] = _Const(0)});
  auto func = std::make_shared<sem::Function>(kname, voidret, params, body);

  KernelInfo ki;
  ki.kname = kname;
  ki.kfunc = func;
  ki.outputs.push_back(bname);
  ki.gwork[0] = size;
  ki.gwork[1] = 1;
  ki.gwork[2] = 1;
  ki.lwork[0] = ki.lwork[1] = ki.lwork[2] = 0;
  ki.tot_bytes = size * ((bit_width(shape.type) + 7) / 8);
  ki.tot_flops = size;
  auto pb = ki.info.mutable_zero();
  pb->set_copy(false);
  ki.ktype = KernelType::kZero;
  return ki;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
