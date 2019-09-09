// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/travel.h"
#include "tile/hal/cm/emitcm.h"

#include "tile/lang/exprtype.h"
#include "tile/lang/sembuilder.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

void TravelVisitor::Visit(const sem::IntConst& node) {
  switch (travel) {
    case GET_STRING:
      node_str << std::to_string(node.value);
      break;
    default:
      return;
  }
}

void TravelVisitor::Visit(const sem::FloatConst& node) {
  switch (travel) {
    case GET_STRING:
      node_str << std::to_string(node.value);
      break;
    default:
      return;
  }
}

void TravelVisitor::Visit(const sem::LookupLVal& node) {
  switch (travel) {
    case GET_STRING:
      node_str << node.name;
      break;
    case CHECK_CM_VECTOR:
      if (vector_params.find(node.name) != vector_params.end()) {
        is_cm_vector = true;
      }
      break;
    case GET_INDEX_STRIDE:
      index_stride = index_stride_map[node.name];
      break;
    case GET_GLOBAL_VAR_WITH_OFFSET:
      if (global_params.find(node.name) != global_params.end()) {
        global_var_with_offset << node.name;
      }
      break;
    default:
      return;
  }
}

void TravelVisitor::Visit(const sem::SubscriptLVal& node) {
  switch (travel) {
    case GET_STRING:
      node_str << "(";
      if (node.ptr) node.ptr->Accept(*this);
      node_str << " [";
      if (node.offset) node.offset->Accept(*this);
      node_str << "])";
      break;
    case CHECK_CM_VECTOR:
      if (node.ptr) node.ptr->Accept(*this);
      break;
    case GET_INDEX_STRIDE:
      if (node.offset) node.offset->Accept(*this);
      break;
    case GET_GLOBAL_VAR_WITH_OFFSET: {
      if (node.ptr) node.ptr->Accept(*this);

      auto s = GetGlobalVarWithOffset();
      if (s.size() > 0) {
        InitNodeStr();
        node.offset->Accept(*this);
        auto node_str = GetNodeStr();
        travel = GET_GLOBAL_VAR_WITH_OFFSET;

        global_var_with_offset << " " << node_str;
      }
    } break;
    default:
      if (node.ptr) node.ptr->Accept(*this);
      if (node.offset) node.offset->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::LoadExpr& node) {
  switch (travel) {
    case GET_GLOBAL_LOAD_EXPRS: {
      InitGlobalVarWithOffset();
      if (node.inner) node.inner->Accept(*this);
      auto s = GetGlobalVarWithOffset();
      travel = GET_GLOBAL_LOAD_EXPRS;

      if (s.length() > 0) {
        global_load_exprs[std::make_shared<sem::LoadExpr>(node)] = s;
      }
    } break;
    default:
      if (node.inner) node.inner->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::StoreStmt& node) {
  switch (travel) {
    case GET_STRING:
      if (node.lhs) node.lhs->Accept(*this);
      node_str << " = ";
      if (node.rhs) node.rhs->Accept(*this);
      break;
    default:
      if (node.lhs) node.lhs->Accept(*this);
      if (node.rhs) node.rhs->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::DeclareStmt& node) {
  switch (travel) {
    case GET_STRING:
      node_str << node.name;
      node_str << " = ";
      if (node.init) node.init->Accept(*this);
      break;
    default:
      if (node.init) node.init->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::UnaryExpr& node) {
  switch (travel) {
    case GET_STRING:
      node_str << node.op;
      node_str << " : ";
      if (node.inner) node.inner->Accept(*this);
      break;
    default:
      if (node.inner) node.inner->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::BinaryExpr& node) {
  switch (travel) {
    case GET_STRING:
      node_str << "(";
      if (node.lhs) node.lhs->Accept(*this);
      node_str << " " << node.op << " ";
      if (node.rhs) node.rhs->Accept(*this);
      node_str << ")";
      break;
    case GET_INDEX_STRIDE: {
      if (!node.op.compare("/")) {
        auto index_expr = std::dynamic_pointer_cast<sem::IndexExpr>(node.lhs);
        auto int_const = std::dynamic_pointer_cast<sem::IntConst>(node.rhs);
        if (index_expr && index_expr->type == sem::IndexExpr::LOCAL && int_const && int_const->value == 1) {
          index_stride = 1;
          return;
        }
      }
      if (!node.op.compare("*")) {
        auto int_const = std::dynamic_pointer_cast<sem::IntConst>(node.lhs);
        auto load_expr = std::dynamic_pointer_cast<sem::LoadExpr>(node.rhs);
        if (int_const && load_expr) {
          index_stride = 0;
          load_expr->Accept(*this);
          index_stride = int_const->value * index_stride;
          return;
        }
      }
      index_stride = 0;
      node.lhs->Accept(*this);
      auto l = index_stride;
      index_stride = 0;
      node.rhs->Accept(*this);
      auto r = index_stride;
      index_stride = std::max(l, r);
    } break;
    default:
      if (node.lhs) node.lhs->Accept(*this);
      if (node.rhs) node.rhs->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::CondExpr& node) {
  switch (travel) {
    case GET_STRING:
      node_str << "(";
      if (node.cond) node.cond->Accept(*this);
      node_str << " ";
      if (node.tcase) node.tcase->Accept(*this);
      node_str << " ";
      if (node.fcase) node.fcase->Accept(*this);
      node_str << ")";
      break;
    default:
      if (node.cond) node.cond->Accept(*this);
      if (node.tcase) node.tcase->Accept(*this);
      if (node.fcase) node.fcase->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::SelectExpr& node) {
  switch (travel) {
    case GET_STRING:
      node_str << "(";
      if (node.cond) node.cond->Accept(*this);
      node_str << " ";
      if (node.tcase) node.tcase->Accept(*this);
      node_str << " ";
      if (node.fcase) node.fcase->Accept(*this);
      node_str << ")";
      break;
    default:
      if (node.cond) node.cond->Accept(*this);
      if (node.tcase) node.tcase->Accept(*this);
      if (node.fcase) node.fcase->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::ClampExpr& node) {
  switch (travel) {
    case GET_STRING:
      node_str << "(";
      if (node.val) node.val->Accept(*this);
      node_str << " ";
      if (node.min) node.min->Accept(*this);
      node_str << " ";
      if (node.max) node.max->Accept(*this);
      node_str << ")";
      break;
    case CHECK_CM_VECTOR:
      is_cm_vector = false;
      break;
    default:
      if (node.val) node.val->Accept(*this);
      if (node.min) node.min->Accept(*this);
      if (node.max) node.max->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::CastExpr& node) {
  switch (travel) {
    default:
      if (node.val) node.val->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::CallExpr& node) {
  switch (travel) {
    case GET_STRING:
      node_str << node.name << "(";
      for (size_t i = 0; i < node.vals.size(); i++) {
        if (node.vals[i]) node.vals[i]->Accept(*this);
        if (i < node.vals.size() - 1) {
          node_str << ", ";
        }
      }
      node_str << ")";
      break;
    default:
      for (size_t i = 0; i < node.vals.size(); i++) {
        if (node.vals[i]) node.vals[i]->Accept(*this);
      }
      return;
  }
}

void TravelVisitor::Visit(const sem::LimitConst& node) {
  switch (travel) {
    case GET_STRING: {
      if (node.which == sem::LimitConst::ZERO) {
        node_str << "0";
      } else if (node.which == sem::LimitConst::ONE) {
        node_str << "1";
      }
      auto it = LimitConstLookup.find(std::make_pair(node.type, node.which));
      if (it == LimitConstLookup.end()) {
        throw std::runtime_error("Invalid type in LimitConst");
      }
      node_str << (it->second);
    } break;
    default:
      return;
  }
}

void TravelVisitor::Visit(const sem::IndexExpr& node) {
  switch (travel) {
    case GET_STRING: {
      switch (node.type) {
        case sem::IndexExpr::GLOBAL:
          node_str << "global " << std::to_string(node.dim);
          break;
        case sem::IndexExpr::GROUP:
          node_str << "group " << std::to_string(node.dim);
          break;
        case sem::IndexExpr::LOCAL:
          node_str << "local" << std::to_string(node.dim);
          break;
        default:
          node_str << "other index";
      }
    } break;
    default:
      return;
  }
}

void TravelVisitor::Visit(const sem::Block& node) {
  switch (travel) {
    case GET_STRING:
      for (const sem::StmtPtr& ptr : node.statements) {
        if (ptr) ptr->Accept(*this);
        node_str << " ";
      }
      break;
    default:
      for (const sem::StmtPtr& ptr : node.statements) {
        if (ptr) ptr->Accept(*this);
      }
      return;
  }
}

void TravelVisitor::Visit(const sem::IfStmt& node) {
  switch (travel) {
    case GET_STRING:
      node_str << "if(";
      if (node.cond) node.cond->Accept(*this);
      node_str << "; ";
      if (node.iftrue) node.iftrue->Accept(*this);
      node_str << "; ";
      if (node.iffalse) node.iffalse->Accept(*this);
      node_str << ")";
      break;
    default:
      if (node.cond) node.cond->Accept(*this);
      if (node.iftrue) node.iftrue->Accept(*this);
      if (node.iffalse) node.iffalse->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::ForStmt& node) {
  switch (travel) {
    case GET_STRING:
      node_str << "for(";
      if (node.inner) node.inner->Accept(*this);
      node_str << ")";
      break;
    default:
      if (node.inner) node.inner->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::WhileStmt& node) {
  switch (travel) {
    case GET_STRING:
      node_str << "while(";
      if (node.cond) node.cond->Accept(*this);
      node_str << "; ";
      if (node.inner) node.inner->Accept(*this);
      node_str << ")";
      break;
    default:
      if (node.cond) node.cond->Accept(*this);
      if (node.inner) node.inner->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::BarrierStmt& node) {
  switch (travel) {
    case GET_STRING:
      node_str << "barrier";
      break;
    default:
      return;
  }
}

void TravelVisitor::Visit(const sem::ReturnStmt& node) {
  switch (travel) {
    case GET_STRING:
      node_str << "return ";
      if (node.value) node.value->Accept(*this);
      break;
    default:
      if (node.value) node.value->Accept(*this);
      return;
  }
}

void TravelVisitor::Visit(const sem::SpecialStmt& node) {
  switch (travel) {
    case GET_STRING:
      node_str << "special(";
      for (size_t i = 0; i < node.params.size(); i++) {
        if (node.params[i]) node.params[i]->Accept(*this);
        node_str << " ";
      }
      node_str << ")";
      break;
    default:
      for (size_t i = 0; i < node.params.size(); i++) {
        if (node.params[i]) node.params[i]->Accept(*this);
      }
      return;
  }
}

void TravelVisitor::Visit(const sem::Function& node) {
  switch (travel) {
    case GET_STRING:
      node_str << "function ";
      if (node.body) node.body->Accept(*this);
      break;
    default:
      if (node.body) node.body->Accept(*this);
      return;
  }
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
