#include <iostream>
#include <limits>

#include "CodeGen_Cuda.h"
#include "CodeGen_Internal.h"
#include "Substitute.h"
#include "IROperator.h"
#include "Param.h"
#include "Var.h"
#include "Lerp.h"
#include "Simplify.h"
#include "Deinterleave.h"

namespace Halide {
namespace Internal {

using std::ostream;
using std::endl;
using std::string;
using std::vector;
using std::ostringstream;
using std::map;

static
string type_to_c_type(Type type, bool include_space, bool c_plus_plus = true) {
  bool needs_space = true;
  ostringstream oss;

  if (type.is_float()) {
    if (type.bits() == 32) {
      oss << "float";
    } else if (type.bits() == 64) {
      oss << "double";
    } else {
      user_error << "Can't represent a float with this many bits in C: " << type << "\n";
    }
    if (type.is_vector()) {
      oss << type.lanes();
    }
  } else if (type.is_handle()) {
    needs_space = false;

    // If there is no type info or is generating C (not C++) and
    // the type is a class or in an inner scope, just use void *.
    if (type.handle_type == NULL ||
        (!c_plus_plus &&
         (!type.handle_type->namespaces.empty() ||
          !type.handle_type->enclosing_types.empty() ||
          type.handle_type->inner_name.cpp_type_type == halide_cplusplus_type_name::Class))) {
      oss << "void *";
    } else {
      if (type.handle_type->inner_name.cpp_type_type ==
        halide_cplusplus_type_name::Struct) {
        oss << "struct ";
      }

      if (!type.handle_type->namespaces.empty() ||
        !type.handle_type->enclosing_types.empty()) {
        oss << "::";
        for (size_t i = 0; i < type.handle_type->namespaces.size(); i++) {
          oss << type.handle_type->namespaces[i] << "::";
        }
        for (size_t i = 0; i < type.handle_type->enclosing_types.size(); i++) {
          oss << type.handle_type->enclosing_types[i].name << "::";
        }
      }
      oss << type.handle_type->inner_name.name;
      if (type.handle_type->reference_type == halide_handle_cplusplus_type::LValueReference) {
        oss << " &";
      } else if (type.handle_type->reference_type == halide_handle_cplusplus_type::LValueReference) {
        oss << " &&";
      }
      for (auto modifier : type.handle_type->cpp_type_modifiers) {
        if (modifier & halide_handle_cplusplus_type::Const) {
          oss << " const";
        }
        if (modifier & halide_handle_cplusplus_type::Volatile) {
          oss << " volatile";
        }
        if (modifier & halide_handle_cplusplus_type::Restrict) {
          oss << " restrict";
        }
        if (modifier & halide_handle_cplusplus_type::Pointer) {
          oss << " *";
        }
      }
    }
  } else {
    // This ends up using different type names than OpenCL does
    // for the integer vector types. E.g. uint16x8_t rather than
    // OpenCL's short8. Should be fine as CodeGen_C introduces
    // typedefs for them and codegen always goes through this
    // routine or its override in CodeGen_OpenCL to make the
    // names. This may be the better bet as the typedefs are less
    // likely to collide with built-in types (e.g. the OpenCL
    // ones for a C compiler that decides to compile OpenCL).
    // This code also supports arbitrary vector sizes where the
    // OpenCL ones must be one of 2, 3, 4, 8, 16, which is too
    // restrictive for already existing architectures.
    switch (type.bits()) {
    case 1:
      // bool vectors are always emitted as uint8 in the C++ backend
      if (type.is_vector()) {
          oss << "uint8x" << type.lanes() << "_t";
      } else {
          oss << "bool";
      }
      break;
    case 8: case 16: case 32: case 64:
      if (type.is_uint()) {
          oss << 'u';
      }
      oss << "int" << type.bits();
      if (type.is_vector()) {
          oss << "x" << type.lanes();
      }
      oss << "_t";
      break;
    default:
      user_error << "Can't represent an integer with this many bits in C: " << type << "\n";
    }
  }
  if (include_space && needs_space)
    oss << " ";
  return oss.str();
}

CodeGen_Cuda::CodeGen_Cuda(ostream &s, Target t):
  IRPrinter(s), target(t), id("$$ BAD ID $$"), extern_c_open(false) {}

CodeGen_Cuda::~CodeGen_Cuda() {}

void CodeGen_Cuda::open_scope() {
  cache.clear();
  do_indent();
  indent += 2;
  stream << "{\n";
}

void CodeGen_Cuda::close_scope(const std::string &comment) {
  cache.clear();
  indent -= 2;
  do_indent();
  if (!comment.empty()) {
      stream << "} // " << comment << "\n";
  } else {
      stream << "}\n";
  }
}

// TODO(wcui) create a new name for the expr and return it.
string CodeGen_Cuda::print_expr(Expr e) {
    id = "$$ BAD ID $$";
    e.accept(this);
    return id;
}

string CodeGen_Cuda::print_cast_expr(const Type &t, Expr e) {
    string value = print_expr(e);
    string type = print_type(t);
    if (t.is_vector() &&
        t.lanes() == e.type().lanes() &&
        t != e.type()) {
        return print_assignment(t, type + "::convert_from<" + print_type(e.type()) + ">(" + value + ")");
    } else {
        return print_assignment(t, "(" + type + ")(" + value + ")");
    }
}

void CodeGen_Cuda::print_stmt(Stmt s) {
  s.accept(this);
}

string CodeGen_Cuda::print_type(Type type, AppendSpaceIfNeeded space_option) {
  return type_to_c_type(type, space_option == AppendSpace);
}

string CodeGen_Cuda::print_reinterpret(Type type, Expr e) {
    ostringstream oss;
    if (type.is_handle() || e.type().is_handle()) {
        // Use a c-style cast if either src or dest is a handle --
        // note that although Halide declares a "Handle" to always be 64 bits,
        // the source "handle" might actually be a 32-bit pointer (from
        // a function parameter), so calling reinterpret<> (which just memcpy's)
        // would be garbage-producing.
        oss << "(" << print_type(type) << ")";
    } else {
        oss << "reinterpret<" << print_type(type) << ">";
    }
    oss << "(" << print_expr(e) << ")";
    return oss.str();
}

string CodeGen_Cuda::print_name(const string &name) {
    ostringstream oss;

    // Prefix an underscore to avoid reserved words (e.g. a variable named "while")
    if (isalpha(name[0])) {
        oss << '_';
    }

    for (size_t i = 0; i < name.size(); i++) {
        if (name[i] == '.') {
            oss << '_';
        } else if (name[i] == '$') {
            oss << "__";
        } else if (name[i] != '_' && !isalnum(name[i])) {
            oss << "___";
        }
        else oss << name[i];
    }
    return oss.str();
}

string CodeGen_Cuda::print_scalarized_expr(Expr e) {
  Type t = e.type();
  internal_assert(t.is_vector());
  string v = unique_name('_');
  do_indent();
  stream << print_type(t, AppendSpace) << v << ";\n";
  for (int lane = 0; lane < t.lanes(); lane++) {
    Expr e2 = extract_lane(e, lane);
    string elem = print_expr(e2);
    ostringstream rhs;
    rhs << v << ".replace(" << lane << ", " << elem << ")";
    v = print_assignment(t, rhs.str());
  }
  return v;
}

string CodeGen_Cuda::print_extern_call(const Call *op) {
  if (op->type.is_vector()) {
    // Need to split into multiple scalar calls.
    return print_scalarized_expr(op);
  }
  ostringstream rhs;
  vector<string> args(op->args.size());
  for (size_t i = 0; i < op->args.size(); i++) {
    args[i] = print_expr(op->args[i]);
    // This substitution ensures const correctness for all calls
    if (args[i] == "__user_context") {
      args[i] = "_ucon";
    }
  }
  if (function_takes_user_context(op->name)) {
    args.insert(args.begin(), "_ucon");
  }
  rhs << op->name << "(" << with_commas(args) << ")";
  return rhs.str();
}

string CodeGen_Cuda::print_assignment(Type t, const std::string &rhs) {
  auto cached = cache.find(rhs);
  if (cached == cache.end()) {
    id = unique_name('_');
    do_indent();
    stream << print_type(t, AppendSpace) << id << " = " << rhs << ";\n";
    cache[rhs] = id;
  } else {
    id = cached->second;
  }
  return id;
}

void CodeGen_Cuda::set_name_mangling_mode(NameMangling mode) {
  if (extern_c_open && mode != NameMangling::C) {
    stream << "\n#ifdef __cplusplus\n";
    stream << "}  // extern \"C\"\n";
    stream << "#endif\n\n";
    extern_c_open = false;
  } else if (!extern_c_open && mode == NameMangling::C) {
    stream << "\n#ifdef __cplusplus\n";
    stream << "extern \"C\" {\n";
    stream << "#endif\n\n";
    extern_c_open = true;
  }
}

void CodeGen_Cuda::forward_declare_type_if_needed(const Type &t) {
  if (!t.handle_type ||
    forward_declared.count(t.handle_type) ||
    t.handle_type->inner_name.cpp_type_type == halide_cplusplus_type_name::Simple) {
    return;
  }
  for (auto &ns : t.handle_type->namespaces) {
    stream << "namespace " << ns << " { ";
  }
  switch (t.handle_type->inner_name.cpp_type_type) {
  case halide_cplusplus_type_name::Simple:
    // nothing
    break;
  case halide_cplusplus_type_name::Struct:
    stream << "struct " << t.handle_type->inner_name.name << ";";
    break;
  case halide_cplusplus_type_name::Class:
    stream << "class " << t.handle_type->inner_name.name << ";";
    break;
  case halide_cplusplus_type_name::Union:
    stream << "union " << t.handle_type->inner_name.name << ";";
    break;
  case halide_cplusplus_type_name::Enum:
    internal_error << "Passing pointers to enums is unsupported\n";
    break;
  }
  for (auto &ns : t.handle_type->namespaces) {
    (void) ns;
    stream << " }";
  }
  stream << "\n";
  forward_declared.insert(t.handle_type);
}

void CodeGen_Cuda::compile(const Module &input) {
  // Forward-declare all the types we need; this needs to happen before
  // we emit function prototypes, since those may need the types.
  stream << "\n";
  for (const auto &f : input.functions()) {
      // debug(0) << "forwrd declare type for " << f.name << "\n";
      for (auto &arg : f.args) {
          forward_declare_type_if_needed(arg.type);
      }
  }
  stream << "\n";

  for (const auto &b : input.buffers()) {
    compile(b);
  }
  for (const auto &f : input.functions()) {
    compile(f);
  }
}

void CodeGen_Cuda::compile(const LoweredFunc &f) {
  const std::vector<LoweredArgument> &args = f.args;

  have_user_context = false;
  for (size_t i = 0; i < args.size(); i++) {
    // TODO: check that its type is void *?
    have_user_context |= (args[i].name == "__user_context");
  }

  NameMangling name_mangling = f.name_mangling;
  if (name_mangling == NameMangling::Default) {
    name_mangling = (target.has_feature(Target::CPlusPlusMangling) ?
                     NameMangling::CPlusPlus : NameMangling::C);
  }

  set_name_mangling_mode(name_mangling);

  std::vector<std::string> namespaces;
  std::string simple_name = extract_namespaces(f.name, namespaces);

  if (!namespaces.empty()) {
    for (const auto &ns : namespaces) {
      stream << "namespace " << ns << " {\n";
    }
    stream << "\n";
  }

  // Emit the function prototype
  if (f.linkage == LoweredFunc::Internal) {
    // If the function isn't public, mark it static.
    stream << "static ";
  }
  stream << "int " << simple_name << "(";
  for (size_t i = 0; i < args.size(); i++) {
    if (args[i].is_buffer()) {
      stream << "struct halide_buffer_t *"
             << print_name(args[i].name)
             << "_buffer";
    } else {
      stream << print_type(args[i].type, AppendSpace)
             << print_name(args[i].name);
    }

    if (i < args.size()-1) stream << ", ";
  }

  stream << ") HALIDE_FUNCTION_ATTRS \n";
  open_scope();

  do_indent();
  stream << "void * const _ucon = "
         << (have_user_context ? "const_cast<void *>(__user_context)" : "nullptr")
         << ";\n";

  // Emit the body
  print(f.body);

  // Return success.
  do_indent();
  stream << "return 0;\n";

  close_scope(simple_name);
  if (!namespaces.empty()) {
    stream << "\n";
    for (size_t i = namespaces.size(); i > 0; i--) {
      stream << "}  // namespace " << namespaces[i-1] << "\n";
    }
    stream << "\n";
  }

  set_name_mangling_mode(NameMangling::Default);
}

void CodeGen_Cuda::compile(const Buffer<> &buffer) {
  stream << "#error embeded images not supported";
}

// Halide asserts have different semantics to C asserts.  They're
// supposed to clean up and make the containing function return
// -1, so we can't use the C version of assert. Instead we convert
// to an if statement.
void CodeGen_Cuda::create_assertion(const string &id_cond, const string &id_msg) {
  if (target.has_feature(Target::NoAsserts)) return;

  do_indent();
  stream << "if (!" << id_cond << ")\n";
  open_scope();
  do_indent();
  stream << "return " << id_msg << ";\n";
  close_scope("");
}

void CodeGen_Cuda::create_assertion(const string &id_cond, Expr message) {
  internal_assert(!message.defined() || message.type() == Int(32))
      << "Assertion result is not an int: " << message;

  if (target.has_feature(Target::NoAsserts)) return;

  // don't call the create_assertion(string, string) version because
  // we don't want to force evaluation of 'message' unless the condition fails
  do_indent();
  stream << "if (!" << id_cond << ") ";
  open_scope();
  string id_msg = print_expr(message);
  do_indent();
  stream << "return " << id_msg << ";\n";
  close_scope("");
}

void CodeGen_Cuda::create_assertion(Expr cond, Expr message) {
  create_assertion(print_expr(cond), message);
}

void CodeGen_Cuda::visit(const Variable *op) {
  id = print_name(op->name);
}

void CodeGen_Cuda::visit(const Cast *op) {
  id = print_cast_expr(op->type, op->value);
}

void CodeGen_Cuda::visit_binop(Type t, Expr a, Expr b, const char * op) {
  string sa = print_expr(a);
  string sb = print_expr(b);
  print_assignment(t, sa + " " + op + " " + sb);
}

void CodeGen_Cuda::visit(const Add *op) {
  visit_binop(op->type, op->a, op->b, "+");
}

void CodeGen_Cuda::visit(const Sub *op) {
  visit_binop(op->type, op->a, op->b, "-");
}

void CodeGen_Cuda::visit(const Mul *op) {
  visit_binop(op->type, op->a, op->b, "*");
}

void CodeGen_Cuda::visit(const Div *op) {
  int bits;
  if (is_const_power_of_two_integer(op->b, &bits)) {
    visit_binop(op->type, op->a, make_const(op->a.type(), bits), ">>");
  } else if (op->type.is_int()) {
    print_expr(lower_euclidean_div(op->a, op->b));
  } else {
    visit_binop(op->type, op->a, op->b, "/");
  }
}

void CodeGen_Cuda::visit(const Mod *op) {
  int bits;
  if (is_const_power_of_two_integer(op->b, &bits)) {
    visit_binop(op->type, op->a, make_const(op->a.type(), (1 << bits)-1), "&");
  } else if (op->type.is_int()) {
    print_expr(lower_euclidean_mod(op->a, op->b));
  } else {
    visit_binop(op->type, op->a, op->b, "%");
  }
}

void CodeGen_Cuda::visit(const Max *op) {
  // clang doesn't support the ternary operator on OpenCL style vectors.
  // See: https://bugs.llvm.org/show_bug.cgi?id=33103
  if (op->type.is_scalar()) {
    print_expr(Call::make(op->type, "::halide_cpp_max", {op->a, op->b}, Call::Extern));
  } else {
    ostringstream rhs;
    rhs << print_type(op->type) << "::max(" << print_expr(op->a) << ", " << print_expr(op->b) << ")";
    print_assignment(op->type, rhs.str());
  }
}

void CodeGen_Cuda::visit(const Min *op) {
  // clang doesn't support the ternary operator on OpenCL style vectors.
  // See: https://bugs.llvm.org/show_bug.cgi?id=33103
  if (op->type.is_scalar()) {
    print_expr(Call::make(op->type, "::halide_cpp_min", {op->a, op->b}, Call::Extern));
  } else {
    ostringstream rhs;
    rhs << print_type(op->type) << "::min(" << print_expr(op->a) << ", " << print_expr(op->b) << ")";
    print_assignment(op->type, rhs.str());
  }
}

void CodeGen_Cuda::visit(const EQ *op) {
  visit_binop(op->type, op->a, op->b, "==");
}

void CodeGen_Cuda::visit(const NE *op) {
  visit_binop(op->type, op->a, op->b, "!=");
}

void CodeGen_Cuda::visit(const LT *op) {
  visit_binop(op->type, op->a, op->b, "<");
}

void CodeGen_Cuda::visit(const LE *op) {
  visit_binop(op->type, op->a, op->b, "<=");
}

void CodeGen_Cuda::visit(const GT *op) {
  visit_binop(op->type, op->a, op->b, ">");
}

void CodeGen_Cuda::visit(const GE *op) {
  visit_binop(op->type, op->a, op->b, ">=");
}

void CodeGen_Cuda::visit(const Or *op) {
  visit_binop(op->type, op->a, op->b, "||");
}

void CodeGen_Cuda::visit(const And *op) {
  visit_binop(op->type, op->a, op->b, "&&");
}

void CodeGen_Cuda::visit(const Not *op) {
  print_assignment(op->type, "!(" + print_expr(op->a) + ")");
}

void CodeGen_Cuda::visit(const IntImm *op) {
  if (op->type == Int(32)) {
    id = std::to_string(op->value);
  } else {
    print_assignment(op->type, "(" + print_type(op->type) + ")(" + std::to_string(op->value) + ")");
  }
}

void CodeGen_Cuda::visit(const UIntImm *op) {
  print_assignment(op->type, "(" + print_type(op->type) + ")(" + std::to_string(op->value) + ")");
}

void CodeGen_Cuda::visit(const StringImm *op) {
  ostringstream oss;
  oss << Expr(op);
  id = oss.str();
}

void CodeGen_Cuda::visit(const Call *op) {
  internal_assert(op->is_extern() || op->is_intrinsic())
      << "Can only codegen extern calls and intrinsics\n";

  ostringstream rhs;

  // Handle intrinsics first
  if (op->is_intrinsic(Call::debug_to_file)) {
    internal_assert(op->args.size() == 3);
    const StringImm *string_imm = op->args[0].as<StringImm>();
    internal_assert(string_imm);
    string filename = string_imm->value;
    string typecode = print_expr(op->args[1]);
    string buffer = print_name(print_expr(op->args[2]));

    rhs << "halide_debug_to_file(_ucon, "
        << "\"" << filename << "\", "
        << typecode
        << ", (struct halide_buffer_t *)" << buffer << ")";
  } else if (op->is_intrinsic(Call::bitwise_and)) {
    internal_assert(op->args.size() == 2);
    string a0 = print_expr(op->args[0]);
    string a1 = print_expr(op->args[1]);
    rhs << a0 << " & " << a1;
  } else if (op->is_intrinsic(Call::bitwise_xor)) {
    internal_assert(op->args.size() == 2);
    string a0 = print_expr(op->args[0]);
    string a1 = print_expr(op->args[1]);
    rhs << a0 << " ^ " << a1;
  } else if (op->is_intrinsic(Call::bitwise_or)) {
    internal_assert(op->args.size() == 2);
    string a0 = print_expr(op->args[0]);
    string a1 = print_expr(op->args[1]);
    rhs << a0 << " | " << a1;
  } else if (op->is_intrinsic(Call::bitwise_not)) {
    internal_assert(op->args.size() == 1);
    rhs << "~" << print_expr(op->args[0]);
  } else if (op->is_intrinsic(Call::reinterpret)) {
    internal_assert(op->args.size() == 1);
    rhs << print_reinterpret(op->type, op->args[0]);
  } else if (op->is_intrinsic(Call::shift_left)) {
    internal_assert(op->args.size() == 2);
    string a0 = print_expr(op->args[0]);
    string a1 = print_expr(op->args[1]);
    rhs << a0 << " << " << a1;
  } else if (op->is_intrinsic(Call::shift_right)) {
    internal_assert(op->args.size() == 2);
    string a0 = print_expr(op->args[0]);
    string a1 = print_expr(op->args[1]);
    rhs << a0 << " >> " << a1;
  } else if (op->is_intrinsic(Call::lerp)) {
    internal_assert(op->args.size() == 3);
    Expr e = lower_lerp(op->args[0], op->args[1], op->args[2]);
    rhs << print_expr(e);
  } else if (op->is_intrinsic(Call::absd)) {
    internal_assert(op->args.size() == 2);
    Expr a = op->args[0];
    Expr b = op->args[1];
    Type t = op->type.with_code(op->type.is_int() ? Type::UInt : op->type.code());
    Expr e = cast(t, select(a < b, b - a, a - b));
    rhs << print_expr(e);
  } else if (op->is_intrinsic(Call::return_second)) {
    internal_assert(op->args.size() == 2);
    string arg0 = print_expr(op->args[0]);
    string arg1 = print_expr(op->args[1]);
    rhs << "return_second(" << arg0 << ", " << arg1 << ")";
  } else if (op->is_intrinsic(Call::if_then_else)) {
    internal_assert(op->args.size() == 3);

    string result_id = unique_name('_');

    do_indent();
    stream << print_type(op->args[1].type(), AppendSpace)
           << result_id << ";\n";

    string cond_id = print_expr(op->args[0]);

    do_indent();
    stream << "if (" << cond_id << ")\n";
    open_scope();
    string true_case = print_expr(op->args[1]);
    do_indent();
    stream << result_id << " = " << true_case << ";\n";
    close_scope("if " + cond_id);
    do_indent();
    stream << "else\n";
    open_scope();
    string false_case = print_expr(op->args[2]);
    do_indent();
    stream << result_id << " = " << false_case << ";\n";
    close_scope("if " + cond_id + " else");

    rhs << result_id;
  } else if (op->is_intrinsic(Call::require)) {
  internal_assert(op->args.size() == 3);
  if (op->args[0].type().is_vector()) {
    rhs << print_scalarized_expr(op);
  } else {
    create_assertion(op->args[0], op->args[2]);
    rhs << print_expr(op->args[1]);
  }
  } else if (op->is_intrinsic(Call::abs)) {
    internal_assert(op->args.size() == 1);
    Expr a0 = op->args[0];
    rhs << print_expr(cast(op->type, select(a0 > 0, a0, -a0)));
  } else if (op->is_intrinsic(Call::memoize_expr)) {
    // TODO(wcui): what's the purpose of memoize_expr?
    internal_assert(op->args.size() >= 1);
    string arg = print_expr(op->args[0]);
    rhs << "(" << arg << ")";
  } else if (op->is_intrinsic(Call::alloca)) {
    internal_assert(op->args.size() == 1);
    internal_assert(op->type.is_handle());
    const Call *call = op->args[0].as<Call>();
    if (op->type == type_of<struct halide_buffer_t *>() &&
      call && call->is_intrinsic(Call::size_of_halide_buffer_t)) {
      do_indent();
      string buf_name = unique_name('b');
      stream << "halide_buffer_t " << buf_name << ";\n";
      rhs << "&" << buf_name;
    } else {
      // Make a stack of uint64_ts
      string size = print_expr(simplify((op->args[0] + 7)/8));
      do_indent();
      string array_name = unique_name('a');
      stream << "uint64_t " << array_name << "[" << size << "];";
      rhs << "(" << print_type(op->type) << ")(&" << array_name << ")";
    }
  } else if (op->is_intrinsic(Call::make_struct)) {
    if (op->args.empty()) {
      internal_assert(op->type.handle_type);
      // Add explicit cast so that different structs can't cache to the same value
      rhs << "(" << print_type(op->type) << ")(NULL)";
    } else {
      // Emit a declaration like:
      // struct {const int f_0, const char f_1, const int f_2} foo = {3, 'c', 4};

      // Get the args
      vector<string> values;
      for (size_t i = 0; i < op->args.size(); i++) {
        values.push_back(print_expr(op->args[i]));
      }
      do_indent();
      stream << "struct {\n";
      // List the types.
      indent++;
      for (size_t i = 0; i < op->args.size(); i++) {
        do_indent();
        stream << "const " << print_type(op->args[i].type()) << " f_" << i << ";\n";
      }
      indent--;
      string struct_name = unique_name('s');
      do_indent();
      stream << "} " << struct_name << " = {\n";
      // List the values.
      indent++;
      for (size_t i = 0; i < op->args.size(); i++) {
        do_indent();
        stream << values[i];
        if (i < op->args.size() - 1) stream << ",";
        stream << "\n";
      }
      indent--;
      do_indent();
      stream << "};\n";
      // Return a pointer to it of the appropriate type
      if (op->type.handle_type) {
        rhs << "(" << print_type(op->type) << ")";
      }
      rhs << "(&" << struct_name << ")";
    }
  } else if (op->is_intrinsic(Call::stringify)) {
    // Rewrite to an snprintf
    vector<string> printf_args;
    string format_string = "";
    for (size_t i = 0; i < op->args.size(); i++) {
      Type t = op->args[i].type();
      printf_args.push_back(print_expr(op->args[i]));
      if (t.is_int()) {
        format_string += "%lld";
        printf_args[i] = "(long long)(" + printf_args[i] + ")";
      } else if (t.is_uint()) {
        format_string += "%llu";
        printf_args[i] = "(long long unsigned)(" + printf_args[i] + ")";
      } else if (t.is_float()) {
        if (t.bits() == 32) {
          format_string += "%f";
        } else {
          format_string += "%e";
        }
      } else if (op->args[i].as<StringImm>()) {
        format_string += "%s";
      } else {
        internal_assert(t.is_handle());
        format_string += "%p";
      }
    }
    string buf_name = unique_name('b');
    do_indent();
    stream << "char " << buf_name << "[1024];\n";
    do_indent();
    stream << "snprintf(" << buf_name << ", 1024, \"" << format_string << "\", " << with_commas(printf_args) << ");\n";
    rhs << buf_name;
  } else if (op->is_intrinsic(Call::register_destructor)) {
    internal_assert(op->args.size() == 2);
    const StringImm *fn = op->args[0].as<StringImm>();
    internal_assert(fn);
    string arg = print_expr(op->args[1]);

    do_indent();
    // Make a struct on the stack that calls the given function as a destructor
    string struct_name = unique_name('s');
    string instance_name = unique_name('d');
    stream << "struct " << struct_name << " { "
           << "void * const ucon; "
           << "void * const arg; "
           << "" << struct_name << "(void *ucon, void *a) : ucon(ucon), arg((void *)a) {} "
           << "~" << struct_name << "() { " << fn->value + "(ucon, arg); } "
           << "} " << instance_name << "(_ucon, " << arg << ");\n";
    rhs << print_expr(0);
  } else if (op->is_intrinsic(Call::div_round_to_zero)) {
    rhs << print_expr(op->args[0]) << " / " << print_expr(op->args[1]);
  } else if (op->is_intrinsic(Call::mod_round_to_zero)) {
    rhs << print_expr(op->args[0]) << " % " << print_expr(op->args[1]);
  } else if (op->is_intrinsic(Call::signed_integer_overflow)) {
    user_error << "Signed integer overflow occurred during constant-folding. Signed"
        " integer overflow for int32 and int64 is undefined behavior in"
        " Halide.\n";
  } else if (op->is_intrinsic(Call::prefetch)) {
    // TODO(wcui): Use ptx prefetch for CUDA
    user_assert((op->args.size() == 4) && is_one(op->args[2]))
        << "Only prefetch of 1 cache line is supported in C backend.\n";
    const Variable *base = op->args[0].as<Variable>();
    internal_assert(base && base->type.is_handle());
    rhs << "__builtin_prefetch("
        << "((" << print_type(op->type) << " *)" << print_name(base->name)
        << " + " << print_expr(op->args[1]) << "), 1)";
  } else if (op->is_intrinsic(Call::indeterminate_expression)) {
    user_error << "Indeterminate expression occurred during constant-folding.\n";
  } else if (op->is_intrinsic(Call::size_of_halide_buffer_t)) {
    rhs << "(sizeof(halide_buffer_t))";
  } else if (op->is_intrinsic()) {
    // TODO: other intrinsics
    internal_error << "Unhandled intrinsic in C backend: " << op->name << '\n';
  } else {
    // Generic extern calls
    rhs << print_extern_call(op);
  }

  print_assignment(op->type, rhs.str());
}

// NaN is the only float/double for which this is true... and
// surprisingly, there doesn't seem to be a portable isnan function
// (dsharlet).
template <typename T>
static bool isnan(T x) { return x != x; }

template <typename T>
static bool isinf(T x)
{
  return std::numeric_limits<T>::has_infinity && (
      x == std::numeric_limits<T>::infinity() ||
      x == -std::numeric_limits<T>::infinity());
}

void CodeGen_Cuda::visit(const FloatImm *op) {
  if (isnan(op->value)) {
    id = "nan_f32()";
  } else if (isinf(op->value)) {
    if (op->value > 0) {
        id = "inf_f32()";
    } else {
        id = "neg_inf_f32()";
    }
  } else {
    // Write the constant as reinterpreted uint to avoid any bits lost in conversion.
    union {
      uint32_t as_uint;
      float as_float;
    } u;
    u.as_float = op->value;

    ostringstream oss;
    if (op->type.bits() == 64) {
      oss << "(double) ";
    }
    oss << "float_from_bits(" << u.as_uint << " /* " << u.as_float << " */)";
    print_assignment(op->type, oss.str());
  }
}

void CodeGen_Cuda::visit(const Let *op) {
  string id_value = print_expr(op->value);
  Expr body = op->body;
  if (op->value.type().is_handle()) {
    // The body might contain a Load that references this directly
    // by name, so we can't rewrite the name.
    do_indent();
    stream << print_type(op->value.type())
           << " " << print_name(op->name)
           << " = " << id_value << ";\n";
  } else {
    Expr new_var = Variable::make(op->value.type(), id_value);
    body = substitute(op->name, new_var, body);
  }
  print_expr(body);
}

void CodeGen_Cuda::visit(const LetStmt *op) {
  string id_value = print_expr(op->value);
  Stmt body = op->body;
  if (op->value.type().is_handle()) {
    // The body might contain a Load or Store that references this
    // directly by name, so we can't rewrite the name.
    do_indent();
    stream << print_type(op->value.type())
           << " " << print_name(op->name)
           << " = " << id_value << ";\n";
  } else {
    Expr new_var = Variable::make(op->value.type(), id_value);
    body = substitute(op->name, new_var, body);
  }
  body.accept(this);
}

void CodeGen_Cuda::visit(const Select *op) {
  ostringstream rhs;
  string type = print_type(op->type);
  string true_val = print_expr(op->true_value);
  string false_val = print_expr(op->false_value);
  string cond = print_expr(op->condition);

  // clang doesn't support the ternary operator on OpenCL style vectors.
  // See: https://bugs.llvm.org/show_bug.cgi?id=33103
  if (op->condition.type().is_scalar()) {
    rhs << "(" << type << ")"
        << "(" << cond
        << " ? " << true_val
        << " : " << false_val
        << ")";
  } else {
    rhs << type << "::select(" << cond << ", " << true_val << ", " << false_val << ")";
  }
  print_assignment(op->type, rhs.str());
}

void CodeGen_Cuda::visit(const AssertStmt *op) {
  create_assertion(op->condition, op->message);
}

void CodeGen_Cuda::visit(const ProducerConsumer *op) {
  do_indent();
  if (op->is_producer) {
    stream << "// produce " << op->name << '\n';
  } else {
    stream << "// consume " << op->name << '\n';
  }
  print_stmt(op->body);
}

void CodeGen_Cuda::visit(const Ramp *op) {
  Type vector_type = op->type.with_lanes(op->lanes);
  string id_base = print_expr(op->base);
  string id_stride = print_expr(op->stride);
  print_assignment(vector_type, print_type(vector_type) + "::ramp(" + id_base + ", " + id_stride + ")");
}

void CodeGen_Cuda::visit(const Broadcast *op) {
  Type vector_type = op->type.with_lanes(op->lanes);
  string id_value = print_expr(op->value);
  string rhs;
  if (op->lanes > 1) {
    rhs = print_type(vector_type) + "::broadcast(" + id_value + ")";
  } else {
    rhs = id_value;
  }

  print_assignment(vector_type, rhs);
}

void CodeGen_Cuda::visit(const Provide *op) {
  internal_error << "Cannot emit Provide statements as Cuda\n";
}

void CodeGen_Cuda::visit(const Realize *op) {
    internal_error << "Cannot emit realize statements to Cuda\n";
}

void CodeGen_Cuda::visit(const Prefetch *op) {
    internal_error << "Cannot emit prefetch statements to Cuda\n";
}

void CodeGen_Cuda::visit(const IfThenElse *op) {
  string cond_id = print_expr(op->condition);

  do_indent();
  stream << "if (" << cond_id << ")\n";
  open_scope();
  op->then_case.accept(this);
  close_scope("if " + cond_id);

  if (op->else_case.defined()) {
    do_indent();
    stream << "else\n";
    open_scope();
    op->else_case.accept(this);
    close_scope("if " + cond_id + " else");
  }
}

void CodeGen_Cuda::visit(const Evaluate *op) {
  if (is_const(op->value)) return;
  string id = print_expr(op->value);
  do_indent();
  stream << "(void)" << id << ";\n";
}

void CodeGen_Cuda::visit(const Shuffle *op) {
  internal_assert(op->vectors.size() >= 1);
  internal_assert(op->vectors[0].type().is_vector());
  for (size_t i = 1; i < op->vectors.size(); i++) {
    internal_assert(op->vectors[0].type() == op->vectors[i].type());
  }
  internal_assert(op->type.lanes() == (int) op->indices.size());
  const int max_index = (int) (op->vectors[0].type().lanes() * op->vectors.size());
  for (int i : op->indices) {
    internal_assert(i >= -1 && i < max_index);
  }

  std::vector<string> vecs;
  for (Expr v : op->vectors) {
    vecs.push_back(print_expr(v));
  }
  string src = vecs[0];
  if (op->vectors.size() > 1) {
    ostringstream rhs;
    string storage_name = unique_name('_');
    do_indent();
    stream << "const " << print_type(op->vectors[0].type()) << " " << storage_name << "[] = { " << with_commas(vecs) << " };\n";

    rhs << print_type(op->type) << "::concat(" << op->vectors.size() << ", " << storage_name << ")";
    src = print_assignment(op->type, rhs.str());
  }
  ostringstream rhs;
  if (op->type.is_scalar()) {
    rhs << src << "[" << op->indices[0] << "]";
  } else {
    string indices_name = unique_name('_');
    do_indent();
    stream << "const int32_t " << indices_name << "[" << op->indices.size() << "] = { " << with_commas(op->indices) << " };\n";
    rhs << print_type(op->type) << "::shuffle(" << src << ", " << indices_name << ")";
  }
  print_assignment(op->type, rhs.str());
}

void CodeGen_Cuda::visit(const Load *op) {
  user_assert(is_one(op->predicate)) << "Predicated load is not supported by Cuda backend.\n";

  // TODO: We could replicate the logic in the llvm codegen which decides whether
  // the vector access can be aligned. Doing so would also require introducing
  // aligned type equivalents for all the vector types.
  ostringstream rhs;

  Type t = op->type;
  string name = print_name(op->name);

  // If we're loading a contiguous ramp into a vector, just load the vector
  Expr dense_ramp_base = strided_ramp_base(op->index, 1);
  if (dense_ramp_base.defined()) {
    internal_assert(t.is_vector());
    string id_ramp_base = print_expr(dense_ramp_base);
    rhs << print_type(t) + "::load(" << name << ", " << id_ramp_base << ")";
  } else if (op->index.type().is_vector()) {
    // If index is a vector, gather vector elements.
    internal_assert(t.is_vector());
    string id_index = print_expr(op->index);
    rhs << print_type(t) + "::load(" << name << ", " << id_index << ")";
  } else {
    string id_index = print_expr(op->index);
    bool type_cast_needed = !(allocations.contains(op->name) &&
                            allocations.get(op->name).type.element_of() == t.element_of());
    if (type_cast_needed) {
      rhs << "((const " << print_type(t.element_of()) << " *)" << name << ")";
    } else {
      rhs << name;
    }
    rhs << "[" << id_index << "]";
  }
  print_assignment(t, rhs.str());
}

void CodeGen_Cuda::visit(const Store *op) {
  user_assert(is_one(op->predicate)) << "Predicated store is not supported by Cuda backend.\n";

  Type t = op->value.type();
  string id_value = print_expr(op->value);
  string name = print_name(op->name);

  // TODO: We could replicate the logic in the llvm codegen which decides whether
  // the vector access can be aligned. Doing so would also require introducing
  // aligned type equivalents for all the vector types.

  // If we're writing a contiguous ramp, just store the vector.
  Expr dense_ramp_base = strided_ramp_base(op->index, 1);
  if (dense_ramp_base.defined()) {
    internal_assert(op->value.type().is_vector());
    string id_ramp_base = print_expr(dense_ramp_base);
    do_indent();
    stream << id_value + ".store(" << name << ", " << id_ramp_base << ");\n";
  } else if (op->index.type().is_vector()) {
    // If index is a vector, scatter vector elements.
    internal_assert(t.is_vector());
    string id_index = print_expr(op->index);
    do_indent();
    stream << id_value + ".store(" << name << ", " << id_index << ");\n";
  } else {
    bool type_cast_needed =
        t.is_handle() ||
        !allocations.contains(op->name) ||
        allocations.get(op->name).type != t;

    string id_index = print_expr(op->index);
    do_indent();
    if (type_cast_needed) {
        stream << "((" << print_type(t) << " *)" << name << ")";
    } else {
        stream << name;
    }
    stream << "[" << id_index << "] = " << id_value << ";\n";
    }
    cache.clear();
}

void CodeGen_Cuda::visit(const Allocate *op) {
  open_scope();

  string op_name = print_name(op->name);
  string op_type = print_type(op->type, AppendSpace);

  // For sizes less than 8k, do a stack allocation
  bool on_stack = false;
  int32_t constant_size;
  string size_id;
  if (op->new_expr.defined()) {
    Allocation alloc;
    alloc.type = op->type;
    allocations.push(op->name, alloc);
    heap_allocations.push(op->name);
    stream << op_type << "*" << op_name << " = (" << print_expr(op->new_expr) << ");\n";
  } else {
    constant_size = op->constant_allocation_size();
    if (constant_size > 0) {
      int64_t stack_bytes = constant_size * op->type.bytes();

      if (stack_bytes > ((int64_t(1) << 31) - 1)) {
        user_error << "Total size for allocation "
                   << op->name << " is constant but exceeds 2^31 - 1.\n";
      } else {
        size_id = print_expr(Expr(static_cast<int32_t>(constant_size)));
        if (can_allocation_fit_on_stack(stack_bytes)) {
          on_stack = true;
        }
      }
    } else {
      // Check that the allocation is not scalar (if it were scalar
      // it would have constant size).
      internal_assert(op->extents.size() > 0);

      size_id = print_assignment(Int(64), print_expr(op->extents[0]));

      for (size_t i = 1; i < op->extents.size(); i++) {
        // Make the code a little less cluttered for two-dimensional case
        string new_size_id_rhs;
        string next_extent = print_expr(op->extents[i]);
        if (i > 1) {
          new_size_id_rhs =  "(" + size_id + " > ((int64_t(1) << 31) - 1)) ? " + size_id + " : (" + size_id + " * " + next_extent + ")";
        } else {
          new_size_id_rhs = size_id + " * " + next_extent;
        }
        size_id = print_assignment(Int(64), new_size_id_rhs);
      }
      do_indent();
      stream << "if ((" << size_id << " > ((int64_t(1) << 31) - 1)) || ((" << size_id <<
        " * sizeof(" << op_type << ")) > ((int64_t(1) << 31) - 1)))\n";
      open_scope();
      do_indent();
      // TODO: call halide_error_buffer_allocation_too_large() here instead
      // TODO: call create_assertion() so that NoAssertions works
      stream << "halide_error(_ucon, "
             << "\"32-bit signed overflow computing size of allocation " << op->name << "\\n\");\n";
      do_indent();
      stream << "return -1;\n";
      close_scope("overflow test " + op->name);
    }

    // Check the condition to see if this allocation should actually be created.
    // If the allocation is on the stack, the only condition we can respect is
    // unconditional false (otherwise a non-constant-sized array declaration
    // will be generated).
    if (!on_stack || is_zero(op->condition)) {
      Expr conditional_size = Select::make(op->condition,
                                           Var(size_id),
                                           Expr(static_cast<int32_t>(0)));
      conditional_size = simplify(conditional_size);
      size_id = print_assignment(Int(64), print_expr(conditional_size));
    }

    Allocation alloc;
    alloc.type = op->type;
    allocations.push(op->name, alloc);

    do_indent();
    stream << op_type;

    if (on_stack) {
      stream << op_name
             << "[" << size_id << "];\n";
    } else {
      stream << "*"
             << op_name
             << " = ("
             << op_type
             << " *)halide_malloc(_ucon, sizeof("
             << op_type
             << ")*" << size_id << ");\n";
      heap_allocations.push(op->name);
    }
  }

  if (!on_stack) {
    create_assertion(op_name, "halide_error_out_of_memory(_ucon)");

    do_indent();
    string free_function = op->free_function.empty() ? "halide_free" : op->free_function;
    stream << "HalideFreeHelper " << op_name << "_free(_ucon, "
           << op_name << ", " << free_function << ");\n";
  }

  op->body.accept(this);

  // Should have been freed internally
  internal_assert(!allocations.contains(op->name));

  close_scope("alloc " + print_name(op->name));
}

void CodeGen_Cuda::visit(const Free *op) {
  if (heap_allocations.contains(op->name)) {
    do_indent();
    stream << print_name(op->name) << "_free.free();\n";
    heap_allocations.pop(op->name);
  }
  allocations.pop(op->name);
}

void CodeGen_Cuda::visit(const For *op) {
  string id_min = print_expr(op->min);
  string id_extent = print_expr(op->extent);

  if (op->for_type == ForType::Parallel) {
    do_indent();
    stream << "#pragma omp parallel for\n";
  } else {
    internal_assert(op->for_type == ForType::Serial)
        << "Can only emit serial or parallel for loops to C\n";
  }

  do_indent();
  stream << "for (int "
         << print_name(op->name)
         << " = " << id_min
         << "; "
         << print_name(op->name)
         << " < " << id_min
         << " + " << id_extent
         << "; "
         << print_name(op->name)
         << "++)\n";

  open_scope();
  op->body.accept(this);
  close_scope("for " + print_name(op->name));

}


}  // namespace Internal
}  // namespace Halide
