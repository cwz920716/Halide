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

}  // namespace Internal
}  // namespace Halide
