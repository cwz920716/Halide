#ifndef HALIDE_CODEGEN_CUDA_H
#define HALIDE_CODEGEN_CUDA_H

/** \file
 *
 * Defines an IRPrinter that emits CUDA code equivalent to a halide stmt
 */

#include "IRPrinter.h"
#include "Module.h"
#include "Scope.h"

namespace Halide {

struct Argument;

namespace Internal {

/** This class emits CUDA code equivalent to a halide Stmt.
 */
class CodeGen_Cuda : public IRPrinter {
 public:
  EXPORT CodeGen_Cuda(std::ostream &dest,
                      Target target);
  EXPORT ~CodeGen_Cuda();

  /** Emit the declarations contained in the module as CUDA code. */
  EXPORT void compile(const Module &input);

  /** The target we're generating code for */
  const Target &get_target() const { return target; }

 protected:
  enum AppendSpaceIfNeeded {
    DoNotAppendSpace,
    AppendSpace,
  };

  /** Open a new C scope (i.e. throw in a brace, increase the indent) */
  void open_scope();

  /** Close a C scope (i.e. throw in an end brace, decrease the indent) */
  void close_scope(const std::string &comment);

  /** Emit an expression as an assignment, then return the id of the
   * resulting var */
  std::string print_expr(Expr);

  /** Like print_expr, but cast the Expr to the given Type */
  std::string print_cast_expr(const Type &, Expr);

  /** Emit a statement */
  void print_stmt(Stmt);

  /** Emit the C name for a halide type. If space_option is AppendSpace,
   *  and there should be a space between the type and the next token,
   *  one is appended. (This allows both "int foo" and "Foo *foo" to be
   *  formatted correctly. Otherwise the latter is "Foo * foo".)
   */
  virtual std::string print_type(Type, AppendSpaceIfNeeded space_option = DoNotAppendSpace);

  /** Emit a statement to reinterpret an expression as another type */
  virtual std::string print_reinterpret(Type, Expr);

  /** Emit a version of a string that is a valid identifier in C (. is replaced with _) */
  virtual std::string print_name(const std::string &);

  /** Bottleneck to allow customization of calls to generic Extern/PureExtern calls.  */
  virtual std::string print_extern_call(const Call *op);

  /** Convert a vector Expr into a series of scalar Exprs, then reassemble into vector of original type.  */
  std::string print_scalarized_expr(Expr e);

  /** Emit an SSA-style assignment, and set id to the freshly generated name. Return id. */
  std::string print_assignment(Type t, const std::string &rhs);

  /** Emit a declaration. */
  // @{
  virtual void compile(const LoweredFunc &func);
  virtual void compile(const Buffer<> &buffer);
  // @}

 private:

  /** If the Type is a handle type, emit a forward-declaration for it
   * if we haven't already. */
  void forward_declare_type_if_needed(const Type &t);

  void set_name_mangling_mode(NameMangling mode);

  template<typename T>
  static std::string with_sep(const std::vector<T> &v, const std::string &sep) {
    std::ostringstream o;
    for (size_t i = 0; i < v.size(); ++i) {
      if (i > 0) {
        o << sep;
      }
      o << v[i];
    }
    return o.str();
  }

  template<typename T>
  static std::string with_commas(const std::vector<T> &v) {
    return with_sep<T>(v, ", ");
  }

  /** The target being generated for. */
  Target target;

  /** Track which handle types have been forward-declared already. */
  std::set<const halide_handle_cplusplus_type *> forward_declared;

  /** An ID for the most recently generated ssa variable */
  std::string id;

  /** True if there is a void * __user_context parameter in the arguments. */
  bool have_user_context;

  /** Track current calling convention scope. */
  bool extern_c_open;

  /** A cache of generated values in scope */
  std::map<std::string, std::string> cache;
};

}  // namespace Internal

}  // namespace Halide

#endif  // HALIDE_CODEGEN_CUDA_H
