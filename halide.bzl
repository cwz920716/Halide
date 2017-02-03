def halide_language_copts():
  # TODO: this is wrong for (e.g.) Windows and will need further specialization.
  return [
      "$(STACK_FRAME_UNLIMITED)", "-fno-exceptions", "-fno-rtti", "-fPIC",
      "-fvisibility-inlines-hidden", "-std=c++11", "-DGOOGLE_PROTOBUF_NO_RTTI"
  ]


def halide_language_linkopts():
  _linux_opts = ["-rdynamic", "-ldl", "-lpthread", "-lz"]
  _osx_opts = ["-Wl,-stack_size", "-Wl,1000000"]
  return select({
      "@halide//:halide_host_config_darwin":
          _osx_opts,
      "@halide//:halide_host_config_darwin_x86_64":
          _osx_opts,
      # TODO: this is wrong for (e.g.) Windows and will need further specialization.
      "//conditions:default":
          _linux_opts,
  })


def halide_runtime_linkopts():
  # TODO: this is wrong for (e.g.) Windows and will need further specialization.
  return [
      "-ldl",
      "-lpthread",
  ]


def halide_opengl_linkopts():
  _linux_opts = ["-lGL", "-lX11"]
  _osx_opts = ["-framework OpenGL"]
  return select({
      "@halide//:halide_platform_config_darwin":
          _osx_opts,
      "@halide//:halide_platform_config_darwin_x86_64":
          _osx_opts,
      # TODO: this is wrong for (e.g.) Windows and will need further specialization.
      "//conditions:default":
          _linux_opts,
  })


# (halide-target-base, cpu, android-cpu, ios-cpu)
_HALIDE_TARGET_CONFIG_INFO = [
    # Android
    ("arm-32-android", None, "armeabi-v7a", None),
    ("arm-64-android", None, "arm64-v8a", None),
    ("x86-32-android", None, "x86", None),
    ("x86-64-android", None, "x86_64", None),
    # iOS
    ("arm-32-ios", None, None, "armv7"),
    ("arm-64-ios", None, None, "arm64"),
    # OSX
    ("x86-32-osx", None, None, "i386"),
    ("x86-64-osx", None, None, "x86_64"),
    # Linux
    ("arm-64-linux", "arm", None, None),
    ("powerpc-64-linux", "ppc", None, None),
    ("x86-64-linux", "k8", None, None),
    ("x86-32-linux", "piii", None, None),
    # Special case: Android-ARMEABI. Note that we are using an illegal Target
    # string for Halide; this is intentional. It allows us to add another
    # config_setting to match the armeabi-without-v7a required for certain build
    # scenarios; we special-case this in _select_multitarget to translate it
    # back into a legal Halide target.
    #
    # Note that this won't produce a build that is useful (it will SIGILL on
    # non-v7a hardware), but isn't intended to be useful for anything other
    # than allowing certain builds to complete.
    ("armeabi-32-android", "armeabi", "armeabi", None),
]
# TODO: add conditions appropriate for other targets/cpus: Windows, etc.

_HALIDE_TARGET_MAP_DEFAULT = {
    "x86-64-osx": [
        "x86-64-osx-sse41-avx-avx2-fma",
        "x86-64-osx-sse41-avx",
        "x86-64-osx-sse41",
        "x86-64-osx",
    ],
    "x86-64-linux": [
        "x86-64-linux-sse41-avx-avx2-fma",
        "x86-64-linux-sse41-avx",
        "x86-64-linux-sse41",
        "x86-64-linux",
    ],
    "x86-32-linux": [
        "x86-32-linux-sse41",
        "x86-32-linux",
    ],
}


def halide_library_default_target_map():
  return _HALIDE_TARGET_MAP_DEFAULT


_HALIDE_RUNTIME_OVERRIDES = {
    # Empty placeholder for now; we may add target-specific
    # overrides here in the future.
}


def _halide_host_config_settings():
  _host_cpus = [
      "darwin",
      "darwin_x86_64",
  ]
  for host_cpu in _host_cpus:
    native.config_setting(
        name="halide_host_config_%s" % host_cpu,
        values={"host_cpu": host_cpu},
        visibility=["//visibility:public"])
    # TODO hokey, improve, this isn't really right in general
    native.config_setting(
        name="halide_platform_config_%s" % host_cpu,
        values={
            # "crosstool_top": "//tools/osx/crosstool",
            "cpu": host_cpu,
        },
        visibility=["//visibility:public"])


def halide_config_settings():
  """Define config_settings for halide_library.

       These settings are used to distinguish build targets for
       halide_library() based on target CPU and configs. This is provided
       to allow algorithmic generation of the config_settings based on
       internal data structures; it should not be used outside of Halide.

  """
  _halide_host_config_settings()
  for base_target, cpu, android_cpu, ios_cpu in _HALIDE_TARGET_CONFIG_INFO:
    if android_cpu == None:
      # "armeabi" is the default value for --android_cpu and isn't considered legal
      # here, so we use the value to assume we aren't building for Android.
      android_cpu = "armeabi"
    if ios_cpu == None:
      # The default value for --ios_cpu is "x86_64", i.e. for the 64b OS X simulator.
      # Assuming that the i386 version of the simulator will be used along side
      # arm32 apps, we consider this value to mean the flag was unspecified; this
      # won't work for 32 bit simulator builds for A6 or older phones.
      ios_cpu = "x86_64"
    if cpu != None:
      values = {
          "cpu": cpu,
          "android_cpu": android_cpu,
          "ios_cpu": ios_cpu,
      }
    else:
      values = {
          "android_cpu": android_cpu,
          "ios_cpu": ios_cpu,
      }
    native.config_setting(
        name=_config_setting_name(base_target),
        values=values,
        visibility=["//visibility:public"])

  # Config settings for Sanitizers (currently, only MSAN)
  native.config_setting(
      name="halide_config_msan",
      values={"compiler": "msan"},
      visibility=["//visibility:public"])


# Alphabetizes the features part of the target to make sure they always match no
# matter the concatenation order of the target string pieces.
def _canonicalize_target(halide_target):
  if halide_target == "host":
    return halide_target
  if "," in halide_target:
    fail("Multitarget may not be specified here")
  tokens = halide_target.split("-")
  if len(tokens) < 3:
    fail("Illegal target: %s" % halide_target)
  # rejoin the tokens with the features sorted
  return "-".join(tokens[0:3] + sorted(tokens[3:]))


# Converts comma and dash separators to underscore and alphabetizes
# the features part of the target to make sure they always match no
# matter the concatenation order of the target string pieces.
def _halide_target_to_bazel_rule_name(multitarget):
  subtargets = multitarget.split(",")
  subtargets = [_canonicalize_target(st).replace("-", "_") for st in subtargets]
  return "_".join(subtargets)


def _config_setting_name(halide_target):
  """Take a Halide target string and converts to a unique name suitable for
   a Bazel config_setting."""
  if "," in halide_target:
    fail("Multitarget may not be specified here: %s" % halide_target)
  tokens = halide_target.split("-")
  if len(tokens) != 3:
    fail("Unexpected halide_target form: %s" % halide_target)
  halide_arch = tokens[0]
  halide_bits = tokens[1]
  halide_os = tokens[2]
  return "halide_config_%s_%s_%s" % (halide_arch, halide_bits, halide_os)


def _config_setting(halide_target):
  return "@halide//:%s" % _config_setting_name(halide_target)


_output_extensions = {
    "static_library": ("a", False),
    "o": ("o", False),
    "h": ("h", False),
    "cpp_stub": ("stub.h", False),
    "assembly": ("s.txt", True),
    "bitcode": ("bc", True),
    "stmt": ("stmt", True),
    "html": ("html", True),
    "cpp": ("cpp", True),
}


def _gengen_outputs(filename, halide_target, outputs):
  new_outputs = {}
  for o in outputs:
    if o not in _output_extensions:
      fail("Unknown output: " + o)
    ext, is_multiple = _output_extensions[o]
    if is_multiple and len(halide_target) > 1:
      # Special handling needed for ".s.txt" and similar: the suffix from the
      # is_multiple case always goes before the final .
      # (i.e. "filename.s_suffix.txt", not "filename_suffix.s.txt")
      # -- this is awkward, but is what Halide does, so we must match it.
      pieces = ext.rsplit(".", 1)
      extra = (".%s" % pieces[0]) if len(pieces) > 1 else ""
      ext = pieces[-1]
      for h in halide_target:
        new_outputs[o + h] = "%s%s_%s.%s" % (
            filename, extra, _canonicalize_target(h).replace("-", "_"), ext)
    else:
      new_outputs[o] = "%s.%s" % (filename, ext)
  return new_outputs


def _gengen_impl(ctx):
  if _has_dupes(ctx.attr.outputs):
    fail("Duplicate values in outputs: " + str(ctx.attr.outputs))

  if not ctx.attr.generator_closure.generator_name:
    fail("generator_name must be specified")

  remaps = [".s=.s.txt"]
  halide_target = ctx.attr.halide_target
  if ctx.attr.sanitizer:
    halide_target = []
    for t in ctx.attr.halide_target:
      ct = _canonicalize_target("%s-%s" % (t, ctx.attr.sanitizer))
      halide_target += [ct]
      remaps += ["%s=%s" % (ct.replace("-", "_"), t.replace("-", "_"))]

  outputs = [
      ctx.new_file(f)
      for f in _gengen_outputs(
          ctx.attr.filename,
          ctx.attr.halide_target,  # *not* halide_target
          ctx.attr.outputs).values()
  ]

  arguments = ["-o", outputs[0].dirname]
  if ctx.attr.generate_runtime:
    arguments += ["-r", ctx.attr.filename]
    if len(halide_target) > 1:
      fail("Only one halide_target allowed here")
    if ctx.attr.halide_function_name:
      fail("halide_function_name not allowed here")
  else:
    arguments += ["-g", ctx.attr.generator_closure.generator_name]
    if ctx.attr.filename:
      arguments += ["-n", ctx.attr.filename]
    if ctx.attr.halide_function_name:
      arguments += ["-f", ctx.attr.halide_function_name]

  if ctx.attr.outputs:
    arguments += ["-e", ",".join(ctx.attr.outputs)]
    arguments += ["-x", ",".join(remaps)]
  arguments += ["target=%s" % ",".join(halide_target)]
  if ctx.attr.halide_generator_args:
    arguments += ctx.attr.halide_generator_args.split(" ")

  env = {
      "HL_DEBUG_CODEGEN": str(ctx.attr.debug_codegen_level),
      "HL_TRACE": str(ctx.attr.trace_level),
  }
  ctx.action(
      # If you need to force the tools to run locally (e.g. for experimentation),
      # uncomment this line.
      # execution_requirements={"local":"1"},
      arguments=arguments,
      env=env,
      # TODO: files_to_run is undocumented but (apparently) reliable
      executable=ctx.attr.generator_closure.generator_binary.files_to_run.
      executable,
      mnemonic="ExecuteHalideGenerator",
      outputs=outputs,
      progress_message="Executing generator %s with target (%s) args (%s)..." %
      (ctx.attr.generator_closure.generator_name,
       ",".join(halide_target),
       ctx.attr.halide_generator_args))


_gengen = rule(
    implementation=_gengen_impl,
    attrs={
        "debug_codegen_level":
            attr.int(),
        "filename":
            attr.string(),
        "generate_runtime":
            attr.bool(default=False),
        "generator_closure":
            attr.label(
                cfg="host", providers=["generator_binary", "generator_name"]),
        "halide_target":
            attr.string_list(),
        "halide_function_name":
            attr.string(),
        "halide_generator_args":
            attr.string(),
        "outputs":
            attr.string_list(),
        "sanitizer":
            attr.string(),
        "trace_level":
            attr.int(),
    },
    outputs=_gengen_outputs,
    output_to_genfiles=True)


def _add_target_features(target, features):
  if "," in target:
    fail("Cannot use multitarget here")
  new_target = target.split("-")
  for f in features:
    if f and f not in new_target:
      new_target += [f]
  return "-".join(new_target)


def _has_dupes(some_list):
  clean = list(set(some_list))
  return sorted(some_list) != sorted(clean)


def _select_multitarget(base_target,
                        halide_target_features,
                        halide_target_map):
  if base_target == "armeabi-32-android":
    base_target = "arm-32-android"
  wildcard_target = halide_target_map.get("*")
  if wildcard_target:
    expected_base = "*"
    targets = wildcard_target
  else:
    expected_base = base_target
    targets = halide_target_map.get(base_target, [base_target])

  multitarget = []
  for t in targets:
    if not t.startswith(expected_base):
      fail(
          "target %s does not start with expected target %s for halide_target_map"
          % (t, expected_base))
    t = t[len(expected_base):]
    if t.startswith("-"):
      t = t[1:]
    # Check for a "match all base targets" entry:
    multitarget.append(_add_target_features(base_target, t.split("-")))

  # Add the extra features (if any).
  if halide_target_features:
    multitarget = [
        _add_target_features(t, halide_target_features) for t in multitarget
    ]

  # Finally, canonicalize all targets
  multitarget = [_canonicalize_target(t) for t in multitarget]
  return multitarget


def _gengen_closure_impl(ctx):
  return struct(
      generator_binary=ctx.attr.generator_binary,
      generator_name=ctx.attr.halide_generator_name)


_gengen_closure = rule(
    implementation=_gengen_closure_impl,
    attrs={
        "generator_binary":
            attr.label(
                executable=True, allow_files=True, mandatory=True, cfg="host"),
        "halide_generator_name":
            attr.string(),
    })


def halide_generator(name,
                     srcs,
                     deps=[],
                     generator_name="",
                     tags=[],
                     visibility=None,
                     includes=[]):
  # TODO: this enforcement may be overkill, but enforcing rather
  # than declaring it "best practice" may simplify the world
  if not name.endswith("_generator"):
    fail("halide_generator rules must end in _generator")

  if not generator_name:
    generator_name = name[:-10]  # strip "_generator" suffix

  native.cc_library(
      name="%s_library" % name,
      srcs=srcs,
      alwayslink=1,
      copts=halide_language_copts(),
      deps=["@halide//:language"] + deps,
      tags=tags,
      visibility=["//visibility:private"])

  native.cc_binary(
      name="%s_binary" % name,
      copts=halide_language_copts(),
      linkopts=halide_language_linkopts(),
      deps=[
          ":%s_library" % name,
          "@halide//:internal_halide_generator_glue",
      ],
      tags=tags,
      visibility=["//visibility:private"])
  _gengen_closure(
      name="%s_closure" % name,
      generator_binary="%s_binary" % name,
      halide_generator_name=generator_name,
      visibility=["//visibility:private"])

  # If srcs is empty, we're building the halide-library-runtime,
  # which has no stub: just skip it.
  stub_gen_hdrs_target = []
  if srcs:
    # The specific target doesn't matter (much), but we need
    # something that is valid, so uniformly choose first entry
    # so that build product cannot vary by build host
    stub_header_target = _select_multitarget(
        base_target=_HALIDE_TARGET_CONFIG_INFO[0][0],
        halide_target_features=[],
        halide_target_map={})
    _gengen(
        name="%s_stub_gen" % name,
        filename=name[:-10],  # strip "_generator" suffix
        generator_closure=":%s_closure" % name,
        halide_target=stub_header_target,
        outputs=["cpp_stub"],
        tags=tags,
        visibility=["//visibility:private"])
    stub_gen_hdrs_target = [":%s_stub_gen" % name]

  native.cc_library(
      name=name,
      alwayslink=1,
      hdrs=stub_gen_hdrs_target,
      deps=[
          ":%s_library" % name,
          "@halide//:language"
        ],
      includes=includes,
      visibility=visibility,
      tags=tags)


def halide_library(name,
                   debug_codegen_level=0,
                   deps=[],
                   extra_outputs=[],
                   function_name=None,
                   generator=None,
                   generator_args="",
                   halide_target_features=[],
                   halide_target_map=halide_library_default_target_map(),
                   includes=[],
                   namespace=None,
                   tags=[],
                   trace_level=0,
                   visibility=None):
  if not function_name:
    function_name = name

  if namespace:
    function_name = "%s::%s" % (namespace, function_name)
    halide_target_features += ["c_plus_plus_name_mangling"]

  # Escape backslashes and double quotes.
  generator_args = generator_args.replace("\\", '\\\\"').replace('"', '\\"')

  if _has_dupes(halide_target_features):
    fail("Duplicate values in halide_target_features: %s" %
         str(halide_target_features))
  if _has_dupes(extra_outputs):
    fail("Duplicate values in extra_outputs: %s" % str(extra_outputs))

  outputs = extra_outputs
  # TODO: yuck. hacky for apps/c_backend.
  if not "cpp" in outputs:
    outputs += ["static_library"]

  generator_closure = "%s_closure" % generator

  condition_deps = {}
  for base_target, _, _, _ in _HALIDE_TARGET_CONFIG_INFO:
    multitarget = _select_multitarget(
        base_target=base_target,
        halide_target_features=halide_target_features,
        halide_target_map=halide_target_map)
    arch = _halide_target_to_bazel_rule_name(base_target)
    sub_name = "%s_%s" % (name, arch)
    _gengen(
        name=sub_name,
        filename=sub_name,
        halide_generator_args=generator_args,
        generator_closure=generator_closure,
        halide_target=multitarget,
        halide_function_name=function_name,
        sanitizer=select({
            "@halide//:halide_config_msan": "msan",
            "//conditions:default": "",
        }),
        debug_codegen_level=debug_codegen_level,
        tags=tags,
        trace_level=trace_level,
        outputs=outputs)
    libname = "halide_internal_%s_%s" % (name, arch)
    if "static_library" in outputs:
      native.cc_library(
          name=libname,
          srcs=[":%s.a" % sub_name],
          tags=tags,
          visibility=["//visibility:private"])
    elif "cpp" in outputs:
      # TODO: yuck. hacky for apps/c_backend.
      if len(multitarget) > 1:
        fail(
            'can only request .cpp output if no multitargets are selected. Try' +
            ' adding halide_target_map={"*":["*"]} to your halide_library ' +
            'rule.'
        )
      native.cc_library(
          name=libname,
          srcs=[":%s.cpp" % sub_name],
          tags=tags,
          visibility=["//visibility:private"])
    else:
      fail("either cpp or static_library required")
    condition_deps[_config_setting(
        base_target)] = _HALIDE_RUNTIME_OVERRIDES.get(
            base_target, []) + [":%s" % libname]

  # Note that we always build the .h file using the first entry in
  # the _HALIDE_TARGET_CONFIG_INFO table.
  header_target = _select_multitarget(
      base_target=_HALIDE_TARGET_CONFIG_INFO[0][0],
      halide_target_features=halide_target_features,
      halide_target_map=halide_target_map)
  if len(header_target) > 1:
    # This can happen if someone uses halide_target_map
    # to force everything to be multitarget. In that
    # case, just use the first entry.
    header_target = [header_target[0]]

  outputs = ["h"]
  _gengen(
      name="%s_header" % name,
      filename=name,
      halide_generator_args=generator_args,
      generator_closure=generator_closure,
      halide_target=header_target,
      halide_function_name=function_name,
      outputs=outputs,
      tags=tags)

  native.cc_library(
      name=name,
      hdrs=[":%s_header" % name],
      deps=["@halide//:runtime"] + select(condition_deps) + deps,
      includes=includes,
      tags=tags,
      visibility=visibility)


# TODO: we probably don't want to keep this; leaving it here temporarily just in case
#
# def halide_gen_and_lib(name,
#                    visibility=None,
#                    namespace=None,
#                    function_name=None,
#                    generator_args="",
#                    debug_codegen_level=0,
#                    trace_level=0,
#                    halide_target_features=[],
#                    halide_target_map=halide_library_default_target_map(),
#                    extra_outputs=[],
#                    includes=[],
#                    srcs=None,
#                    filter_deps=[],
#                    generator_deps=[],
#                    generator_name=""):
#   halide_generator(name="%s_generator" % name,
#                    srcs=srcs,
#                    generator_name=generator_name,
#                    deps=generator_deps,
#                    includes=includes,
#                    visibility=["//visibility:private"])
#   halide_library(name=name,
#                  generator=":%s_generator" % name,
#                  deps=filter_deps,
#                  visibility=visibility,
#                  namespace=namespace,
#                  function_name=function_name,
#                  generator_args=generator_args,
#                  debug_codegen_level=debug_codegen_level,
#                  trace_level=trace_level,
#                  halide_target_features=halide_target_features,
#                  halide_target_map=halide_target_map,
#                  extra_outputs=extra_outputs,
#                  includes=includes)