const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const optimize = b.standardOptimizeOption(.{});
    const mod = b.addModule("zmemalloc", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    const mimalloc_dep = b.dependency("mimalloc", .{});
    const mimalloc_module = b.addTranslateC(.{
        .root_source_file = mimalloc_dep.path("include/mimalloc.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }).createModule();

    const mimalloc_lib = build_library(b, "mimalloc", mimalloc_module);

    mimalloc_lib.addCSourceFiles(.{
        .root = mimalloc_dep.path("src"),
        .files = &.{"static.c"},
        .flags = &.{
            "-std=c11",
            "-O3",
            "-march=native",
            "-flto",
            "-ffunction-sections",
            "-fdata-sections",
            "-fvisibility=hidden",
            "-Wstrict-prototypes",
            "-ftls-model=initial-exec",
            "-DMI_SECURE=0", // 0 || 1 for debug
            if (optimize == .ReleaseSafe or optimize == .Debug) "-DMI_SECURE=4" else "-DMI_SECURE=0",
            "-DMI_DEBUG=0", // 0  || 3 for debug
            "-DMI_STAT=0", // 0   || 1 for debug
            "-DMI_LIBC_MUSL=1",
            "-DMI_NO_PTHREADS=1",

            "-DMI_NO_VERSION=1",
            "-DMI_OVERRIDE=0",
            "-DMI_TRACK_ASAN=0",
            "-DMI_NO_THP=1", //transperent huge pages
            "-fno-sanitize=undefined",
            "-fno-sanitize-trap=undefined",
            "-ffast-math",
            "-fomit-frame-pointer",
        },
    });

    mimalloc_lib.addIncludePath(mimalloc_dep.path("include"));
    mimalloc_lib.installHeadersDirectory(mimalloc_dep.path("include"), ".", .{});

    const mimalloc_zig = build_module(b, .{
        .root_source_file = b.path("src/mimalloc.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "mimalloc", .module = mimalloc_module },
        },
        .link_libc = true,
    });

    const exe = b.addExecutable(.{
        .name = "zmemalloc",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zmemalloc", .module = mod },
                .{ .name = "mimalloc_zig", .module = mimalloc_zig },
            },
        }),
    });

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    // A run step that will run the test executable.
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    // A run step that will run the second test executable.
    const run_exe_tests = b.addRunArtifact(exe_tests);

    // A top level step for running all tests. dependOn can be called multiple
    // times and since the two run steps do not depend on one another, this will
    // make the two of them run in parallel.
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // Just like flags, top level steps are also listed in the `--help` menu.
    //
    // The Zig build system is entirely implemented in userland, which means
    // that it cannot hook into private compiler APIs. All compilation work
    // orchestrated by the build system will result in other Zig compiler
    // subcommands being invoked with the right flags defined. You can observe
    // these invocations when one fails (or you pass a flag to increase
    // verbosity) to validate assumptions and diagnose problems.
    //
    // Lastly, the Zig build system is relatively simple and self-contained,
    // and reading its source code will allow you to master it.
}

fn build_library(b: *std.Build, name: []const u8, module: *std.Build.Module) *std.Build.Step.Compile {
    const l = b.addLibrary(.{
        .name = name,
        .root_module = module,
        .linkage = .static,
    });
    strip_step(l);

    return l;
}

fn strip_step(step: *std.Build.Step.Compile) void {
    if (step.root_module.optimize != .Debug and step.root_module.optimize != .ReleaseSafe) {
        step.use_llvm = true;
        step.link_eh_frame_hdr = false;
        step.link_emit_relocs = false;
        step.lto = .full;
        step.bundle_compiler_rt = true;
        step.pie = false;
        step.bundle_ubsan_rt = false;
        step.link_gc_sections = true;
        step.link_function_sections = true;
        step.link_data_sections = true;
        step.discard_local_symbols = true;
        step.compress_debug_sections = .none;
    } else {
        step.use_llvm = true;
    }
    if (@hasField(std.meta.Child(@TypeOf(step)), "llvm_codegen_threads"))
        step.llvm_codegen_threads = step.llvm_codegen_threads orelse 0;
}

fn build_module(
    b: *std.Build,
    options: std.Build.Module.CreateOptions,
) *std.Build.Module {
    const m = b.createModule(options);
    strip(m);
    return m;
}

fn strip(root_module: *std.Build.Module) void {
    if (root_module.optimize != .Debug and root_module.optimize != .ReleaseSafe) {
        root_module.strip = true;
        root_module.omit_frame_pointer = true;
        root_module.unwind_tables = .none;
        root_module.sanitize_c = .off;
    } else {
        root_module.strip = false;
        root_module.omit_frame_pointer = false;
        root_module.unwind_tables = .sync;
        root_module.sanitize_c = .full;
    }
}
