{
  lib,
  rustPlatform,
  cmake,
  pkg-config,
}:

rustPlatform.buildRustPackage {
  pname = "tabcomplete-engine";
  version =
    (builtins.fromTOML (builtins.readFile ../tools/tabcomplete_engine/Cargo.toml)).package.version;
  src = lib.cleanSourceWith {
    src = ../tools/tabcomplete_engine;
    filter = path: type: lib.cleanSourceFilter path type && baseNameOf path != "target";
  };
  cargoLock.lockFile = ../tools/tabcomplete_engine/Cargo.lock;
  nativeBuildInputs = [
    cmake
    pkg-config
    rustPlatform.bindgenHook
  ];
  # llama.cpp is shipped in the pinned sys crate. Cargo builds it offline.
  # x86-64-v3 includes AVX2 and is supported by the ThinkPad i7-8650U.
  env = {
    RUSTFLAGS = "-C target-cpu=x86-64-v3";
    CMAKE_BUILD_PARALLEL_LEVEL = "2";
    GGML_CUDA = "OFF";
    GGML_HIP = "OFF";
    GGML_VULKAN = "OFF";
    GGML_OPENCL = "OFF";
    GGML_METAL = "OFF";
    GGML_CPU_REPACK = "OFF";
  };
  # No application build.rs: cmake is used only inside the vendored sys crate.
  dontUseCmakeConfigure = true;
  meta = {
    description = "CPU local next-edit backend with bounded editor context";
    platforms = [ "x86_64-linux" ];
    mainProgram = "tabcomplete-engine";
  };
}
