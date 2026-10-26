{
  lib,
  rustPlatform,
  cmake,
  pkg-config,
}:

rustPlatform.buildRustPackage {
  pname = "tabcomplete-nes";
  version =
    (builtins.fromTOML (builtins.readFile ../../tools/nes_engine/Cargo.toml)).package.version;
  src = lib.cleanSourceWith {
    src = ../../tools/nes_engine;
    filter =
      path: type:
      lib.cleanSourceFilter path type
      && !(builtins.elem (baseNameOf path) [
        "target"
        "eval"
        "nvim"
        "README.md"
      ]);
  };
  cargoLock.lockFile = ../../tools/nes_engine/Cargo.lock;
  nativeBuildInputs = [
    cmake
    pkg-config
    rustPlatform.bindgenHook
  ];
  # llama.cpp ships inside llama-cpp-sys-2 and is built offline by its build
  # script. x86-64-v3 (AVX2, FMA, F16C) matches the ThinkPad i7-8650U; the
  # sys crate forwards these target features to ggml's CMake options.
  env = {
    RUSTFLAGS = "-C target-cpu=x86-64-v3";
    GGML_CUDA = "OFF";
    GGML_VULKAN = "OFF";
    GGML_METAL = "OFF";
  };
  dontUseCmakeConfigure = true;
  # Unit tests run in development (`cargo test`); they would rebuild llama.cpp
  # in a second profile here.
  doCheck = false;
  meta = {
    description = "CPU next-edit engine for Sweep next-edit models with a workspace context engine";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
    mainProgram = "tabcomplete-nes";
  };
}
