# Engine, pinned public model weights, and the Neovim client in one package.
{ pkgs }:

let
  lib = pkgs.lib;
  engine = pkgs.callPackage ./package.nix { };
  modelQ8 = pkgs.fetchurl {
    name = "sweep-next-edit-0.5b.q8_0.gguf";
    url = "https://huggingface.co/sweepai/sweep-next-edit-0.5B/resolve/8175624004cec8bf0b550fba35d6ad92126e48ac/sweep-next-edit-0.5b.q8_0.gguf";
    sha256 = "2d2f5ca856765a086d4ee3d0dc4d30358e9d562e36480e0c754240e9cf45ed64";
  };
  # Q4_0 is repacked by llama.cpp into an AVX2 layout at load time; on the
  # i7-8650U it processes prompts ~25% faster than Q8_0 and saves as much
  # typing in replay evaluation. Single-threaded quantization is deterministic.
  modelQ4 = pkgs.runCommand "sweep-next-edit-0.5b.q4_0.gguf" {
    nativeBuildInputs = [ pkgs.llama-cpp ];
  } ''
    llama-quantize --allow-requantize ${modelQ8} "$out" Q4_0 1
  '';
  # The output layer scores only all-ASCII tokens (94.8k of 151.9k), which is
  # exact for ASCII source: same suggestions in replay, ~6% less time per
  # generated token. Requires the engine's llama.cpp patch.
  modelQ4Ascii = pkgs.runCommand "sweep-next-edit-0.5b.q4_0-ascii.gguf" {
    nativeBuildInputs = [ (pkgs.python3.withPackages (p: [ p.gguf p.numpy ])) ];
  } ''
    python3 ${../../tools/nes_engine/vocab/prune_output_vocab.py} ${modelQ4} "$out" ascii
  '';
  models = {
    q8_0 = modelQ8;
    q4_0 = modelQ4Ascii;
    q4_0_full = modelQ4;
  };
  collector = lib.cleanSource ../../tools/trajectory_collector/nvim;
  client = lib.cleanSource ../../tools/nes_engine/nvim;
  # Collector modules plus the next-edit client. The legacy predictor's
  # plugin file (which auto-runs its setup) is deliberately left out.
  nvim = pkgs.runCommand "tabcomplete-nes-nvim" { } ''
    mkdir -p "$out/lua/tabcomplete_trajectory" "$out/plugin"
    for m in init buffers config events repository spool transport util; do
      cp ${collector}/lua/tabcomplete_trajectory/$m.lua "$out/lua/tabcomplete_trajectory/"
    done
    cp ${collector}/plugin/tabcomplete_trajectory.lua "$out/plugin/"
    cp -r ${client}/lua/tabcomplete_nes "$out/lua/"
  '';
in
pkgs.runCommand "tabcomplete-nes-${engine.version}"
  {
    passthru = {
      inherit engine models nvim;
      # Telemetry identity: source weights hash plus the quantization name.
      modelName = "sweep-next-edit-0.5b";
      modelSha256 = "2d2f5ca856765a086d4ee3d0dc4d30358e9d562e36480e0c754240e9cf45ed64";
      protocol = "sweep-next-edit-v1";
    };
    meta = engine.meta;
  }
  ''
    mkdir -p "$out/bin" "$out/share/tabcomplete"
    ln -s ${engine}/bin/tabcomplete-nes "$out/bin/tabcomplete-nes"
    ln -s ${modelQ4} "$out/share/tabcomplete/model.gguf"
    ln -s ${modelQ8} "$out/share/tabcomplete/model-q8_0.gguf"
    ln -s ${nvim} "$out/share/tabcomplete/nvim"
  ''
