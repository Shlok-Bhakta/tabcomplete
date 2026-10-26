# Engine, pinned public model weights, and the Neovim client in one package.
{ pkgs }:

let
  lib = pkgs.lib;
  engine = pkgs.callPackage ./package.nix { };
  model = pkgs.fetchurl {
    name = "sweep-next-edit-0.5b.q8_0.gguf";
    url = "https://huggingface.co/sweepai/sweep-next-edit-0.5B/resolve/8175624004cec8bf0b550fba35d6ad92126e48ac/sweep-next-edit-0.5b.q8_0.gguf";
    sha256 = "2d2f5ca856765a086d4ee3d0dc4d30358e9d562e36480e0c754240e9cf45ed64";
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
      inherit engine model nvim;
      modelName = "sweep-next-edit-0.5b-q8_0";
      modelSha256 = "2d2f5ca856765a086d4ee3d0dc4d30358e9d562e36480e0c754240e9cf45ed64";
      protocol = "sweep-next-edit-v1";
    };
    meta = engine.meta;
  }
  ''
    mkdir -p "$out/bin" "$out/share/tabcomplete"
    ln -s ${engine}/bin/tabcomplete-nes "$out/bin/tabcomplete-nes"
    ln -s ${model} "$out/share/tabcomplete/model.gguf"
    ln -s ${nvim} "$out/share/tabcomplete/nvim"
  ''
