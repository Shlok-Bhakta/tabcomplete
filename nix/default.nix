{
  pkgs,
  modelVariant ? "qwen",
}:

let
  lib = pkgs.lib;
  models = import ./models.nix { inherit pkgs; };
  selected = models.${modelVariant};
  engine = pkgs.callPackage ./package.nix { };
  plugin = lib.cleanSource ../tools/trajectory_collector/nvim;
  payloads = lib.mapAttrs (
    variant: model:
    pkgs.runCommand "tabcomplete-${variant}-embedded-${engine.version}"
      {
        nativeBuildInputs = [ pkgs.python311 ];
        preferLocalBuild = true;
        allowSubstitutes = false;
        disallowedReferences = [
          engine
          model.file
        ];
        passthru = { inherit (model) sha256 protocol outputTokens; };
      }
      ''
        mkdir -p "$out/bin"
        python3 ${./embed-model.py} --engine ${engine}/bin/tabcomplete-engine \
          --model ${model.file} --sha256 ${model.sha256} --variant ${variant} \
          --output "$out/bin/tabcomplete-${variant}"
      ''
  ) models;
  lock = builtins.fromTOML (builtins.readFile ../tools/tabcomplete_engine/Cargo.lock);
  nativeCrates = builtins.filter (
    crate:
    builtins.elem crate.name [
      "llama-cpp-2"
      "llama-cpp-sys-2"
    ]
  ) lock.package;
  manifest = pkgs.writeText "tabcomplete-runtime-manifest.json" (
    builtins.toJSON {
      schema_version = 2;
      model_variant = modelVariant;
      model_storage = "executable-mmap";
      model_selection = "declarative";
      model_sha256 = selected.sha256;
      protocol = selected.protocol;
      installed_models = lib.mapAttrs (_: model: {
        inherit (model)
          sha256
          protocol
          bytes
          outputTokens
          ;
      }) models;
      native_crates = map (crate: {
        inherit (crate) name version;
        checksum = crate.checksum or null;
        source = crate.source or "vendored-wrapper";
      }) nativeCrates;
      wrapper_upstream = builtins.fromJSON (
        builtins.readFile ../tools/tabcomplete_engine/vendor/llama-cpp-2/UPSTREAM.json
      );
      cpu_target = "x86-64-v3";
      gpu_enabled = false;
      automatic_training = false;
      cargo_dependencies = builtins.unsafeDiscardStringContext (toString engine.cargoDeps);
    }
  );
in
pkgs.runCommand "tabcomplete-${modelVariant}-${engine.version}"
  {
    preferLocalBuild = true;
    allowSubstitutes = false;
    passthru = {
      inherit engine modelVariant payloads;
      sha256 = selected.sha256;
      protocol = selected.protocol;
      outputTokens = selected.outputTokens;
      modelName = selected.model;
      allowedModels = {
        ${if modelVariant == "qwen" then "q25" else "sweep"} = {
          model_sha256 = selected.sha256;
          model_protocol = selected.protocol;
          output_tokens = selected.outputTokens;
        };
      };
    };
    meta = engine.meta;
  }
  ''
    mkdir -p "$out/bin" "$out/share/tabcomplete"
    ln -s ${payloads.qwen}/bin/tabcomplete-qwen "$out/bin/tabcomplete-qwen"
    ln -s ${payloads.sweep}/bin/tabcomplete-sweep "$out/bin/tabcomplete-sweep"
    ln -s "$out/bin/tabcomplete-${modelVariant}" "$out/bin/tabcomplete-engine"
    ln -s ${plugin} "$out/share/tabcomplete/nvim"
    ln -s ${manifest} "$out/share/tabcomplete/runtime-manifest.json"
  ''
