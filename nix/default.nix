{
  pkgs,
  modelVariant ? "qwen",
  modelFile ? null,
  modelSha256 ? null,
}:

let
  lib = pkgs.lib;
  models = import ./models.nix { inherit pkgs; };
  selected = models.${modelVariant};
  weights = if modelFile == null then selected.file else modelFile;
  sha256 = if modelSha256 == null then selected.sha256 else modelSha256;
  engine = pkgs.callPackage ./package.nix { };
  plugin = lib.cleanSource ../tools/trajectory_collector/nvim;
  installedModels = {
    q25 =
      models.qwen
      // lib.optionalAttrs (modelVariant == "qwen") {
        file = weights;
        inherit sha256;
      };
    sweep =
      models.sweep
      // lib.optionalAttrs (modelVariant == "sweep") {
        file = weights;
        inherit sha256;
      };
  };
  modelRegistry = pkgs.writeText "tabcomplete-models.json" (
    builtins.toJSON (
      lib.mapAttrs (_: model: {
        path = toString model.file;
        inherit (model) sha256 protocol;
      }) installedModels
    )
  );
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
      schema_version = 1;
      model_variant = modelVariant;
      model_sha256 = sha256;
      installed_models = lib.mapAttrs (_: model: {
        inherit (model) sha256 protocol;
        bytes = if modelFile != null && model.file == weights then null else model.bytes;
      }) installedModels;
      protocol = selected.protocol;
      native_crates = map (crate: {
        inherit (crate)
          name
          version
          checksum
          source
          ;
      }) nativeCrates;
      cpu_target = "x86-64-v3";
      gpu_enabled = false;
      engine = toString engine;
      # Record build provenance without retaining source crates at runtime.
      cargo_dependencies = builtins.unsafeDiscardStringContext (toString engine.cargoDeps);
      automatic_training = false;
    }
  );
in
pkgs.runCommand "tabcomplete-${modelVariant}-${engine.version}"
  {
    nativeBuildInputs = [ pkgs.coreutils ];
    modelSource = weights;
    expectedSha256 = sha256;
    qwenSource = installedModels.q25.file;
    qwenSha256 = installedModels.q25.sha256;
    sweepSource = installedModels.sweep.file;
    sweepSha256 = installedModels.sweep.sha256;
    passthru = {
      inherit
        engine
        sha256
        modelVariant
        modelRegistry
        ;
      protocol = selected.protocol;
      outputTokens = selected.outputTokens;
      modelName = selected.model;
      allowedModels = lib.mapAttrs (_: model: {
        model_sha256 = model.sha256;
        model_protocol = model.protocol;
        output_tokens = model.outputTokens;
      }) installedModels;
    };
    meta = engine.meta;
  }
  ''
    printf '%s  %s\n' "$expectedSha256" "$modelSource" | sha256sum --check --status
    printf '%s  %s\n' "$qwenSha256" "$qwenSource" | sha256sum --check --status
    printf '%s  %s\n' "$sweepSha256" "$sweepSource" | sha256sum --check --status
    mkdir -p "$out/bin" "$out/share/tabcomplete/models"
    ln -s '${engine}/bin/tabcomplete-engine' "$out/bin/tabcomplete-engine"
    ln -s "$modelSource" "$out/share/tabcomplete/model.gguf"
    ln -s "$qwenSource" "$out/share/tabcomplete/models/q25.gguf"
    ln -s "$sweepSource" "$out/share/tabcomplete/models/sweep.gguf"
    ln -s '${modelRegistry}' "$out/share/tabcomplete/models.json"
    ln -s '${plugin}' "$out/share/tabcomplete/nvim"
    ln -s '${manifest}' "$out/share/tabcomplete/runtime-manifest.json"
  ''
