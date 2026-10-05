{
  pkgs,
  fimModel,
  fimProfile,
}:

let
  lib = pkgs.lib;
  engine = pkgs.callPackage ./package.nix { };
  embeddedProfile = builtins.fromJSON (builtins.readFile fimProfile);
  sha256 = embeddedProfile.model_sha256;
  servingProfile = embeddedProfile.serving_profile;
  editorProfile = servingProfile // {
    tokenizer = builtins.removeAttrs servingProfile.tokenizer [ "tokenizer_vocab_ids" ];
  };
  plugin = lib.cleanSource ../tools/trajectory_collector/nvim;
in
assert builtins.pathExists fimModel;
assert builtins.pathExists fimProfile;
assert embeddedProfile.schema == "tabcomplete-q25-fim-embedded-profile-v1";
assert builtins.match "[0-9a-f]{64}" sha256 != null;
pkgs.runCommand "tabcomplete-q25-fim-candidate-${engine.version}"
  {
    nativeBuildInputs = [ pkgs.python311 ];
    preferLocalBuild = true;
    allowSubstitutes = false;
    disallowedReferences = [
      engine
      fimModel
      fimProfile
    ];
    passthru = {
      inherit sha256;
      alias = "q25-fim";
      modelName = "q25-fim";
      protocol = "q25-fim-line-completion-v1";
      contextLayout = "q25-fim-psm-bounded-v2";
      precision = "Q4_K_M";
      adapterIdentity = "full-weight";
      outputTokens = 96;
      modeStateFile = "tabcomplete-q25-fim-mode.json";
      allowedModels.q25-fim = {
        model_sha256 = sha256;
        model_protocol = "q25-fim-line-completion-v1";
        output_tokens = 96;
        fim_profile = editorProfile;
      };
    };
    meta = engine.meta // {
      description = "Opt-in CPU embedded Qwen FIM research candidate";
      mainProgram = "tabcomplete-q25-fim";
    };
  }
  ''
    mkdir -p "$out/bin" "$out/share/tabcomplete"
    python3 ${./embed-fim-model.py} \
      --engine ${engine}/bin/tabcomplete-engine \
      --model ${fimModel} \
      --profile ${fimProfile} \
      --output "$out/bin/tabcomplete-q25-fim"
    ln -s "$out/bin/tabcomplete-q25-fim" "$out/bin/tabcomplete-engine"
    ln -s ${plugin} "$out/share/tabcomplete/nvim"
  ''
