{
  pkgs,
  fimModel,
  fimProfile,
}:

let
  engine = pkgs.callPackage ./package.nix { };
in
assert builtins.pathExists fimModel;
assert builtins.pathExists fimProfile;
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
      alias = "q25-fim";
      protocol = "q25-fim-line-completion-v1";
      outputTokens = 96;
    };
    meta = engine.meta // {
      description = "Opt-in CPU embedded Qwen FIM research candidate";
      mainProgram = "tabcomplete-q25-fim";
    };
  }
  ''
    mkdir -p "$out/bin"
    python3 ${./embed-fim-model.py} \
      --engine ${engine}/bin/tabcomplete-engine \
      --model ${fimModel} \
      --profile ${fimProfile} \
      --output "$out/bin/tabcomplete-q25-fim"
  ''
