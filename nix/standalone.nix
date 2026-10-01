{
  nixosConfig ? "/home/shlok/nixos-config",
  modelVariant ? "qwen",
  experimentalAutoOptIn ? true,
}:

let
  flake = builtins.getFlake "path:${nixosConfig}";
  pkgs = import flake.inputs.nixpkgs { system = "x86_64-linux"; };
  package = import ./default.nix { inherit pkgs modelVariant; };
  home = flake.inputs.home-manager.lib.homeManagerConfiguration {
    inherit pkgs;
    modules = [
      ./home-manager.nix
      {
        home.username = "shlok";
        home.homeDirectory = "/home/shlok";
        home.stateVersion = "26.05";
        services.tabcomplete = {
          inherit package experimentalAutoOptIn;
          enable = true;
        };
      }
    ];
  };
  service = home.config.xdg.configFile."systemd/user/tabcomplete-engine.service".source;
  plugin = home.config.xdg.configFile."lazyvim/lua/plugins/tabcomplete-trajectory.lua".source;
in
pkgs.runCommand "tabcomplete-standalone-${modelVariant}" { passthru = { inherit package; }; } ''
  mkdir -p "$out/bin" "$out/systemd/user" "$out/lazyvim/lua/plugins"
  ln -s '${pkgs.jq}/bin/jq' "$out/bin/jq"
  ln -s '${service}' "$out/systemd/user/tabcomplete-engine.service"
  ln -s '${plugin}' "$out/lazyvim/lua/plugins/tabcomplete-trajectory.lua"
  ln -s '${package}' "$out/package"
''
