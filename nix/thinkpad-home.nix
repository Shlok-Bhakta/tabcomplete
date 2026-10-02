{
  nixosConfig ? "/home/shlok/nixos-config",
  modelVariant ? "qwen",
}:

let
  flake = builtins.getFlake "path:${nixosConfig}";
  system = flake.nixosConfigurations.shlokthinkpad.extendModules {
    modules = [
      ({ pkgs, ... }: {
        home-manager.users.shlok = {
          imports = [ ./home-manager.nix ];
          services.tabcomplete = {
            enable = true;
            package = import ./default.nix { inherit pkgs modelVariant; };
            experimentalAutoOptIn = true;
            automaticNormalMode = true;
          };
        };
      })
    ];
  };
in
system.config.home-manager.users.shlok.home.activationPackage
