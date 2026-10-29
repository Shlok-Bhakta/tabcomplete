# Home Manager module: local Sweep next-edit engine plus the LazyVim client.
{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.services.tabcomplete-nes;
  package = cfg.package;
  engineArgs = lib.escapeShellArgs (
    [
      "--model"
      "${package.models.${cfg.quantization}}"
      "--model-name"
      "${package.modelName}-${cfg.quantization}"
      "--model-sha256"
      package.modelSha256
      "--threads"
      (toString cfg.threads)
      "--threads-batch"
      (toString cfg.threadsBatch)
      "--ctx-size"
      (toString cfg.contextSize)
      "--slots"
      (toString cfg.slots)
      "--retrieval-chunks"
      (toString cfg.retrievalChunks)
      "--hide-active-edit"
      (lib.boolToString cfg.hideActiveEdit)
      "--min-confidence"
      cfg.minConfidence
      "--multiline-min-confidence"
      cfg.multilineMinConfidence
    ]
    ++ cfg.extraArgs
    ++ [
      "serve"
      "--host"
      "127.0.0.1"
      "--port"
      (toString cfg.port)
    ]
  );
  clientOptions = {
    url = "http://127.0.0.1:${toString cfg.port}";
    jump = cfg.jump;
    normal_mode = cfg.normalMode;
    capture_content = cfg.captureContent;
    accept_key = cfg.acceptKey;
    dismiss_key = cfg.dismissKey;
    predict_key = cfg.predictKey;
    tab = cfg.tab;
  };
in
{
  options.services.tabcomplete-nes = {
    enable = lib.mkEnableOption "local CPU next-edit suggestions (Sweep next-edit model)";
    package = lib.mkOption {
      type = lib.types.package;
      default = import ./default.nix { inherit pkgs; };
      description = "Engine, model weights and Neovim client.";
    };
    port = lib.mkOption {
      type = lib.types.port;
      default = 19094;
    };
    quantization = lib.mkOption {
      type = lib.types.enum [
        "q4_0"
        "q8_0"
      ];
      default = "q4_0";
      description = ''
        q4_0 is ~23% faster on the T480s than q8_0 (which shows slightly fewer
        wrong edits); its output layer is limited to ASCII tokens, which gives
        the same suggestions for ASCII code at ~6% less time per token.
      '';
    };
    minConfidence = lib.mkOption {
      type = lib.types.str;
      default = "-1.0";
      description = ''
        Withhold edits whose weakest changed token has a lower log-probability.
        With hideActiveEdit, -1.0 keeps replay quality near the unhidden
        prompt while staying 25% faster.
      '';
    };
    multilineMinConfidence = lib.mkOption {
      type = lib.types.str;
      default = "-0.5";
      description = "Stricter gate for edits that rewrite more than one existing line (they are wrong more often).";
    };
    hideActiveEdit = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Keep the in-progress edit out of the history section so the prompt cache survives typing.";
    };
    threads = lib.mkOption {
      type = lib.types.ints.between 1 16;
      default = 4;
      description = "Decode threads.";
    };
    threadsBatch = lib.mkOption {
      type = lib.types.ints.between 1 16;
      default = 6;
      description = "Prompt-processing threads (6 measured fastest on the i7-8650U).";
    };
    contextSize = lib.mkOption {
      type = lib.types.ints.between 2048 16384;
      default = 4096;
      description = "Context size of each file slot. Each slot has its own small context; cache shifts and attention cost grow with context size.";
    };
    slots = lib.mkOption {
      type = lib.types.ints.between 1 8;
      default = 4;
      description = "Files whose prompt caches are kept, so switching back to a file skips its cold start.";
    };
    retrievalChunks = lib.mkOption {
      type = lib.types.ints.between 0 6;
      default = 2;
      description = "Cross-file definition snippets per prompt.";
    };
    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Additional engine flags (prompt format experiments).";
    };
    jump = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Predict edits away from the cursor (Tab jumps, Tab again accepts).";
    };
    normalMode = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Also predict after normal-mode edits and leaving insert mode.";
    };
    captureContent = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Upload prompts, completions and proposed text to the trajectory collector.";
    };
    tab = lib.mkOption {
      type = lib.types.bool;
      default = true;
    };
    acceptKey = lib.mkOption {
      type = lib.types.str;
      default = "<M-l>";
    };
    dismissKey = lib.mkOption {
      type = lib.types.str;
      default = "<M-BS>";
    };
    predictKey = lib.mkOption {
      type = lib.types.str;
      default = "<M-p>";
    };
    neovim = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
      };
      appName = lib.mkOption {
        type = lib.types.str;
        default = "lazyvim";
      };
      collectorUrl = lib.mkOption {
        type = lib.types.str;
        default = "http://100.100.163.102:8787";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.threadsBatch >= 1 && cfg.threads >= 1;
        message = "tabcomplete-nes needs at least one thread.";
      }
    ];

    home.packages = [ package ];
    home.sessionVariables.TABCOMPLETE_PREDICTOR_URL = "http://127.0.0.1:${toString cfg.port}";

    # Neovim's bytecode cache keys on size and mtime, which a Nix store
    # replacement can keep; drop the cached copy of the spec we own.
    home.activation.tabcompleteNesSpecCache = lib.mkIf cfg.neovim.enable (
      lib.hm.dag.entryAfter [ "linkGeneration" ] ''
        run ${pkgs.bash}/bin/bash ${../invalidate-owned-spec-cache.sh} \
          ${config.programs.neovim.finalPackage or pkgs.neovim}/bin/nvim ${lib.escapeShellArg cfg.neovim.appName} \
          ${lib.escapeShellArg "${config.xdg.configHome}/${cfg.neovim.appName}/lua/plugins/tabcomplete-trajectory.lua"} \
          ${lib.escapeShellArg "${config.xdg.stateHome}/tabcomplete-install-backups/home-manager-cache"}
      ''
    );

    systemd.user.services.tabcomplete-nes = {
      Unit = {
        Description = "TabComplete next-edit engine (Sweep next-edit 0.5B, CPU)";
        After = [ "network.target" ];
      };
      Service = {
        ExecStart = "${package}/bin/tabcomplete-nes ${engineArgs}";
        Restart = "on-failure";
        RestartSec = 5;
        TimeoutStopSec = 10;
        MemoryMax = "1500M";
        MemorySwapMax = "0";
        # No CPUQuota or Nice: llama.cpp threads spin at per-op barriers, so a
        # throttled or preempted thread stalls the others for the whole request.
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        # Workspace indexing reads source files; nothing is written there.
        ProtectHome = "read-only";
        RestrictAddressFamilies = [
          "AF_UNIX"
          "AF_INET"
          "AF_INET6"
        ];
        UMask = "0077";
      };
      Install.WantedBy = [ "default.target" ];
    };

    xdg.configFile."${cfg.neovim.appName}/lua/plugins/tabcomplete-trajectory.lua" =
      lib.mkIf cfg.neovim.enable
        (
          lib.mkForce {
            text = ''
              local plugin = ${builtins.toJSON "${package}/share/tabcomplete/nvim"}
              return {
                {
                  dir = plugin,
                  name = "tabcomplete-nes",
                  lazy = false,
                  priority = 1000,
                  init = function() vim.opt.runtimepath:prepend(plugin) end,
                  config = function()
                    require("tabcomplete_trajectory").setup({
                      server_url = vim.env.TABCOMPLETE_COLLECTOR_URL or ${builtins.toJSON cfg.neovim.collectorUrl},
                    })
                    require("tabcomplete_nes").setup(vim.json.decode(${builtins.toJSON (builtins.toJSON clientOptions)}))
                  end,
                },
              }
            '';
          }
        );
  };
}
