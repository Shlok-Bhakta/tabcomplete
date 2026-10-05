{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.services.tabcomplete;
  package = cfg.package;
  sha256 = package.sha256;
  protocol = package.protocol;
  outputTokens = package.outputTokens;
  maximumOutputTokens = lib.foldl' lib.max outputTokens (
    map (model: model.output_tokens) (builtins.attrValues package.allowedModels)
  );
  arguments = lib.escapeShellArgs [
    "--host"
    "127.0.0.1"
    "--port"
    (toString cfg.port)
    "--threads"
    (toString cfg.threads)
    "--prompt-threads"
    (toString cfg.promptThreads)
    "--context-size"
    (toString cfg.contextSize)
    "--context-layout"
    (package.contextLayoutArgument or cfg.contextLayout)
    "--input-tokens"
    (toString cfg.inputTokens)
    "--batch-size"
    (toString cfg.batchSize)
    "--microbatch-size"
    (toString cfg.microbatchSize)
    "--output-tokens"
    (toString outputTokens)
    "--cache-type"
    cfg.cacheType
    "--syntax-validation"
    (if cfg.syntaxValidation then "true" else "false")
  ];
  predictorOptions = {
    url = "http://127.0.0.1:${toString cfg.port}";
    model = package.modelName or "local-model";
    model_revision = sha256;
    protocol_version = protocol;
    precision = package.precision or "Q4_K_M";
    adapter_identity = package.adapterIdentity or "full-weight";
    single_line_input_tokens = cfg.inputTokens;
    max_prompt_tokens = cfg.inputTokens;
    target_prompt_tokens = cfg.inputTokens;
    backend = "rust-editor-v1";
    allowed_models = package.allowedModels;
    runtime_config_hash = builtins.hashString "sha256" (
      builtins.toJSON {
        inherit sha256 protocol;
        inherit (cfg)
          contextSize
          contextLayout
          inputTokens
          batchSize
          microbatchSize
          cacheType
          threads
          promptThreads
          syntaxValidation
          ;
        inherit outputTokens;
      }
    );
    mode = if cfg.experimentalAutoOptIn then "automatic" else "manual";
    experimental_auto_opt_in = cfg.experimentalAutoOptIn;
    automatic_quality_validated = false;
    automatic_personalization_enabled = false;
    automatic_prefix_guard = cfg.automaticPrefixGuard;
    automatic_normal_mode = cfg.automaticNormalMode;
    debounce_ms = 250;
    accept_key = cfg.acceptKey;
    predict_key = cfg.predictKey;
    dismiss_key = cfg.dismissKey;
    synthetic = false;
    persist_mode = true;
  };
in
{
  options.services.tabcomplete = {
    enable = lib.mkEnableOption "local CPU TabComplete inference";
    package = lib.mkOption {
      type = lib.types.package;
      default = import ./default.nix {
        inherit pkgs;
        modelVariant = cfg.modelVariant;
      };
      description = "An executable-embedded model package and its matching Neovim plugin.";
    };
    modelVariant = lib.mkOption {
      type = lib.types.enum [
        "qwen"
        "sweep"
      ];
      default = "qwen";
      description = "Select one fixed embedded model executable declaratively.";
    };
    port = lib.mkOption {
      type = lib.types.port;
      default = 19094;
    };
    threads = lib.mkOption {
      type = lib.types.ints.between 1 8;
      default = 4;
    };
    promptThreads = lib.mkOption {
      type = lib.types.ints.between 1 8;
      default = 4;
    };
    contextSize = lib.mkOption {
      type = lib.types.ints.between 1024 4096;
      default = 2304;
    };
    contextLayout = lib.mkOption {
      type = lib.types.enum [
        "trained-v2"
        "cursor-last-v1"
        "q25-fim-psm-bounded-v2"
      ];
      default = package.contextLayout or "cursor-last-v1";
      description = "Qwen context layout; Sweep always uses its fixed sweep-window-v1 layout.";
    };
    inputTokens = lib.mkOption {
      type = lib.types.ints.between 256 1984;
      default = 1024;
    };
    batchSize = lib.mkOption {
      type = lib.types.ints.between 64 512;
      default = 256;
    };
    microbatchSize = lib.mkOption {
      type = lib.types.ints.between 1 512;
      default = 64;
    };
    cacheType = lib.mkOption {
      type = lib.types.enum [
        "f16"
        "q8"
      ];
      default = "f16";
    };
    syntaxValidation = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Withhold newly introduced Rust syntax errors; disable for experimental raw proposal testing.";
    };
    experimentalAutoOptIn = lib.mkOption {
      type = lib.types.bool;
      default = false;
    };
    automaticPrefixGuard = lib.mkOption {
      type = lib.types.bool;
      default = false;
    };
    automaticNormalMode = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Also predict after normal-mode cursor movement, edits, and buffer entry.";
    };
    acceptKey = lib.mkOption {
      type = lib.types.str;
      default = "<M-l>";
    };
    predictKey = lib.mkOption {
      type = lib.types.str;
      default = "<M-p>";
      description = "Request a prediction in insert or normal mode, preserving occupied mappings.";
    };
    dismissKey = lib.mkOption {
      type = lib.types.str;
      default = "<M-BS>";
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
        assertion = cfg.microbatchSize <= cfg.batchSize;
        message = "TabComplete microbatchSize must not exceed batchSize.";
      }
      {
        assertion = cfg.inputTokens + maximumOutputTokens <= cfg.contextSize;
        message = "TabComplete contextSize must fit the input and every installed model's output budget.";
      }
      {
        assertion = builtins.match "[0-9a-f]{64}" sha256 != null;
        message = "The TabComplete package model hash must be a lowercase hexadecimal SHA-256.";
      }
      {
        assertion =
          protocol != "q25-fim-line-completion-v1" || cfg.contextLayout == "q25-fim-psm-bounded-v2";
        message = "The FIM candidate requires its trained q25-fim-psm-bounded-v2 layout.";
      }
    ];

    home.packages = [ package ];
    home.sessionVariables.TABCOMPLETE_PREDICTOR_URL = "http://127.0.0.1:${toString cfg.port}";

    home.activation.tabcompleteOwnedSpecCache = lib.mkIf cfg.neovim.enable (
      lib.hm.dag.entryAfter [ "linkGeneration" ] ''
        run ${pkgs.bash}/bin/bash ${./invalidate-owned-spec-cache.sh} \
          ${pkgs.neovim}/bin/nvim ${lib.escapeShellArg cfg.neovim.appName} \
          ${lib.escapeShellArg "${config.xdg.configHome}/${cfg.neovim.appName}/lua/plugins/tabcomplete-trajectory.lua"} \
          ${lib.escapeShellArg "${config.xdg.stateHome}/tabcomplete-install-backups/home-manager-cache"}
      ''
    );

    systemd.user.services.tabcomplete-engine = {
      Unit = {
        Description = "TabComplete CPU local next-edit engine";
        After = [ "network.target" ];
      };
      Service = {
        ExecStart = "${package}/bin/tabcomplete-engine ${arguments}";
        Restart = "on-failure";
        RestartSec = 5;
        TimeoutStopSec = 10;
        MemoryMax = "1500M";
        MemorySwapMax = "0";
        CPUQuota = "400%";
        Nice = 10;
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = "read-only";
        RestrictAddressFamilies = [
          "AF_UNIX"
          "AF_INET"
          "AF_INET6"
        ];
        UMask = "0077";
        StateDirectory = "tabcomplete";
        StateDirectoryMode = "0700";
        Environment = [
          "TABCOMPLETE_OBSERVABILITY_CAPTURE_CONTENT=0"
          "TABCOMPLETE_AUTOMATIC_TRAINING=0"
        ];
      };
      Install.WantedBy = [ "default.target" ];
    };

    # Replace only the existing collector spec. Loading directly from the store
    # puts these matching modules ahead of older copied collector modules.
    xdg.configFile."${cfg.neovim.appName}/lua/plugins/tabcomplete-trajectory.lua" =
      lib.mkIf cfg.neovim.enable
        (
          lib.mkForce {
            text = ''
              local plugin = ${builtins.toJSON "${package}/share/tabcomplete/nvim"}
              return {
                {
                  dir = plugin,
                  name = "tabcomplete-trajectory",
                  lazy = false,
                  priority = 1000,
                  init = function() vim.opt.runtimepath:prepend(plugin) end,
                  config = function()
                    require("tabcomplete_trajectory").setup({
                      server_url = vim.env.TABCOMPLETE_COLLECTOR_URL or ${builtins.toJSON cfg.neovim.collectorUrl},
                    })
                    local predictor_options = vim.json.decode(${builtins.toJSON (builtins.toJSON predictorOptions)})
                    predictor_options.mode_state_path = vim.fn.stdpath("state") .. ${
                      builtins.toJSON ("/" + (package.modeStateFile or "tabcomplete-rust-editor-mode.json"))
                    }
                    require("tabcomplete_trajectory.predict").setup(predictor_options)
                  end,
                },
              }
            '';
          }
        );
  };
}
