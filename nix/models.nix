{ pkgs }:

let
  require =
    name: sha256:
    pkgs.requireFile {
      inherit name sha256;
      message = ''
        This private local GGUF is never downloaded or published by this package.
        Transfer the authorized artifact and run:
          nix-store --add-fixed sha256 /path/to/${name}
        The filename and SHA-256 must match this model declaration.
      '';
    };
in
{
  qwen = rec {
    name = "q25-public-functional-lr1e5-Q4_K_M.gguf";
    sha256 = "4b83699a7d64b2163315138f4b590113e5d579296642d88853897612754f9acb";
    bytes = 491399808;
    protocol = "single-line-edit-v1";
    outputTokens = 64;
    model = "q25-public-functional-lr1e5";
    file = require name sha256;
  };
  sweep = rec {
    name = "sweep-next-edit-1.5b.q4_k_m.gguf";
    sha256 = "936a3a1e49d867449a8e3e277cb8be4d2883c825ecd3c6b9d3d2ee6688f8bed4";
    bytes = 883289056;
    protocol = "sweep-full-file-v1";
    outputTokens = 192;
    model = "sweep-next-edit-1.5b";
    file = require name sha256;
  };
}
