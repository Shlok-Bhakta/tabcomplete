# ThinkPad package

The package builds the Rust engine and its native llama.cpp library for
`x86_64-linux`, with an `x86-64-v3` CPU target. The ThinkPad i7-8650U supports
that target. Cargo dependencies come from the checked-in lockfile and Nix
vendors them before the offline build. No Cargo, compiler, Python, network
download, or model conversion runs when the service starts.

`default.nix` bundles the engine, two immutable Q4 GGUFs, the matching Neovim
plugin, and `share/tabcomplete/runtime-manifest.json`. The model is a symlink
to an immutable Nix store file, so the engine can mmap it without copying it
into a mutable service directory. The manifest records the native crate
checksums and the vendored dependency store path. Weights stay outside Git
and this package does not download or publish them.

## Private model import

Transfer the two authorized Q4 artifacts to the ThinkPad. Keep their original
filenames, then import them into the store. The fixed-output check rejects a
different model or altered bytes.

```sh
nix-store --add-fixed sha256 /path/to/q25-public-functional-lr1e5-Q4_K_M.gguf
nix-store --add-fixed sha256 /path/to/sweep-next-edit-1.5b.q4_k_m.gguf
```

`models.nix` pins the hashes, protocols, byte sizes, and output token budgets. The two
artifacts total 1,374,688,864 bytes. Avoid keeping extra copies on the laptop.

## Build without changing the running system

Use the existing ThinkPad flake's pinned package set. These commands run on
the ThinkPad after copying this checkout's `nix/`, Rust crate, and Neovim
plugin into `/home/shlok/.local/state/tabcomplete-rust-editor-r1/`.

```sh
cd /home/shlok/.local/state/tabcomplete-rust-editor-r1
nix build --impure --expr '
  let
    f = builtins.getFlake "path:/home/shlok/nixos-config";
    pkgs = import f.inputs.nixpkgs { system = "x86_64-linux"; };
  in import ./nix/default.nix { inherit pkgs; }
' --out-link result
./result/bin/tabcomplete-engine --help
cat ./result/share/tabcomplete/runtime-manifest.json
```

Add `modelVariant = "sweep";` to the import arguments to start with Sweep.
Both Q4 artifacts are installed. The runtime picker unloads one model before
loading the other and saves only the chosen alias in
`~/.local/state/tabcomplete/selected-model.json`. No model weights are fetched.
Nix builds need the locked source
crates, a recent Rust toolchain from the pinned package set, CMake, GCC,
pkg-config, and libclang. The derivation supplies them and limits native
compilation to two jobs. Build and runtime disable GPU backends.

## Home Manager integration

Import `home-manager.nix` from `hosts/thinkpad/home.nix` and set
`services.tabcomplete.enable = true;`. Supply the bundle through
`services.tabcomplete.package`. The bundle owns the model pin, protocol,
registry, and output limits: 64 tokens for Qwen and 192 for Sweep.
For an external immutable GGUF, pass `modelFile` and `modelSha256` to the
`nix/default.nix` package import, together with its matching `modelVariant`.
The package verifies the bytes and updates its registry and plugin allowlist.

The module defines a user service on `127.0.0.1:19094` with a 1500 MiB memory
limit, no swap allowance, four CPU threads, a 2304-token context, a 1024-token
input budget, and a single inference process. It does not add firewall rules
or change collector storage. The engine bounds its active inference slot.
Automatic training and content telemetry capture stay disabled.
Qwen uses the `cursor-last-v1` layout by default, with context policy
`single-line-cursor-last-context-v1`. It preserves the prompt information and
places the changing cursor metadata after the source context. Set
`services.tabcomplete.contextLayout = "trained-v2";` to select the earlier
layout. Sweep uses its fixed `sweep-window-v1` layout regardless of this
Qwen setting. The context layout is part of the runtime configuration hash.

The generated LazyVim spec replaces only
`lazyvim/lua/plugins/tabcomplete-trajectory.lua`. It loads the complete
plugin directly from the bundle and retains `$TABCOMPLETE_COLLECTOR_URL`,
falling back to the existing `http://100.100.163.102:8787`. All events keep
using the existing collector. Other LazyVim files and mappings retain their
current definitions. Acceptance defaults to `<M-l>` and dismissal to
`<M-BS>`, both configurable. Automatic display requires
`experimentalAutoOptIn = true`; every edit still requires acceptance.
The Rust plugin saves the user's selected mode in its own
`tabcomplete-rust-editor-mode.json` under Neovim's state directory. The first
opted-in launch starts automatic display; later launches preserve manual,
shadow, or off choices without changing the legacy predictor's saved mode.

`thinkpad-home.nix` evaluates the new module against the current ThinkPad
Home Manager configuration. It does not edit the NixOS checkout.

```sh
nix build --impure --file ./nix/thinkpad-home.nix --out-link hm-result
```

This Home Manager result is for evaluation only during this campaign. Do not
run `./hm-result/activate`: it would also apply the existing dirty Home Manager
changes, including AI and Hyprland configuration. Immediate installation
uses the standalone result described below.

The remote `AGENTS.md` says never to run `nrs` and to ask the user to run it.
Do not invoke it. The ThinkPad Nix checkout contains unrelated staged and
unstaged changes, including networking and fingerprint configuration. No
full Home Manager or NixOS activation belongs to this installation.

For an immediate installation that also leaves pending Home Manager changes
unapplied, build `standalone.nix`. It exports exactly the user unit and
LazyVim spec, plus a link to their package. It has no activation script.

```sh
nix build --impure --file ./nix/standalone.nix --out-link standalone-result
```

Back up the existing collector spec and any existing TabComplete user unit,
then link these two exported files into the corresponding paths beneath
`~/.config/`. Reload the user systemd manager and enable only
`tabcomplete-engine.service`. This applies the same definitions as the Home
Manager module without activating pending Hyprland or AI configuration.
`bash ./nix/install-standalone.sh ./standalone-result` performs these steps
after verification. It backs up prior files, symlink targets, service state,
and the dirty configuration diffs under a private state directory before
atomically replacing either file. It does not edit the NixOS checkout.
The installer restarts the single service, checks for a healthy response,
and restores prior files if startup fails. Run
`bash ./nix/install-standalone.sh --rollback BACKUP_DIRECTORY` for an explicit
rollback using the backup directory printed at installation.

## Permanent Nix configuration

After the tested source is committed publicly, add
`pkgs/tabcomplete-engine/default.nix` to the ThinkPad Nix checkout. It uses
`pkgs.fetchFromGitHub` with that exact commit and unpacked source hash, then
imports the fetched `nix/default.nix`. The private model files continue to
come from their existing fixed-output store imports.

Add `home/features/tabcomplete/default.nix` to import the matching fetched
`nix/home-manager.nix`, select the package, and enable experimental automatic
display. Add this feature only to `hosts/thinkpad/home.nix`. Preserve the
other staged and unstaged configuration changes and save their diffs before
editing. The source commit and hash must come from the final tested code;
never point this configuration at the mutable staging checkout.

Review the exact package, module, and import patch before applying it. Track
new files so flake evaluation can see them, then evaluate the integrated
configuration. This prepares the module for the user's next normal rebuild
without activating unrelated pending configuration now.

## Runtime verification

```sh
systemctl --user status tabcomplete-engine.service --no-pager
systemctl --user show tabcomplete-engine.service -p MemoryMax -p MemorySwapMax -p CPUQuotaPerSecUSec
curl --fail --silent http://127.0.0.1:19094/health
curl --fail --silent http://127.0.0.1:19094/v1/models
ss -ltnp 'sport = :19094'
NVIM_APPNAME=lazyvim nvim --headless '+TabCompleteStatus' '+qa'
```

Verify a real synthetic editor request and the corresponding collector events
before calling deployment complete. A successful build and health request
alone do not verify automatic display, acceptance, event delivery, or latency.
