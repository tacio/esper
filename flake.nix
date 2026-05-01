{
  description = "Esper: Hermetic Neuro-Symbolic Reasoning Engine Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      nixpkgsFor = forAllSystems (system: import nixpkgs { inherit system; });
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgsFor.${system};

          commonPkgs = with pkgs; [
            uv
            curl
            cacert
            git
            zlib
            ncurses
            libxml2
            python3
          ];

          profileScript = ''
            export PROJECT_ROOT="$(pwd)"
            export HOME="$PROJECT_ROOT/.hermetic_home"
            export PATH="$HOME/.magic/bin:$PATH"
            export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"

            mkdir -p "$HOME"

            bootstrap_esper() {
              echo "Bootstrapping Esper environment..."
              # using bash to pipe curl stream
              curl -ssL https://magic.modular.com | /usr/bin/env bash
              export PATH="$HOME/.magic/bin:$PATH"
              magic install mojo
              echo "Esper Hermetic Environment Initialized."
            }

            export -f bootstrap_esper

            echo "Esper Environment Loaded."
            echo "Run 'bootstrap_esper' to install Mojo and configure the environment."
          '';

          shell = if pkgs.stdenv.isLinux then
            (pkgs.buildFHSUserEnv {
              name = "esper-fhs-env";
              targetPkgs = pkgs: commonPkgs ++ [ pkgs.glibc ];
              profile = profileScript;
            }).env
          else
            pkgs.mkShell {
              buildInputs = commonPkgs;
              shellHook = profileScript;
            };
        in
        {
          default = shell;
        }
      );
    };
}
