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
          ];

          profileScript = ''
            export PROJECT_ROOT="$(pwd)"
            export MODULAR_HOME="$PROJECT_ROOT/.modular-home"
            export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$MODULAR_HOME/bin:$PATH"
            export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"

            # Automated Bootstrapping: If Mojo isn't installed locally, install it
            if [ ! -f "$MODULAR_HOME/pkg/packages.modular.com_mojo/bin/mojo" ]; then
              echo "Mojo not found in local hermetic environment. Bootstrapping..."

              # Download and execute the modular installer script
              curl -ssL https://magic.modular.com/b0a703d8-fb5a-47d0-a05d-e0cb20e3a6fa > install_mod.sh
              chmod +x install_mod.sh
              ./install_mod.sh
              rm install_mod.sh

              # Use the newly installed modular CLI to install mojo
              $MODULAR_HOME/bin/modular install mojo
            fi

            echo "Esper Hermetic Environment Initialized."
          '';

          # Determine the appropriate shell based on the OS
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
