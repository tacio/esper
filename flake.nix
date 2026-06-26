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
        in
        {
          default = pkgs.mkShell {
            buildInputs = with pkgs; [
              uv
              curl
              git
              clang
              zlib
              ncurses
              libxml2
            ];

            shellHook = ''
              export PROJECT_ROOT="$(pwd)"
              export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"

              echo "Checking Esper environment..."

              if [ ! -f "$PROJECT_ROOT/.venv/bin/mojo" ]; then
                echo "Mojo not found in local environment. Bootstrapping via uv..."

                # We do not use `uv init esper` here because we are already in the repository root.
                # Initialize the virtual environment
                uv venv

                # Activate the environment
                source .venv/bin/activate

                # Install Mojo 1.0 beta from PyPI (prerelease).
                uv pip install "mojo==1.0.0b2" --prerelease allow

                echo "Esper Hermetic Environment Initialized."
              else
                source .venv/bin/activate
                echo "Esper Environment Loaded."
              fi
            '';
          };
        }
      );
    };
}
