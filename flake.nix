{
  description = "Esper: Neuro-Symbolic Reasoning Engine Development Environment";

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
            ];

            shellHook = ''
              export PROJECT_ROOT="$(pwd)"
              export PATH="$HOME/.modular/pkg/packages.modular.com_mojo/bin:$PATH"
              export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"

              echo "Esper Development Environment initialized."
              echo "Using uv for Python environment management."
            '';
          };
        }
      );
    };
}
