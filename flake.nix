{
  description = "Esper: Hermetic Neuro-Symbolic Reasoning Engine Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      nixpkgsFor = forAllSystems (system: import nixpkgs { inherit system; });
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgsFor.${system};

          # Create a Filesystem Hierarchy Standard (FHS) chroot
          # Required because Mojo provides pre-compiled ELF binaries that expect
          # standard Linux paths (/lib64/ld-linux-x86-64.so.2, etc.)
          esperEnv = pkgs.buildFHSUserEnv {
            name = "esper-fhs-env";

            # Target packages that the FHS environment will have access to
            targetPkgs = pkgs: with pkgs; [
              uv
              curl
              git
              glibc
              zlib
              ncurses
              libxml2
            ];

            # Initialization script runs inside the FHS chroot
            profile = ''
              export PROJECT_ROOT="$(pwd)"
              export MODULAR_HOME="$PROJECT_ROOT/.modular-home"
              export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$MODULAR_HOME/bin:$PATH"
              export UV_PROJECT_ENVIRONMENT="$PROJECT_ROOT/.venv"

              # Automated Bootstrapping: If Mojo isn't installed locally, install it
              if [ ! -f "$MODULAR_HOME/pkg/packages.modular.com_mojo/bin/mojo" ]; then
                echo "Mojo not found in local hermetic environment. Bootstrapping..."

                # Download and execute the modular installer script
                # (Assuming the installer does not implicitly use forbidden shell keywords)
                curl -ssL https://magic.modular.com/b0a703d8-fb5a-47d0-a05d-e0cb20e3a6fa > install_mod.sh
                chmod +x install_mod.sh
                ./install_mod.sh
                rm install_mod.sh

                # Use the newly installed modular CLI to install mojo
                $MODULAR_HOME/bin/modular install mojo
              fi

              echo "Esper Hermetic Environment Initialized."
            '';
          };
        in
        {
          # The default shell is simply executing the FHS environment
          default = esperEnv.env;
        }
      );
    };
}
