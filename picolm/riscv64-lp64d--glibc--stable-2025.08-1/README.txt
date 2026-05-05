riscv64-lp64d--glibc--stable-2025.08-1

This is a toolchain provided by the https://toolchains.bootlin.com/
service. This toolchain was built using Buildroot.

The complete source code for all components of the toolchain can be
found at http://toolchains.bootlin.com/downloads/releases/sources/.

The complete license text of all components of the toolchain can be
found at http://toolchains.bootlin.com/downloads/releases/licenses/.

This toolchain is based on:

    gcc                  version     14.3.0
    binutils             version     2.43.1
    gdb                  version       15.2
    glibc                version 2.41-70-g1502c248d58cb99a203731707987a4342926e830

For those who would like to reproduce the toolchain, you can just
follow these steps:

    git clone https://github.com/bootlin/buildroot-toolchains.git buildroot
    cd buildroot
    git checkout toolchains.bootlin.com-2025.08.1

    curl http://toolchains.bootlin.com/downloads/releases/toolchains/riscv64-lp64d/build_fragments/riscv64-lp64d--glibc--stable-2025.08-1.defconfig > .config
    make olddefconfig
    make
    make sdk
