Bare Metal Deployment Guide
============================

Deploying SmallAIOS to Raspberry Pi 4/5 and NVIDIA Jetson dev boards
from a local Linux server with minimal manual steps.

.. contents:: Table of Contents
   :depth: 3

Prerequisites
-------------

**Build Server (your Linux machine):**

- Rust nightly with ``aarch64-unknown-none`` target
- ``qemu-system-aarch64`` (for pre-flight testing before hardware deploy)
- ``dnsmasq`` (DHCP + TFTP server for network boot)
- ``picocom`` or ``minicom`` (serial console)
- USB-to-UART adapter (for Raspberry Pi serial)

**Network:**

- Both the server and dev board on the same LAN (or direct Ethernet cable)
- Server has a static IP or known address

Install prerequisites on Debian/Ubuntu::

    sudo apt install -y dnsmasq picocom qemu-system-arm \
        gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu


Architecture Overview
---------------------

::

    ┌──────────────────┐         ┌──────────────────────┐
    │  Linux Server    │  LAN    │  Dev Board           │
    │                  │◄───────►│  (RPi 4/5 / Jetson)  │
    │  - Cross-compile │  ETH    │                      │
    │  - TFTP server   │         │  Boots via:          │
    │  - DHCP (opt)    │         │  1. Network (TFTP)   │
    │  - Serial console│  USB/   │  2. SD card          │
    │    viewer        │  UART   │  3. USB flash        │
    └──────────────────┘         └──────────────────────┘

Three deployment methods are provided, ordered by automation level:

1. **Network Boot (PXE/TFTP)** — most automated, reboot-to-deploy
2. **SD Card Image** — portable, no network infra needed
3. **Jetson USB Recovery Flash** — NVIDIA-specific, uses vendor tools


Method 1: Network Boot (Recommended for Development)
-----------------------------------------------------

This is the fastest iteration loop: ``make deploy-netboot`` on the server,
reboot the board, and it picks up the new kernel automatically.

One-Time Server Setup
~~~~~~~~~~~~~~~~~~~~~

Run the setup script::

    sudo ./scripts/deploy-netboot.sh setup --server-ip 192.168.1.100

This configures ``dnsmasq`` as a TFTP server (and optionally a DHCP server
for a dedicated dev subnet). The TFTP root is ``/srv/tftp/smallaios/``.

If you already have a DHCP server on the network (e.g., your router), you
can run TFTP-only mode — the board just needs to know the TFTP server IP.

One-Time Board Setup
~~~~~~~~~~~~~~~~~~~~

**Raspberry Pi 4/5:**

1. Download the RPi UEFI firmware (EDK2 port)::

       # On the server — downloads and extracts to an SD card
       ./scripts/deploy-rpi-sdcard.sh uefi-only /dev/sdX

2. Insert the SD card into the RPi and boot
3. Enter UEFI setup (press ESC at the splash screen)
4. Navigate to: **Boot Maintenance → Boot Options → Add Boot Option**

   - For network boot: select the MAC-based PXE option
   - Set as first boot priority

5. Under **Device Manager → Raspberry Pi Configuration**:

   - Set **System Table Selection** → ACPI + Devicetree
   - Set **CPU Clock** → Max

After this one-time setup, the RPi will network-boot every time.

**NVIDIA Jetson Orin Nano:**

The Jetson dev kit's U-Boot already supports TFTP. Configure it:

1. Connect the Jetson's micro-USB port to the server for serial console
2. Boot the Jetson and interrupt U-Boot (press any key)
3. At the U-Boot prompt::

       setenv bootcmd 'dhcp; tftpboot ${kernel_addr_r} smallaios-aarch64; booti ${kernel_addr_r} - ${fdt_addr}'
       setenv serverip 192.168.1.100
       saveenv
       reset

After this, the Jetson will TFTP-boot SmallAIOS on every power-on.

Deploy Loop (Automated)
~~~~~~~~~~~~~~~~~~~~~~~~

After one-time setup, the development cycle is::

    # Build + deploy in one command
    make deploy-netboot

    # Or manually:
    cargo build --release --target aarch64-unknown-none \
        -p smallaios-arch-aarch64 \
        -Z build-std=core,compiler_builtins,alloc \
        -Z build-std-features=compiler-builtins-mem
    sudo cp target/aarch64-unknown-none/release/smallaios-aarch64 \
        /srv/tftp/smallaios/smallaios-aarch64

    # Reboot the board (via serial or SSH if Linux is running)
    # Or physically press reset

    # Watch serial console
    make serial DEV=/dev/ttyUSB0

The board fetches the new kernel from TFTP and boots in seconds.


Method 2: SD Card Deploy (Raspberry Pi)
----------------------------------------

For situations where network boot isn't available, or for field deployment.

Create a Bootable SD Card
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    # Build and write to SD card in one step
    make deploy-rpi-sdcard DEV=/dev/sdX

    # Or manually:
    ./scripts/deploy-rpi-sdcard.sh full /dev/sdX

This script:

1. Downloads RPi UEFI firmware (EDK2) if not cached
2. Creates a FAT32 boot partition with UEFI firmware
3. Copies the SmallAIOS kernel binary
4. Configures ``startup.nsh`` for automatic UEFI boot

Update Kernel Only (Fast Path)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the SD card already has UEFI firmware::

    # Just update the kernel binary
    make deploy-rpi-sdcard-update DEV=/dev/sdX

    # Or:
    ./scripts/deploy-rpi-sdcard.sh update /dev/sdX

This mounts the existing boot partition, copies the new kernel, and unmounts.
Takes a few seconds.


Method 3: Jetson USB Recovery Flash
------------------------------------

For the initial Jetson setup, or when U-Boot is bricked, use NVIDIA's
USB recovery mode. This requires a physical USB-C cable between the
server and the Jetson.

Prerequisites
~~~~~~~~~~~~~

Install the NVIDIA L4T Board Support Package on your server::

    # Download L4T BSP for your Jetson variant
    # Jetson Orin Nano: https://developer.nvidia.com/embedded/jetson-linux
    # Extract to ~/nvidia/nvidia_sdk/

    # Or install via SDK Manager:
    # sudo apt install nvidia-sdk-manager

Enter Recovery Mode
~~~~~~~~~~~~~~~~~~~

1. Power off the Jetson
2. Hold the **RECOVERY** button (middle button on the carrier board)
3. While holding RECOVERY, press and release **RESET** (or plug in power)
4. Release RECOVERY after 2 seconds
5. Verify on the server::

       lsusb | grep -i nvidia
       # Should show: "NVIDIA Corp. APX" or similar

Flash
~~~~~

::

    # Flash SmallAIOS as the boot image
    ./scripts/deploy-jetson-flash.sh /path/to/nvidia_sdk

    # This replaces the kernel in the L4T boot partition with SmallAIOS

To restore stock JetPack Linux later::

    cd ~/nvidia/nvidia_sdk/JetPack_*/Linux_for_Tegra
    sudo ./flash.sh jetson-orin-nano-devkit mmcblk0p1


Serial Console
--------------

Both dev boards expose serial consoles for debugging. SmallAIOS outputs
all early boot messages, kernel logs, and panic traces to the serial port.

Raspberry Pi 4/5
~~~~~~~~~~~~~~~~~

The RPi exposes UART0 on the 40-pin GPIO header:

===== ========== ==============
Pin   Function   Wire Color
===== ========== ==============
6     GND        Black
8     UART TX    White (→ RX)
10    UART RX    Green (→ TX)
===== ========== ==============

Connect a USB-to-UART adapter (3.3V logic — **not** 5V RS-232) to these
pins, then::

    make serial DEV=/dev/ttyUSB0

    # Or directly:
    picocom -b 115200 /dev/ttyUSB0

.. warning::

    **Do not connect the 5V pin** — the RPi GPIO is 3.3V only.
    Use a 3.3V USB-to-UART adapter (FTDI FT232R, CP2102, CH340 are common).

NVIDIA Jetson Dev Kits
~~~~~~~~~~~~~~~~~~~~~~~

Jetson dev kits have a built-in FTDI USB-to-UART chip. Connect the
micro-USB (Jetson Nano) or USB-C debug port (Orin Nano) to your server::

    # Find the device
    ls /dev/ttyACM* /dev/ttyUSB*

    # Connect
    make serial DEV=/dev/ttyACM0

    # Default baud rate for Jetson is 115200

JTAG Debugging (Advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both platforms support JTAG for hardware-level debugging:

- **RPi 4/5**: JTAG on GPIO pins 22-27 (requires ``enable_jtag_gpio=1``
  in ``config.txt``). Use with OpenOCD + ARM-USB-TINY-H or J-Link.
- **Jetson Orin Nano**: 10-pin JTAG header on the dev carrier board.
  Use with NVIDIA's Tegra JTAG tools or Lauterbach TRACE32.

For most development, serial console + QEMU GDB stub is sufficient.
JTAG is only needed for debugging early boot issues on bare metal.


Quick Reference: Dev Board Comparison
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Raspberry Pi 4/5
     - Jetson Orin Nano
   * - CPU
     - 4x A72 (Pi4) / A76 (Pi5)
     - 6x A78AE
   * - GPU Compute
     - None (CPU inference only)
     - 1024 CUDA cores (Ampere)
   * - RAM
     - 2-8 GB LPDDR4/4X
     - 4/8 GB LPDDR5 (shared)
   * - Serial Console
     - GPIO UART + USB adapter
     - Built-in USB serial
   * - Network Boot
     - UEFI PXE (needs EDK2 SD)
     - U-Boot TFTP (built-in)
   * - Flash Method
     - SD card / USB
     - USB recovery / eMMC
   * - Best For
     - ARM64 testing, low cost
     - GPU inference testing
   * - Board Cost
     - ~$35-80
     - ~$200-250


Development Workflow Summary
-----------------------------

Fastest iteration loop (network boot)::

    # Terminal 1: Serial console (leave open)
    make serial DEV=/dev/ttyUSB0

    # Terminal 2: Build and deploy
    make deploy-netboot     # Builds + copies to TFTP
    # Press reset on the board (or `reboot` via serial if OS supports it)
    # Watch Terminal 1 — SmallAIOS boots in seconds

Pre-flight check with QEMU before hardware::

    make run-arm            # Test in emulator first
    make deploy-netboot     # Then push to hardware

Build + SD card for field deployment::

    make deploy-rpi-sdcard DEV=/dev/sdX
    # Eject, insert into RPi, power on


Troubleshooting
----------------

Board doesn't boot from network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Check TFTP server is running: ``sudo systemctl status dnsmasq``
2. Check the kernel image exists: ``ls -la /srv/tftp/smallaios/``
3. Watch TFTP requests: ``sudo journalctl -f -u dnsmasq``
4. Verify network connectivity: ping the board from the server
5. On RPi: verify UEFI PXE is set as first boot option
6. On Jetson: verify U-Boot ``serverip`` is correct (``printenv serverip``)

No serial output
~~~~~~~~~~~~~~~~~

1. Check the correct ``/dev/tty*`` device: ``ls /dev/tty{USB,ACM}*``
2. Verify baud rate is 115200
3. For RPi: confirm TX/RX wires aren't swapped (TX→RX, RX→TX)
4. For RPi: ensure UART is enabled (``enable_uart=1`` in ``config.txt``)
5. For Jetson: try both ``/dev/ttyACM0`` and ``/dev/ttyUSB0``
6. Check SmallAIOS early boot: first output should be the boot banner

Kernel loads but crashes immediately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Verify the kernel was built for the correct target (``aarch64-unknown-none``)
2. Check linker script load address matches the bootloader's expectation
3. Run in QEMU first: ``make run-arm`` — if it works in QEMU but not on
   hardware, the issue is likely device-tree or hardware-specific init
4. Enable JTAG for hardware-level single-stepping

Permission denied on /dev/ttyUSB0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    sudo usermod -a -G dialout $USER
    # Log out and back in for the group change to take effect
