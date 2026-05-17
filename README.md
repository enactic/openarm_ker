# OpenArm KER
A teleoperation system for OpenArm robots using KER (Kinematic Equivalent Replica).

## Features
- **Joint mapping**: Flexible configuration-based mapping from leader to follower joints
- **USB communication**: Interface with M5Stack CoreS3 via USB vendor mode

## Quick Start

### 1. Install system dependencies

```bash
sudo apt install libusb-1.0-0-dev
```

### 2. Set up udev rules (run once)

**USB vendor mode only (normal use):**

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="303a", MODE="0666"' | sudo tee /etc/udev/rules.d/99-m5stack.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**If you also want to flash firmware (adds stable device name `/dev/m5_ker_485`):**

Put M5Stack into flashing mode (hold RST 3 seconds until green LED lights up), then run:

```bash
SERIAL=$(udevadm info -q property -n /dev/ttyACM0 | grep ID_SERIAL_SHORT | cut -d= -f2)
sudo tee /etc/udev/rules.d/99-m5stack.rules << EOF
# USB vendor mode (normal operation)
SUBSYSTEM=="usb", ATTRS{idVendor}=="303a", MODE="0666"

# Serial mode (flashing) with stable device name
SUBSYSTEM=="tty", ATTRS{idVendor}=="303a", ATTRS{idProduct}=="1001", ATTRS{serial}=="$SERIAL", MODE="0666", SYMLINK+="m5_ker_485"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Press RST once to reboot normally.

### 3. Install

```bash
uv pip install openarm_ker
```

### 4. Connect M5Stack and verify

Plug the M5Stack CoreS3 into your PC via USB and run:

```bash
ker-cli ping
```

Expected output:

```json
{
  "fw": "v1.0.0",
  "hw": "KER-v1.0.0",
  "updated": "2026-05-25"
}
```

### 5. Sample usage

```python
from openarm_ker import KERStream

with KERStream(transport="usb") as stream:
    data = stream.latest()
    if data is not None:
        ts      = data["timestamp"]
        angles  = data["angles"]
        enc_val = data["encoder_value"]
        enc_btn = data["encoder_button"]
        angles_str = " | ".join([f"CH{i+1:02d}: {a:8.2f}°" for i, a in enumerate(angles)])
        print(f"TS: {ts:10d} | {angles_str} | ENC: {enc_val:4d} (Btn: {int(enc_btn)})", end='\r')
```

## CLI

```bash
# Check device connection and fetch schema
ker-cli ping

# Stream raw data to terminal
ker-cli stream

# Serial transport
ker-cli stream --transport serial --port /dev/m5_ker_485 --baud 2000000
```

## Related Links
- 📚 Read the [documentation](https://docs.openarm.dev/software/can/)
- 💬 Join the community on [Discord](https://discord.gg/FsZaZ4z3We)
- 📬 Contact us through <openarm@enactic.ai>

## License
Licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt) for details.
Copyright 2026 Enactic, Inc.

## Code of Conduct
All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).