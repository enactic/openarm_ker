# OpenArm KER

A teleoperation system for OpenArm robots using KER (Kinematic Equivalent Replica).

## Features

- **Joint mapping**: Flexible configuration-based mapping from leader to follower joints
- **Serial communication**: Interface with M5Stack CoreS3 over UART

## Quick start

### Install

```bash
pip install openarm_ker
```

### Serial device permissions

On Linux, serial devices such as `/dev/ttyACM0` are usually owned by the
`dialout` group. Add your user to that group, then log out and log back in.
If you run the examples from VS Code or another terminal, restart that program
so it picks up the new group permissions.

```bash
sudo usermod -aG dialout "$USER"
```

For a temporary test, you can also relax the permission of the current device
node directly:

```bash
sudo chmod 666 /dev/ttyACM0
```

This usually resets after the device is unplugged or the device node is
recreated. Use this only as a short-lived local test because it makes the device
writable by every local user. For regular use, prefer the `dialout` group or a
udev rule with `MODE="0660"`.

To use a stable device name such as `/dev/m5_ker_485`, create a udev rule.
First inspect the device properties:

```bash
udevadm info -q property -n /dev/ttyACM0
```

Record fields like these:

```bash
ID_VENDOR_ID=xxxx
ID_MODEL_ID=yyyy
ID_SERIAL_SHORT=zzzz
```

Create a rule file:

```bash
sudo nano /etc/udev/rules.d/99-openarm-ker.rules
```

Add a rule like this, replacing `xxxx`, `yyyy`, and `zzzz` with your device's
actual values:

```udev
SUBSYSTEM=="tty", ENV{ID_VENDOR_ID}=="xxxx", ENV{ID_MODEL_ID}=="yyyy", ENV{ID_SERIAL_SHORT}=="zzzz", SYMLINK+="m5_ker_485", GROUP="dialout", MODE="0660"
```

If you do not need to distinguish between multiple devices of the same model,
you can omit `ENV{ID_SERIAL_SHORT}=="zzzz"`. For a personal development machine,
you can instead use `MODE="0666"` and omit `GROUP` to allow all local users to
access it.

Reload the rules:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then reconnect the device and check that the stable device name was created:

```bash
ls -l /dev/m5_ker_485
```

You can then use `/dev/m5_ker_485` as the serial device path.

### Sample usage

```python
import numpy as np
import openarm_ker

m5_port = openarm_ker.M5Port("/dev/ttyACM0")

leader_joint_names = [f"right_arm_joint{i}" for i in range(1, 9)]
mapper = openarm_ker.Mapper(
    mappingyaml_path="mapping_m5.yaml",
    leader_joint_names=leader_joint_names,
    mapping_key="right_arm_mappings",
)

m5_port.fetch_present_status_bulk()
leader_position = m5_port.present_position
follower_position = mapper.map(np.deg2rad(leader_position))
```

### Mapper config

The main M5 mapping file is `src/openarm_ker/config/mapping_m5.yaml` in this
repository. It is bundled in the installed package under `openarm_ker/config/`,
so you can pass the bundled filename `mapping_m5.yaml`, or pass an explicit path
to a custom YAML file. For the left arm, use `left_arm_joint*` leader names with
`mapping_key="left_arm_mappings"`.

## Related links

- 📚 Read the [documentation](https://docs.openarm.dev/software/can/)
- 💬 Join the community on [Discord](https://discord.gg/FsZaZ4z3We)
- 📬 Contact us through <openarm@enactic.ai>

## License

Licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
