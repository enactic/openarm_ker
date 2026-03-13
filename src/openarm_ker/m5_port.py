# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""M5 Encoder Port for reading MT6835 angles via Serial."""

import serial
import struct
import numpy as np
import time
import json


class M5Port:
    """M5 Encoder Port for reading MT6835 angles via Serial.

    Supports both 'binary' (production/fast) and 'json' (debugging) modes.
    """

    def __init__(
        self,
        device,
        num_sensors=16,
        baudrate=2000000,
        timeout=0.005,
        mode="json",
    ):
        """Initialize M5 port.

        Args:
            device: Serial port device (e.g., '/dev/ttyACM0', 'COM3').
            num_sensors: Number of encoder sensors (default: 16)
            baudrate: Serial baud rate (default: 2000000).
            timeout: Serial timeout in seconds. (0.05 ensures OS-level blocking to drop CPU usage)
            mode: 'binary' or 'json'

        """
        self.device = device
        self.num_sensors = num_sensors
        self.baudrate = baudrate
        self.mode = mode.lower()

        # Initialize arrays (store in degrees)
        self.present_position = np.full(num_sensors, np.nan)
        self.present_errors = np.zeros(num_sensors, dtype=bool)
        self.timestamp = 0

        # Initialize peripherals
        self.chain_encoder = 0
        self.chain_encoder_button = 0
        self.joystick_x = 0
        self.joystick_y = 0
        self.joystick_button = 0

        # Buffers for serial data
        self.byte_buffer = bytearray()
        self.string_buffer = ""

        # Binary packet format:
        # < : Little Endian
        # H : Header (2 bytes)
        # I : Timestamp (4 bytes)
        # 16f : Angles (16 * 4 = 64 bytes)
        # H : Error Mask (2 bytes)
        # h : Chain Encoder (2 bytes)
        # h : Joystick X (2 bytes)
        # h : Joystick Y (2 bytes)
        # B : Buttons Mask (1 byte)
        # B : Checksum (1 byte)
        # Total: 80 bytes
        self.PACKET_FORMAT = "<HI16fHhhhBB"
        self.PACKET_SIZE = struct.calcsize(self.PACKET_FORMAT)

        # --- Serial Setup ---
        self.serial = serial.Serial(device, baudrate, timeout=timeout)

        # Wait for initial data to sync
        self._wait_for_initial_data()

    def _wait_for_initial_data(self, max_wait=2.0):
        """Wait for initial valid data from M5."""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            if self._read_once():
                return
            # sleep is ok here since it's just startup
            time.sleep(0.01)

    def _read_once(self):
        """Read data from Serial and parse the LATEST complete packet.

        Returns: True if valid data was read, False otherwise.
        """
        if self.mode == "binary":
            return self._read_binary()
        elif self.mode == "json":
            return self._read_json()
        return False

    # ---------------------------------------------------------
    # BINARY PARSING LOGIC
    # ---------------------------------------------------------
    def _read_binary(self):
        try:
            to_read = max(1, self.serial.in_waiting)
            chunk = self.serial.read(to_read)

            if not chunk:
                return False  # Timeout

            self.byte_buffer.extend(chunk)

            # Prevent memory leak if PC stalls
            if len(self.byte_buffer) > 4096:
                del self.byte_buffer[: -self.PACKET_SIZE * 2]

            parsed_any = False

            # Process all complete packets in the buffer
            while len(self.byte_buffer) >= self.PACKET_SIZE:
                # Search for Little-Endian header: 0xA55A (0x5A, 0xA5)
                if self.byte_buffer[0] == 0x5A and self.byte_buffer[1] == 0xA5:
                    packet = self.byte_buffer[: self.PACKET_SIZE]

                    # Calculate and verify XOR checksum
                    calculated_checksum = 0
                    for i in range(self.PACKET_SIZE - 1):
                        calculated_checksum ^= packet[i]

                    if calculated_checksum == packet[-1]:
                        # Valid packet! Parse it.
                        self._parse_binary_packet(packet)
                        parsed_any = True
                        del self.byte_buffer[: self.PACKET_SIZE]
                    else:
                        # Corrupted data. Drop the first byte to re-align.
                        del self.byte_buffer[0:1]
                else:
                    # Header not found at start. Drop 1 byte and search again.
                    del self.byte_buffer[0:1]

            return parsed_any

        except Exception:
            return False

    def _parse_binary_packet(self, packet):
        """Unpack strictly mapped C++ struct."""
        data = struct.unpack(self.PACKET_FORMAT, packet)

        self.timestamp = data[1]
        angles = data[2:18]
        error_mask = data[18]

        for i in range(self.num_sensors):
            is_err = bool((error_mask >> i) & 1)
            self.present_errors[i] = is_err
            if is_err:
                self.present_position[i] = np.nan
            else:
                self.present_position[i] = angles[i]

        self.chain_encoder = data[19]
        self.joystick_x = data[20]
        self.joystick_y = data[21]

        button_mask = data[22]
        self.chain_encoder_button = button_mask & 0x01
        self.joystick_button = (button_mask >> 1) & 0x01

    # ---------------------------------------------------------
    # JSON PARSING LOGIC
    # ---------------------------------------------------------
    def _read_json(self):
        try:
            to_read = max(1, self.serial.in_waiting)
            chunk = self.serial.read(to_read).decode("utf-8", errors="ignore")

            if not chunk:
                return False

            self.string_buffer += chunk

            latest_valid_line = None
            while "\n" in self.string_buffer:
                line, self.string_buffer = self.string_buffer.split("\n", 1)
                line = line.strip()
                if line:
                    latest_valid_line = line

            if latest_valid_line:
                data = json.loads(latest_valid_line)

                self.timestamp = data.get("timestamp", data.get("ts", 0))
                angles = data.get("angles", data.get("ang", []))
                errors = data.get("errors", data.get("err", []))

                for i in range(min(len(angles), self.num_sensors)):
                    angle = angles[i]
                    is_error = (angle is None) or (i < len(errors) and errors[i])
                    self.present_errors[i] = is_error
                    self.present_position[i] = np.nan if is_error else float(angle)

                self.chain_encoder = data.get("u_enc", 0)
                self.chain_encoder_button = data.get("u_btn", 0)
                self.joystick_x = data.get("jx", 0)
                self.joystick_y = data.get("jy", 0)
                self.joystick_button = data.get("jbtn", 0)

                return True
        except Exception:
            # Silently ignore JSON parse errors which are common on startup
            pass
        return False

    # ---------------------------------------------------------
    # PUBLIC API / GETTERS
    # ---------------------------------------------------------
    def fetch_present_status(self):
        """Fetch present status."""
        self._read_once()

    def fetch_present_status_bulk(self):
        """Fetch present status in bulk."""
        self._read_once()

    def cleanup(self):
        """Close connections."""
        try:
            if hasattr(self, "serial") and self.serial.is_open:
                self.serial.close()
        except Exception:
            pass

    def get_angles_degrees(self):
        """Get angels in degrees."""
        return self.present_position.copy()

    def get_angles_radians(self):
        """Get angels in radians."""
        return self.present_position * (np.pi / 180.0)

    def get_chain_encoder(self):
        """Get chain encoder."""
        return self.chain_encoder

    def get_chain_button(self):
        """Get chain button."""
        return self.chain_encoder_button

    def get_joystick_x_raw(self):
        """Get the raw value of joystick x."""
        return self.joystick_x

    def get_joystick_y_raw(self):
        """Get the raw value of joystick y."""
        return self.joystick_y

    def get_joystick_x(self):
        """Get the mapped joystick x value (-1.0 to 1.0)."""
        if self.joystick_x > (1 << 12) - 1:
            return (self.joystick_x - ((1 << 16) - 1)) / ((1 << 12) - 1)
        return self.joystick_x / ((1 << 12) - 1)

    def get_joystick_y(self):
        """Get the mapped joystick y value (-1.0 to 1.0)."""
        if self.joystick_y > (1 << 12) - 1:
            return (self.joystick_y - ((1 << 16) - 1)) / ((1 << 12) - 1)
        return self.joystick_y / ((1 << 12) - 1)

    def get_joystick_button(self):
        """Get joystick button."""
        return self.joystick_button

    def has_errors(self):
        """Return whether an error has occurred."""
        return np.any(self.present_errors)

    def get_error_status(self):
        """Get errors."""
        return self.present_errors.copy()
