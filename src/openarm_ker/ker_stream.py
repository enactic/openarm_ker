"""Module for streaming data from KER devices via USB or Serial transport.

This module provides the `KERStream` class to handle protocol handshaking,
schema fetching, and continuous asynchronous data reading.
"""

import struct
import threading
import time
from queue import Queue, Empty
from typing import Any

# =====================================================
# Protocol Constants
# =====================================================
HEADER_STREAM = b"\xa5\x5a"
HEADER_PING = b"\xa5\x50"

CMD_PING = b"\x00"
CMD_STANDBY = b"\x01"
CMD_STREAM = b"\x02"

TYPE_MAP = {
    0: ("I", 4, "UINT32"),
    1: ("H", 2, "UINT16"),
    2: ("B", 1, "UINT8"),
    3: ("i", 4, "INT32"),
    4: ("h", 2, "INT16"),
    5: ("f", 4, "FLOAT"),
    6: ("?", 1, "BOOL"),
}


def _verify_checksum(packet: bytes) -> bool:
    """Verify the checksum of a given byte packet."""
    cs = 0
    for b in packet[2:-1]:
        cs ^= b
    return cs == packet[-1]


class KERStream:
    """Handler for KER device communication.

    Establishes a connection (USB/Serial), retrieves the schema,
    and maintains a background thread to read the latest streaming data.
    """

    def __init__(
        self,
        transport: str = "usb",
        port: str = "/dev/ttyACM0",
        baud: int = 2000000,
        vid: int = 0x303A,
        pid: int = 0x4002,
        timeout: float = 0.01,
    ):
        """Initialize the stream configuration."""
        self._transport = transport
        self._port = port
        self._baud = baud
        self._vid = vid
        self._pid = pid
        self._timeout = timeout

        self._dev = None
        self._ep_in = None
        self._ep_out = None
        self._serial = None
        self._buf = bytearray()

        self.metadata = {}
        self._fields = []
        self._fmt = ""
        self._packet_size = 0

        # Read thread
        self._latest_data = None
        self._lock = threading.Lock()
        self._queue = Queue(maxsize=2)
        self._running = False
        self._thread = None

    # --------------------------------------------------
    # Connect
    # --------------------------------------------------
    def connect(self):
        """Establish connection to the hardware and start the read thread."""
        if self._transport == "usb":
            self._connect_usb()
        elif self._transport == "serial":
            self._connect_serial()
        else:
            raise ValueError(f"Unknown transport: {self._transport}")

        self._ping_and_fetch_schema()

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _connect_usb(self):
        import usb.core
        import usb.util

        dev = usb.core.find(idVendor=self._vid, idProduct=self._pid)
        if dev is None:
            raise RuntimeError(
                f"USB device {self._vid:#06x}:{self._pid:#06x} not found"
            )

        if dev.is_kernel_driver_active(0):
            dev.detach_kernel_driver(0)

        dev.set_configuration()
        cfg = dev.get_active_configuration()
        intf = cfg[(0, 0)]

        self._ep_in = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress)
            == usb.util.ENDPOINT_IN,
        )
        self._ep_out = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress)
            == usb.util.ENDPOINT_OUT,
        )
        self._dev = dev

        try:
            self._dev.write(self._ep_out.bEndpointAddress, CMD_STANDBY)
        except Exception:
            pass

        flush_end = time.time() + 0.2
        while time.time() < flush_end:
            try:
                self._dev.read(self._ep_in.bEndpointAddress, 512, timeout=10)
            except usb.core.USBError:
                break

    def _connect_serial(self):
        import serial

        self._serial = serial.Serial(
            port=self._port, baudrate=self._baud, timeout=self._timeout
        )

        try:
            self._serial.write(CMD_STANDBY)
            time.sleep(0.05)
        except Exception:
            pass
        self._serial.reset_input_buffer()

    # --------------------------------------------------
    # Connection Status Property
    # --------------------------------------------------
    @property
    def is_connected(self) -> bool:
        """Return whether the stream is currently connected and running."""
        return self._running

    # --------------------------------------------------
    # Command: Ping Only
    # --------------------------------------------------
    def ping_only(self) -> dict[str, Any] | None:
        """Connect temporarily to fetch device metadata.

        Sends a PING command to fetch device metadata and fields schema,
        then cleanly disconnects without starting the stream thread.

        Returns:
            Dictionary containing metadata, or None if it fails.

        """
        try:
            if self._transport == "usb":
                self._connect_usb()
            elif self._transport == "serial":
                self._connect_serial()
            else:
                raise ValueError(f"Unknown transport: {self._transport}")

            self._ping_and_fetch_schema()
            return self.metadata

        except Exception as e:
            print(f"[Ping Failed] Error: {e}")
            return None

        finally:
            self.close()

    # --------------------------------------------------
    # Handshake & Schema parsing
    # --------------------------------------------------
    def _ping_and_fetch_schema(self):
        self._buf.clear()

        start_time = time.time()
        last_ping = 0

        while time.time() - start_time < 3.0:
            if time.time() - last_ping >= 0.5:
                try:
                    self.send_command(CMD_PING)
                except Exception:
                    pass
                last_ping = time.time()

            chunk = self._read_raw(512)
            if chunk:
                self._buf.extend(chunk)

            idx = self._buf.find(HEADER_PING)
            if idx != -1:
                self._buf = self._buf[idx:]
                if self._parse_ping_response():
                    return

        raise TimeoutError(
            f"Failed to fetch schema. Received buffer: {self._buf.hex()}"
        )

    def _parse_ping_response(self) -> bool:
        if len(self._buf) < 47:
            return False

        pos = 2
        fw = self._buf[pos : pos + 16].decode("utf-8", "ignore").rstrip("\x00")
        pos += 16
        hw = self._buf[pos : pos + 16].decode("utf-8", "ignore").rstrip("\x00")
        pos += 16
        updated = self._buf[pos : pos + 12].decode("utf-8", "ignore").rstrip("\x00")
        pos += 12

        self.metadata = {"fw": fw, "hw": hw, "updated": updated}

        field_count = self._buf[pos]
        pos += 1

        if len(self._buf) < pos + (field_count * 18):
            return False

        self._fields = []
        fmt_str = "<"

        for _ in range(field_count):
            key = self._buf[pos : pos + 16].decode("utf-8", "ignore").rstrip("\x00")
            pos += 16
            type_id = self._buf[pos]
            pos += 1
            count = self._buf[pos]
            pos += 1

            fmt_char, _, _ = TYPE_MAP.get(type_id, ("x", 1, "UNKNOWN"))
            self._fields.append({"key": key, "count": count, "format": fmt_char})
            fmt_str += f"{count}{fmt_char}" if count > 1 else fmt_char

        self._fmt = fmt_str
        self._packet_size = 2 + struct.calcsize(self._fmt) + 1

        self._buf.clear()
        return True

    # --------------------------------------------------
    # Read thread
    # --------------------------------------------------
    def _enqueue(self, d: dict[str, Any]) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Empty:
                pass
        self._queue.put_nowait(d)

    def _read_loop(self):
        while self._running:
            packets = self._read_all()
            for d in packets:
                with self._lock:
                    self._latest_data = d
                self._enqueue(d)
            if not packets:
                time.sleep(0.001)

    def latest(self) -> dict[str, Any] | None:
        """Retrieve the most recently parsed data packet.

        Returns:
            A dictionary of parsed fields, or None if no data is available yet.

        """
        with self._lock:
            return self._latest_data

    def recv(self) -> dict[str, Any] | None:
        """Get next packet from queue.

        Returns:
            A dictionary of parsed fields, or None if queue is empty.

        """
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    # --------------------------------------------------
    # Internal read
    # --------------------------------------------------
    def _read_all(self) -> list[dict[str, Any]]:
        chunk = self._read_raw(4096)
        if chunk:
            self._buf.extend(chunk)

        results = []

        while len(self._buf) >= self._packet_size:
            idx = self._buf.find(HEADER_STREAM)
            if idx == -1:
                self._buf.clear()
                break
            if idx > 0:
                self._buf = self._buf[idx:]
            if len(self._buf) < self._packet_size:
                break

            packet = bytes(self._buf[: self._packet_size])
            self._buf = self._buf[self._packet_size :]

            if not _verify_checksum(packet):
                continue

            results.append(self._parse_stream_packet(packet))

        return results

    def _read_raw(self, size) -> bytes:
        if self._transport == "usb":
            import usb.core

            if self._dev is None:
                return b""

            try:
                return bytes(
                    self._dev.read(self._ep_in.bEndpointAddress, size, timeout=20)
                )
            except usb.core.USBError as e:
                if e.errno in (110, 116) or "timeout" in str(e).lower():
                    return b""
                print("\n[Disconnected] USB disconnected... now lose connect ,,,,")
                self._running = False
                return b""
            except Exception as e:
                print(f"\n[Error] Unexpected error: {e}")
                self._running = False
                return b""
        else:
            import serial

            if self._serial is None:
                return b""

            try:
                waiting = self._serial.in_waiting
                if waiting > 0:
                    return self._serial.read(min(waiting, size))
                else:
                    return b""
            except serial.SerialException:
                print("\n[Disconnected] Serial disconnected... now lose connect ,,,,")
                self._running = False
                return b""
            except Exception as e:
                print(f"\n[Error] Unexpected error: {e}")
                self._running = False
                return b""

    def _parse_stream_packet(self, packet: bytes) -> dict[str, Any]:
        unpacked = struct.unpack(self._fmt, packet[2:-1])

        data = {}
        index = 0
        for f in self._fields:
            key = f["key"]
            count = f["count"]
            if count == 1:
                data[key] = unpacked[index]
            else:
                data[key] = list(unpacked[index : index + count])
            index += count

        return data

    # --------------------------------------------------
    # Send / Close
    # --------------------------------------------------
    def send_command(self, cmd: bytes):
        """Send a raw byte command to the connected device."""
        if self._transport == "usb":
            if self._ep_out is None:
                raise RuntimeError("USB not connected")
            self._dev.write(self._ep_out.bEndpointAddress, cmd)
        else:
            if self._serial is None:
                raise RuntimeError("Serial not connected")
            self._serial.write(cmd)

    def close(self):
        """Terminate the stream thread and release hardware resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._serial is not None:
            self._serial.close()
            self._serial = None

        if self._transport == "usb" and self._dev is not None:
            import usb.util

            try:
                usb.util.dispose_resources(self._dev)
            except Exception:
                pass

        self._dev = None
        self._ep_in = None
        self._ep_out = None

    def __enter__(self):
        """Enter context manager, automatically connecting to the device."""
        self.connect()
        return self

    def __exit__(self, *args):
        """Exit context manager, automatically closing the connection."""
        self.close()
