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
CMD_ZERO_ALL = b"\x03"
CMD_DIAG = b"\x05"

# Diagnostics are polled rather than streamed, so they cost nothing until asked
# for. Layouts below match buildDiagPayload() in the M5 firmware.
# Window over which the frame rate is measured.
RATE_WINDOW_S = 0.5

RECONNECT_BACKOFF_MIN_S = 0.5
RECONNECT_BACKOFF_MAX_S = 4.0

DIAG_HEAD_FMT = "<BBBBHHHbBfHHHIIIIIII"
DIAG_CH_FMT = "<BBHIH"

CH_STATE_NAMES = {0: "INIT", 1: "OK", 2: "STALE", 3: "HELD", 4: "SUSPECT"}
PROTO_NAMES = {0: "auto", 1: "crc6", 2: "legacy"}
FAULT_POLICY_NAMES = {0: "HOLD", 1: "FREEZE_ALL", 2: "STOP"}

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


def _is_usb_timeout(exc: Exception) -> bool:
    """Return whether a USBError is just a read timeout.

    A timeout is the normal way a read ends when the device has nothing to say,
    so it must not be treated like a transport failure.
    """
    # 110 = ETIMEDOUT, 116 = ESTALE (libusb reports either depending on backend)
    return getattr(exc, "errno", None) in (110, 116) or "timeout" in str(exc).lower()


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
        auto_resume: bool = False,
    ):
        """Initialize the stream configuration.

        auto_resume re-sends CMD_STREAM after the link comes back. It is off by
        default on purpose: the device keeps tracking while the link is down, so
        resuming hands the follower whatever pose the leader is in now, and it
        will move there in one step. Reconnecting the link is safe; deciding
        that it is safe to start moving again is the application's call.
        """
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

        # What we took from the OS and therefore have to give back on close().
        # Leaving either of these behind is what made the device look broken
        # until it was physically unplugged.
        self._claimed_interface = None
        self._detached_kernel_driver = False

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

        # Link statistics. The device sends a "seq" field, which is the only way
        # to tell "the device sent nothing" apart from "the frame was lost on the
        # way here"; without it a stuttering stream has no explanation.

        # The transport can drop and come back without the session ending, so
        # "is the session alive" (_running) and "is the link usable right now"
        # (_link_up) are separate. A fatal read error takes the link down; the
        # read thread then keeps trying to bring it back.
        self._link_up = False
        self._link_down_since = None
        self._reconnects = 0
        self._auto_resume = auto_resume

        self._last_seq = None
        self._lost_frames = 0
        self._checksum_errors = 0
        self._received_frames = 0
        self._started_at = None

        # Rate is measured over a short window, not averaged since connect().
        # A cumulative average starts near zero - streaming only begins when the
        # caller asks for it, well after connect() - and then creeps upwards for
        # minutes, which reads as the device speeding up when nothing changed.
        self._rate_hz = 0.0
        self._rate_mark_t = 0.0
        self._rate_mark_n = 0

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

        self._link_up = True
        self._running = True
        self._started_at = time.time()
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

        # Only detach a driver that is actually bound. A vendor-class interface
        # normally has none, and an unbalanced detach is exactly what leaves the
        # device unusable until it is replugged, so don't do it speculatively.
        self._detached_kernel_driver = False
        try:
            if dev.is_kernel_driver_active(0):
                dev.detach_kernel_driver(0)
                self._detached_kernel_driver = True
        except (NotImplementedError, usb.core.USBError):
            pass

        # SET_CONFIGURATION resets every endpoint on the device, so only send it
        # if the device is not configured yet. Re-sending it on each connect
        # disturbs a device that was already working.
        try:
            configured = dev.get_active_configuration() is not None
        except usb.core.USBError:
            configured = False
        if not configured:
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
        if self._ep_in is None or self._ep_out is None:
            raise RuntimeError("USB device has no bulk IN/OUT endpoint pair")

        self._dev = dev

        # Claim explicitly. pyusb would claim implicitly on the first transfer,
        # but then a stale claim from a previous run surfaces as a confusing
        # error in the middle of streaming instead of here, and there is nothing
        # for release_interface() to release on the way out.
        try:
            usb.util.claim_interface(dev, intf.bInterfaceNumber)
            self._claimed_interface = intf.bInterfaceNumber
        except usb.core.USBError as exc:
            raise RuntimeError(
                f"Cannot claim USB device {self._vid:#06x}:{self._pid:#06x}: {exc}. "
                "Another process is probably still holding it - check for a "
                "leftover node from a previous run."
            ) from exc

        # A process that died mid-transfer can leave an endpoint halted, and that
        # state lives in the device, not in the process: it survives until it is
        # cleared or the device is reset. Clearing it here is what removes the
        # need to pull the cable.
        for endpoint in (self._ep_in, self._ep_out):
            try:
                dev.clear_halt(endpoint.bEndpointAddress)
            except usb.core.USBError:
                pass

        try:
            self._dev.write(self._ep_out.bEndpointAddress, CMD_STANDBY)
        except usb.core.USBError:
            pass

        # Drain anything the previous session left in flight. A read timeout
        # means the pipe is empty and the drain is done; any other error is a
        # real transport problem and is not worth retrying for 200 ms.
        flush_end = time.time() + 0.2
        while time.time() < flush_end:
            try:
                if not self._dev.read(self._ep_in.bEndpointAddress, 512, timeout=10):
                    break
            except usb.core.USBError as exc:
                if not _is_usb_timeout(exc):
                    print(f"[Warning] Error while draining stale data: {exc}")
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
        """Return whether the session is alive.

        This stays true while the transport is down and being retried: a
        transient failure is no longer the end of the session.
        """
        return self._running

    @property
    def is_link_up(self) -> bool:
        """Return whether the transport is usable right now."""
        return self._link_up

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

    def _mark_link_down(self, reason: str) -> None:
        """Record that the transport failed; the read loop will retry it."""
        if self._link_up:
            self._link_up = False
            self._link_down_since = time.time()
            print(f"\n[Disconnected] {reason} - retrying")

    def _update_rate(self) -> None:
        """Refresh the windowed frame rate."""
        now = time.monotonic()
        if self._rate_mark_t == 0.0:
            self._rate_mark_t, self._rate_mark_n = now, self._received_frames
            return
        dt = now - self._rate_mark_t
        if dt < RATE_WINDOW_S:
            return
        instant = (self._received_frames - self._rate_mark_n) / dt
        # Light smoothing: enough to stop the last digit flickering, not enough
        # to hide a real drop.
        self._rate_hz = (
            instant if self._rate_hz == 0.0 else (0.6 * self._rate_hz + 0.4 * instant)
        )
        self._rate_mark_t, self._rate_mark_n = now, self._received_frames

    def _read_loop(self):
        backoff = RECONNECT_BACKOFF_MIN_S
        while self._running:
            self._update_rate()
            if not self._link_up:
                # The transport is gone. Keep the session alive and keep trying:
                # a cable knock or a re-enumeration used to end the session
                # permanently, and nothing ever tried again.
                if self._reconnect():
                    backoff = RECONNECT_BACKOFF_MIN_S
                else:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)
                continue

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

        while self._packet_size and len(self._buf) >= self._packet_size:
            idx = self._buf.find(HEADER_STREAM)
            if idx == -1:
                # Keep a trailing first header byte: its partner may not have
                # arrived yet.
                keep = 1 if self._buf[-1:] == HEADER_STREAM[:1] else 0
                del self._buf[: len(self._buf) - keep]
                break
            if idx > 0:
                del self._buf[:idx]
            if len(self._buf) < self._packet_size:
                break

            packet = bytes(self._buf[: self._packet_size])

            if not _verify_checksum(packet):
                # The header was a coincidence inside a payload - float angle
                # data hits 0xA5 0x5A on about 0.05% of frames, i.e. roughly
                # twice a second at 1 kHz. Skip only the two header bytes:
                # consuming a whole packet's worth would also discard the real
                # frame starting inside it, turning a false positive into a
                # genuinely lost sample.
                del self._buf[:2]
                self._checksum_errors += 1
                continue

            del self._buf[: self._packet_size]
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
                if _is_usb_timeout(e):
                    return b""
                self._mark_link_down(f"USB error: {e}")
                return b""
            except Exception as e:
                self._mark_link_down(f"unexpected USB error: {e}")
                return b""
        else:
            import serial

            if self._serial is None:
                return b""

            try:
                # Block on one byte, then take whatever else is already
                # buffered. Checking in_waiting and returning empty instead
                # pushes the wait into _read_loop's time.sleep(), which costs a
                # syscall per poll and adds up to a millisecond of latency to
                # every frame - on a link whose frames are 1 ms apart.
                first = self._serial.read(1)
                if not first:
                    return b""
                extra = self._serial.in_waiting
                if extra and size > 1:
                    return first + self._serial.read(min(extra, size - 1))
                return first
            except serial.SerialException as e:
                self._mark_link_down(f"serial error: {e}")
                return b""
            except Exception as e:
                self._mark_link_down(f"unexpected serial error: {e}")
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

        self._received_frames += 1
        self._account_sequence(data.get("seq"))
        return data

    def _account_sequence(self, seq: int | None) -> None:
        """Count frames that never arrived, using the device's frame counter."""
        if seq is None:
            return
        if self._last_seq is not None:
            gap = (seq - self._last_seq - 1) & 0xFFFFFFFF
            # A huge gap is a device restart (the counter went back to zero),
            # not a million lost frames.
            if 0 < gap < (1 << 31):
                self._lost_frames += gap
        self._last_seq = seq

    def stats(self) -> dict[str, Any]:
        """Return link statistics for diagnostics.

        Returns:
            received: frames parsed successfully
            lost: frames the device sent that never arrived (from "seq")
            checksum_errors: frames rejected, including false header matches
            elapsed: seconds since connect()
            rate_hz: current frames per second, over a short window
            avg_rate_hz: frames per second averaged since connect()
            reconnects: times the transport was recovered after a failure
            link_up: whether the transport is usable right now
            link_down_for: seconds the transport has been down, else 0

        """
        elapsed = (time.time() - self._started_at) if self._started_at else 0.0
        return {
            "received": self._received_frames,
            "lost": self._lost_frames,
            "checksum_errors": self._checksum_errors,
            "elapsed": elapsed,
            # Current rate, measured over the last RATE_WINDOW_S.
            "rate_hz": self._rate_hz,
            # Frames per second since connect(), which includes the idle time
            # before streaming was started - useful for a session total, not
            # for watching a live link.
            "avg_rate_hz": (self._received_frames / elapsed) if elapsed > 0 else 0.0,
            "reconnects": self._reconnects,
            "link_up": self._link_up,
            "link_down_for": (
                time.time() - self._link_down_since if self._link_down_since else 0.0
            ),
        }

    @property
    def fields(self) -> list[dict[str, Any]]:
        """Return the field schema reported by the device."""
        return list(self._fields)

    @property
    def packet_size(self) -> int:
        """Return the stream packet size in bytes, including header and checksum."""
        return self._packet_size

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

    def _teardown_transport(self):
        """Release the transport, leaving the session and thread untouched.

        Everything taken in _connect_usb() is handed back here in reverse order.
        Skipping any of it is what used to make a replug necessary.
        """
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

        if self._transport == "usb" and self._dev is not None:
            import usb.util

            if self._claimed_interface is not None:
                try:
                    usb.util.release_interface(self._dev, self._claimed_interface)
                except Exception:
                    pass
                self._claimed_interface = None

            if self._detached_kernel_driver:
                try:
                    self._dev.attach_kernel_driver(0)
                except Exception:
                    pass
                self._detached_kernel_driver = False

            try:
                usb.util.dispose_resources(self._dev)
            except Exception:
                pass

        self._dev = None
        self._ep_in = None
        self._ep_out = None
        self._buf.clear()

    def _reconnect(self) -> bool:
        """Try once to bring the link back. Returns whether it succeeded."""
        self._teardown_transport()
        try:
            if self._transport == "usb":
                self._connect_usb()
            else:
                self._connect_serial()
            self._ping_and_fetch_schema()
        except Exception:
            return False

        # Frames missed while the link was down are an outage, not packet loss;
        # counting them as lost would bury the number that matters.
        self._last_seq = None
        self._rate_hz = 0.0
        self._rate_mark_t = 0.0
        self._link_up = True
        self._reconnects += 1
        down_for = time.time() - self._link_down_since if self._link_down_since else 0.0
        self._link_down_since = None
        print(
            f"[Reconnected] link restored after {down_for:.1f}s "
            f"(reconnect #{self._reconnects})"
        )
        if self._auto_resume:
            try:
                self.send_command(CMD_STREAM)
            except Exception:
                pass
        else:
            print("[Reconnected] streaming NOT resumed - press START to resume")
        return True

    def close(self):
        """Terminate the stream thread and release hardware resources.

        Everything taken in _connect_usb() is handed back here, in reverse
        order, so the next process finds the device in the same state we did.
        Skipping this is what made a replug look necessary.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Leave the device idle rather than streaming into a closed pipe.
        try:
            self.send_command(CMD_STANDBY)
        except Exception:
            pass

        self._teardown_transport()
        self._link_up = False

    def __enter__(self):
        """Enter context manager, automatically connecting to the device."""
        self.connect()
        return self

    def __exit__(self, *args):
        """Exit context manager, automatically closing the connection."""
        self.close()
