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

"""Command-line interface utilities for OpenArm KER."""

import argparse
import re
import shutil
import sys
import time
from typing import Any, NoReturn

from .ker_stream import KERStream

FMT_TO_TYPE = {
    "I": "UINT32",
    "H": "UINT16",
    "B": "UINT8",
    "i": "INT32",
    "h": "INT16",
    "f": "FLOAT",
    "?": "BOOL",
}

# ANSI, used only on a real terminal so piped output stays clean.
CSI = "\033["
DIM, RED, YELLOW, GREEN, CYAN, BOLD, RESET = (
    "\033[2m",
    "\033[31m",
    "\033[33m",
    "\033[32m",
    "\033[36m",
    "\033[1m",
    "\033[0m",
)

_ANSI_RE = re.compile(r"\033\[[0-9;?]*[A-Za-z]")

DIAG_INTERVAL_S = 0.25

BAR_WIDTH = 13
ANGLE_LIMIT = 180.0

# Channel names for the standard 16-channel arm pair, so a row says which joint
# it is instead of leaving the reader to count. Falls back to plain numbering
# when the device reports a different channel count.
ARM_LAYOUT = {
    16: (
        (
            "RIGHT ARM",
            [
                (i, n)
                for i, n in enumerate(
                    ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "GRIP"]
                )
            ],
        ),
        (
            "LEFT ARM",
            [
                (i + 8, n)
                for i, n in enumerate(
                    ["J1", "J2", "J3", "J4", "J5", "J6", "J7", "GRIP"]
                )
            ],
        ),
    ),
}


class _Style:
    """Colour helpers that collapse to plain text when not on a terminal."""

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, text: str, *codes: str) -> str:
        """Wrap text in the given ANSI codes, or return it unchanged."""
        if not self.enabled or not codes:
            return text
        return "".join(codes) + text + RESET


def _bar(value: float, style: _Style) -> str:
    """Draw an angle as a bar filled outward from zero.

    A single marker sliding along a track is hard to read down a column of
    sixteen rows: filling from the centre makes sign and magnitude visible at a
    glance, and lines the rows up against each other.
    """
    centre = BAR_WIDTH // 2
    if value != value:  # NaN
        return style("[" + "?" * BAR_WIDTH + "]", DIM)

    clamped = max(-ANGLE_LIMIT, min(ANGLE_LIMIT, value))
    pos = int(round(centre + clamped / ANGLE_LIMIT * centre))
    pos = max(0, min(BAR_WIDTH - 1, pos))

    lo, hi = min(centre, pos), max(centre, pos)
    cells = []
    for i in range(BAR_WIDTH):
        if lo <= i <= hi and not (i == centre and pos == centre):
            cells.append("=")
        elif i == centre:
            cells.append(style("|", DIM))
        else:
            cells.append(" ")
    return style("[", DIM) + "".join(cells) + style("]", DIM)


# Fixed field widths. The column layout must not depend on the *content* of a
# row: letting it grow for a status word made the whole right-hand column jump
# sideways the moment a fault appeared, which is exactly when the reader wants
# the numbers to stay put.
_W_LABEL, _W_ANGLE, _W_STATE = 4, 8, 6
_W_ROW_BAR = _W_LABEL + 1 + _W_ANGLE + 1 + (BAR_WIDTH + 2) + 1 + _W_STATE
_W_ROW_PLAIN = _W_LABEL + 1 + _W_ANGLE + 1 + _W_STATE
_COL_GAP = 3


def _row(
    name: str,
    angle: float,
    errored: bool,
    style: _Style,
    show_bar: bool,
    state_name: str | None = None,
) -> str:
    """Format one channel: name, angle, optional bar, and its state.

    When diagnostics are available the state column names the actual reason
    (STALE / HELD / SUSPECT) rather than a generic "frozen", which is the
    difference between "something is wrong" and knowing what to go and check.
    """
    label = style(f"{name:<{_W_LABEL}}", BOLD)

    if errored:
        # Keep the last known angle visible - it is what the follower is being
        # commanded to - but say plainly that it is not a live reading.
        angle_txt = style(f"{angle:>+{_W_ANGLE}.2f}", DIM)
        word = (state_name or "FROZEN")[:_W_STATE]
        state = style(f"{word:<{_W_STATE}}", RED, BOLD)
    else:
        angle_txt = f"{angle:>+{_W_ANGLE}.2f}"
        # Nothing to say about a healthy channel. Sixteen rows of "ok" only
        # make the rows that do matter harder to find.
        state = " " * _W_STATE

    parts = [label, angle_txt]
    if show_bar:
        parts.append(_bar(angle, style))
    parts.append(state)
    return " ".join(parts)


def _fmt_uptime(seconds: float) -> str:
    """Render elapsed time as 42s / 1m30s / 2h05m."""
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m{total % 60:02d}s"
    return f"{total // 3600}h{(total % 3600) // 60:02d}m"


def _render(
    stream: KERStream, data: dict[str, Any], style: _Style, columns: int
) -> list[str]:
    """Build the live view for one frame."""
    stats = stream.stats()
    meta = stream.metadata
    # Firmware built with STREAM_COMPACT_ANGLES sends hundredths of a degree as
    # int16, which halves the frame and gets it under one USB packet.
    angles = data.get("angles")
    if angles is None:
        compact = data.get("angles_cd")
        angles = [v / 100.0 for v in compact] if compact else []
    count = len(angles)
    diag = None

    # Newer firmware packs the flags into one 16-bit mask; sixteen booleans
    # cost sixteen bytes per frame to carry sixteen bits. Accept either.
    mask = data.get("error_mask")
    if mask is None:
        errors = data.get("errors") or []

        def is_err(i: int) -> bool:
            return bool(errors[i]) if i < len(errors) else False
    else:

        def is_err(i: int) -> bool:
            return bool(mask >> i & 1)

    ch_diag = (diag or {}).get("channels") or []

    def state_of(i: int) -> str | None:
        return ch_diag[i]["state"] if i < len(ch_diag) else None

    width = max(40, min(columns - 1, 100))

    # Widest layout first, dropping the bar and then the second column as the
    # terminal narrows, so the numbers never wrap.
    show_bar = width >= 2 * _W_ROW_BAR + _COL_GAP
    two_col = show_bar or width >= 2 * _W_ROW_PLAIN + _COL_GAP
    if not two_col:
        show_bar = width >= _W_ROW_BAR

    lines = []
    title = f"OpenArm KER   fw {meta.get('fw', '?')}  hw {meta.get('hw', '?')}"
    uptime = f"up {_fmt_uptime(stats['elapsed'])}"
    lines.append(
        style(title, BOLD)
        + " " * max(1, width - len(title) - len(uptime))
        + style(uptime, DIM)
    )

    # A dropped link is the loudest thing that can be happening; say so before
    # the numbers, and make clear the angles below are frozen from before it.
    if not stats.get("link_up", True):
        lines.append(
            style(
                f"LINK DOWN - reconnecting ({stats.get('link_down_for', 0):.0f}s)"
                "   values below are stale",
                RED,
                BOLD,
            )
        )

    lost, badsum = stats["lost"], stats["checksum_errors"]
    bits = [
        style(f"{stats['rate_hz']:.1f} Hz", GREEN, BOLD),
        f"recv {stats['received']}",
        style(f"lost {lost}", RED, BOLD) if lost else style("lost 0", DIM),
        style(f"badsum {badsum}", YELLOW, BOLD) if badsum else style("badsum 0", DIM),
    ]
    reconnects = stats.get("reconnects", 0)
    if reconnects:
        # A link that recovers looks identical to one that never failed unless
        # the recoveries are counted; this is the signal that it is marginal.
        bits.append(style(f"reconnects {reconnects}", YELLOW, BOLD))
    lines.append("  ".join(bits))
    lines.append("")

    groups = ARM_LAYOUT.get(count)
    if groups is None:
        groups = (("CHANNELS", [(i, f"{i + 1}") for i in range(count)]),)
        two_col = False

    cell_w = _W_ROW_BAR if show_bar else _W_ROW_PLAIN

    if two_col and len(groups) == 2:
        (lname, litems), (rname, ritems) = groups
        lines.append(style(f"{lname:<{cell_w + _COL_GAP}}{rname}", BOLD, CYAN))
        for (li, ln), (ri, rn) in zip(litems, ritems):
            left = _row(ln, angles[li], is_err(li), style, show_bar, state_of(li))
            right = _row(rn, angles[ri], is_err(ri), style, show_bar, state_of(ri))
            lines.append(
                left.rstrip()
                + " " * (cell_w - len(_strip(left).rstrip()) + _COL_GAP)
                + right.rstrip()
            )
    else:
        for gname, items in groups:
            lines.append(style(gname, BOLD, CYAN))
            for i, n in items:
                lines.append(
                    _row(n, angles[i], is_err(i), style, show_bar, state_of(i)).rstrip()
                )

    lines.append("")

    if diag:
        # The chain round-trip is the one number that says whether the 1 ms
        # budget is really met; everything else about the timing is estimate.
        chain = (
            f"chain {diag['chain_us']}us"
            f" (min {diag['chain_us_min']} / max {diag['chain_us_max']})"
        )
        over = diag["chain_us_max"] > 1000
        lines.append(
            (style(chain, YELLOW, BOLD) if over else style(chain, DIM))
            + style(f"   policy {diag['fault_policy']}", DIM)
        )
        lines.append(
            style(
                f"bus  frame_err {diag['frame_errors']}"
                f"  unknown_id {diag['unknown_id']}"
                f"  heal {diag.get('heal_triggers', 0)}"
                f"   usb sent {diag['usb_sent']}"
                f" drop {diag['usb_dropped']} stall {diag['usb_stall']}",
                DIM,
            )
        )
        lines.append("")

    bad = [i for i in range(count) if is_err(i)]
    if bad:
        # The RS-485 chain is a bucket brigade: a device replies only after
        # hearing the one before it, so a single silent module starves every
        # module after it. When the failures run contiguously to the last
        # channel, the first one is the actual fault and the rest are its
        # consequences - worth saying, instead of printing ten identical lines.
        cascade = len(bad) > 2 and bad == list(range(bad[0], count))
        shown = bad[:1] if cascade else bad[:4]

        for i in shown:
            head = f"! {_channel_name(i, count):<12}"
            if i < len(ch_diag):
                c = ch_diag[i]
                detail = (
                    f"{c['state']:<8} crc_err {c['crc_errors']:<6}"
                    f" rx {c['rx_count']:<9} age {c['age_ms']}ms"
                    f"  proto {c['proto']}"
                )
            else:
                detail = "not live (value frozen)"
            lines.append(style(head + detail, RED, BOLD))

        if cascade:
            first = _channel_name(bad[0], count)
            lines.append(
                style(
                    f"  -> CH{bad[0] + 1} and all {len(bad) - 1} channels after it."
                    f" Check {first} first: downstream never gets its trigger.",
                    YELLOW,
                )
            )
        elif len(bad) > len(shown):
            rest = ", ".join(_channel_name(i, count) for i in bad[len(shown) :])
            lines.append(style(f"  -> also {rest}", RED))
    elif diag and diag.get("host_lost"):
        lines.append(
            style("! PC not reading - the device stopped streaming", RED, BOLD)
        )
    else:
        lines.append(style("all channels live", DIM))

    lines.append(style("Ctrl+C to stop", DIM))
    return lines


def _channel_name(index: int, count: int) -> str:
    """Return a human name like 'RIGHT J7' for a channel index."""
    groups = ARM_LAYOUT.get(count)
    if groups:
        for gname, items in groups:
            for i, n in items:
                if i == index:
                    return f"{gname.split()[0]} {n}"
    return f"CH{index + 1}"


def _strip(text: str) -> str:
    """Return text with ANSI escapes removed, for width calculations."""
    return _ANSI_RE.sub("", text)


def _print_schema(stream: KERStream) -> None:
    """Print the field schema the device reported."""
    fields = stream.fields
    if not fields:
        return
    print("  Field           Type      Count")
    print("  " + "-" * 32)
    for field in fields:
        type_name = FMT_TO_TYPE.get(field.get("format", ""), "?")
        print(f"  {field['key']:<15} {type_name:<9} {field['count']:>5}")


def _cmd_ping(stream: KERStream, vid: int, pid: int) -> int:
    """Fetch and print device metadata and the stream schema."""
    metadata = stream.ping_only()
    if not metadata:
        print(
            "Error: no response from the device.\n"
            "  - is it powered and enumerated?  (lsusb | grep 303a)\n"
            "  - does your user have permission for it?  (udev rule)",
            file=sys.stderr,
        )
        return 1

    print(f"OpenArm KER   USB {vid:04x}:{pid:04x}\n")
    print(f"  Firmware   {metadata.get('fw', '?')}")
    print(f"  Hardware   {metadata.get('hw', '?')}")
    print(f"  Updated    {metadata.get('updated', '?')}\n")

    size = stream.packet_size
    if size:
        print(
            f"  Stream packet   {size} bytes"
            f"  (2 header + {size - 3} payload + 1 checksum)\n"
        )
    _print_schema(stream)
    return 0


def _cmd_stream(stream: KERStream) -> int:
    """Show a live view of the incoming stream."""
    interactive = sys.stdout.isatty()
    style = _Style(interactive)

    try:
        stream.connect()
    except Exception as exc:
        print(f"Error: cannot connect: {exc}", file=sys.stderr)
        return 1

    if interactive:
        sys.stdout.write(CSI + "2J" + CSI + "H" + CSI + "?25l")

    last_diag = 0.0

    try:
        while stream.is_connected:
            data = stream.latest()
            if data is None:
                time.sleep(0.02)
                continue

            # Diagnostics are polled, so the rate is ours to choose: often
            # enough to be useful, rare enough to stay invisible beside the
            # data stream.
            now = time.monotonic()
            if now - last_diag >= DIAG_INTERVAL_S:
                last_diag = now
                try:
                    stream.request_diagnostics()
                except Exception:
                    pass

            columns = shutil.get_terminal_size((80, 24)).columns
            lines = _render(stream, data, style, columns)

            if interactive:
                out = [CSI + "H"]
                for line in lines:
                    out.append(line + CSI + "K\n")
                out.append(CSI + "J")
                sys.stdout.write("".join(out))
            else:
                stats = stream.stats()
                sys.stdout.write(
                    f"{stats['rate_hz']:.1f}Hz recv={stats['received']} "
                    f"lost={stats['lost']} badsum={stats['checksum_errors']} "
                    f"angles={[f'{a:.2f}' for a in (data.get('angles') or [])]}\n"
                )
            sys.stdout.flush()
            time.sleep(0.05)

        print("\nStream ended: the session was closed.")
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        if interactive:
            sys.stdout.write(CSI + "?25h")
            sys.stdout.flush()
        stream.close()
        stats = stream.stats()
        print(
            f"\nreceived {stats['received']} frames in {stats['elapsed']:.1f} s"
            f"  (avg {stats.get('avg_rate_hz', 0.0):.1f} Hz)"
            f"   lost {stats['lost']}   checksum errors {stats['checksum_errors']}"
            f"   reconnects {stats.get('reconnects', 0)}"
        )


def main() -> NoReturn | None:
    """Run the KER CLI.

    Provides diagnostic utilities such as pinging the device and raw streaming.
    """
    parser = argparse.ArgumentParser(
        description="KERStream Command-Line Interface (CLI) Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["ping", "stream"],
        help="Command to execute: 'ping' to fetch schema and device metadata, "
        "'stream' to show a live view of the incoming data.",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="usb",
        choices=["usb", "serial"],
        help="Transport protocol connection type.",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port path (only applicable when transport is set to 'serial').",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=2000000,
        help="Baud rate speed (only applicable when transport is set to 'serial').",
    )
    parser.add_argument(
        "--vid", type=lambda v: int(v, 0), default=0x303A, help="USB vendor id."
    )
    parser.add_argument(
        "--pid", type=lambda v: int(v, 0), default=0x4002, help="USB product id."
    )
    args = parser.parse_args()

    stream = KERStream(
        transport=args.transport,
        port=args.port,
        baud=args.baud,
        vid=args.vid,
        pid=args.pid,
    )

    if args.command == "ping":
        sys.exit(_cmd_ping(stream, args.vid, args.pid))
    sys.exit(_cmd_stream(stream))


if __name__ == "__main__":
    main()
