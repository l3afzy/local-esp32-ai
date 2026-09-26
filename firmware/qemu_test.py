"""Run the built firmware in Espressif's ESP32 emulator and ask it questions.

    cd firmware && pio run -e esp32dev
    python qemu_test.py --qemu /path/to/qemu-system-xtensa < questions.txt
    pio run -e esp32-s3-16mb && python qemu_test.py --chip esp32s3 --qemu ... < questions.txt

Needs Espressif's QEMU fork (github.com/espressif/qemu/releases, the
qemu-xtensa-softmmu build). Prints: question <TAB> answer. Timings in the
emulator are not real ESP32 timings, so none are printed."""

import argparse
import glob
import os
import select
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


# chip -> (bootloader offset, flash size) for the merged image
CHIPS = {"esp32": ("0x1000", "4MB"), "esp32s3": ("0x0", "16MB")}


def merged_flash(build, out, chip):
    esptool = glob.glob(os.path.expanduser("~/.platformio/packages/tool-esptoolpy/esptool.py"))[0]
    boot_at, size = CHIPS[chip]
    subprocess.run([sys.executable, esptool, "--chip", chip, "merge_bin", "-o", out,
                    "--fill-flash-size", size, boot_at, f"{build}/bootloader.bin",
                    "0x8000", f"{build}/partitions.bin", "0x10000", f"{build}/firmware.bin"],
                   check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qemu", default="qemu-system-xtensa")
    ap.add_argument("--chip", default="esp32", choices=sorted(CHIPS),
                    help="esp32 (default), or esp32s3 for the 16 MB large-model build")
    ap.add_argument("--build", default=None, help="default: .pio/build/esp32dev or esp32-s3-16mb")
    args = ap.parse_args()
    build = args.build or os.path.join(HERE, ".pio/build", "esp32dev" if args.chip == "esp32" else "esp32-s3-16mb")
    questions = [l.rstrip("\n") for l in sys.stdin if l.strip()]

    flash = os.path.join(tempfile.mkdtemp(), "flash.bin")
    merged_flash(build, flash, args.chip)
    p = subprocess.Popen([args.qemu, "-nographic", "-machine", args.chip, "-m", "4M",
                          "-drive", f"file={flash},if=mtd,format=raw", "-serial", "mon:stdio"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buf = b""

    def read_until(token, timeout):
        nonlocal buf
        end = time.time() + timeout
        while token not in buf and time.time() < end:
            if select.select([p.stdout], [], [], 0.2)[0]:
                buf += os.read(p.stdout.fileno(), 4096)
        if token not in buf:
            return None
        out, buf = buf.split(token, 1)
        return out.decode(errors="replace")

    boot = read_until(b"you: ", 120)
    if boot is None:
        p.kill()
        sys.exit("firmware did not reach the prompt")
    print(next((l for l in boot.splitlines() if "tinyai" in l), boot.strip()), file=sys.stderr)
    for q in questions:
        p.stdin.write(q.encode() + b"\r")
        p.stdin.flush()
        out = read_until(b"you: ", 300)
        if out is None:
            p.kill()
            sys.exit(f"no answer to {q!r}")
        ans = next((l[5:] for l in out.splitlines() if l.startswith("esp: ")), "")
        print(f"{q}\t{ans}", flush=True)
    p.kill()


if __name__ == "__main__":
    main()
