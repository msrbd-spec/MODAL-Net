#!/usr/bin/env python3
"""enc.py - encrypt the figure images of this folder into portable JSON files.

Run from this folder:  python enc.py

Default mapping (root = folder of this script):
  figures/*.png                      to  encrypted/figures_part1_of2.json    (6 images)
                                         encrypted/figures_part2_of2.json    (5 images)
  phase3b_joint/p3b_joint/xai/*.png  to  encrypted/phase3b_joint__p3b_joint__xai_part1_of1.json

Rebuild the images and the folder structure with:  python dec.py

Cryptography uses only the Python standard library: PBKDF2-HMAC-SHA256 key
derivation, a SHA-256 CTR style keystream that is XORed with the image bytes,
plus an HMAC-SHA256 tag over the ciphertext (encrypt then MAC).  The image bytes
are stored exactly as they are on disk (never re-encoded), so decryption restores
the original PNG bit for bit and there is no quality loss.
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FORMAT = "MODALNET-FIG-ENC"
VERSION = 1
DEFAULT_PASSPHRASE = "MODAL-NET-FIGURES"
PASS_ENV = "MODAL_ENC_PASSPHRASE"
PBKDF2_ITERS = 100000
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
DEFAULT_GROUPS = ("figures=2", "phase3b_joint/p3b_joint/xai=1")
DEFAULT_OUT = "encrypted"


def b64e(raw):
    return base64.b64encode(raw).decode("ascii")


def b64d(text):
    return base64.b64decode(str(text).encode("ascii"))


def hmac256(key, *parts):
    return hmac.new(key, b"".join(parts), hashlib.sha256).digest()


def aad_for(rel_path, size):
    """Binds every encrypted blob to its relative path and byte length."""
    return ("%s|%d|%s|len=%d|" % (FORMAT, VERSION, rel_path, int(size))).encode("ascii")


def derive_key(passphrase, salt, iters=PBKDF2_ITERS):
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), bytes(salt), int(iters), 32)


def keystream_xor(key, data):
    """XOR data with a SHA-256 derived keystream (its own inverse)."""
    out = bytearray(len(data))
    pos = 0
    for counter in range((len(data) + 31) // 32):
        block = hashlib.sha256(key + counter.to_bytes(8, "big")).digest()
        n = min(32, len(data) - pos)
        mixed = (int.from_bytes(data[pos:pos + n], "big") ^ int.from_bytes(block[:n], "big")).to_bytes(n, "big")
        out[pos:pos + n] = mixed
        pos += n
    return bytes(out)


def seal(plain, master, aad):
    """Encrypt one blob, return the json fields describing it."""
    nonce = os.urandom(16)
    ct = keystream_xor(hmac256(master, b"ENC", aad, nonce), plain)
    tag = hmac256(hmac256(master, b"MAC", aad, nonce), ct)
    return {"nonce": b64e(nonce), "ct": b64e(ct), "tag": b64e(tag)}


def open_blob(item, master, aad):
    """Decrypt and authenticate one entry, return the original bytes."""
    nonce = b64d(item["nonce"])
    ct = b64d(item["ct"])
    mac_key = hmac256(master, b"MAC", aad, nonce)
    if not hmac.compare_digest(hmac256(mac_key, ct), b64d(item["tag"])):
        raise RuntimeError("authentication failed: wrong passphrase or corrupted json")
    data = keystream_xor(hmac256(master, b"ENC", aad, nonce), ct)
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise RuntimeError("sha256 mismatch: %s" % item["path"])
    return data


def parse_groups(values):
    groups = []
    for value in values:
        spec = value.replace("\\", "/").strip("/")
        rel, sep, tail = spec.rpartition("=")
        if sep and tail.isdigit():
            rel, parts = rel.rstrip("/"), int(tail)
        else:
            rel, parts = spec, 1
        if not rel:
            raise SystemExit("bad --group value %r (use REL_DIR or REL_DIR=NPARTS)" % value)
        groups.append((rel, max(1, parts)))
    return groups


def collect_images(root, rel_dir, out_dir):
    base = root / rel_dir
    if not base.is_dir():
        raise SystemExit("folder not found: %s" % base)
    skip = out_dir.resolve()
    found = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        real = path.resolve()
        if skip in real.parents:
            continue
        found.append(real.relative_to(root).as_posix())
    return found


def split_even(items, parts):
    parts = max(1, min(int(parts), len(items)))
    size = -(-len(items) // parts)
    return [items[i:i + size] for i in range(0, len(items), size)]


def resolve_passphrase(args):
    text = args.passphrase or os.environ.get(PASS_ENV)
    if text:
        return text, False
    if args.require_passphrase:
        raise SystemExit("--require-passphrase: set %s or pass --passphrase TEXT" % PASS_ENV)
    return DEFAULT_PASSPHRASE, True


def build_parser():
    ap = argparse.ArgumentParser(description="encrypt figure images into json containers")
    ap.add_argument("--root", help="folder holding the images (default: this script folder)")
    ap.add_argument("--out", help="output folder for the json files (default: ROOT/encrypted)")
    ap.add_argument("--group", action="append", metavar="REL_DIR[=NPARTS]",
                    help="folder to encrypt, optionally split into NPARTS json files (repeatable)")
    ap.add_argument("--passphrase", metavar="TEXT", help="key phrase, also read from env %s" % PASS_ENV)
    ap.add_argument("--require-passphrase", action="store_true", help="refuse the built-in default passphrase")
    ap.add_argument("--ext", action="append", metavar=".PNG", help="image extension to include (repeatable)")
    ap.add_argument("--pretty", action="store_true", help="indent the json output (bigger files)")
    ap.add_argument("--delete-sources", action="store_true", help="remove the originals after successful encryption")
    return ap


def main(argv=None):
    global IMAGE_EXTS
    args = build_parser().parse_args(argv)
    if args.ext:
        IMAGE_EXTS = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext}
    root = Path(args.root).resolve() if args.root else SCRIPT_DIR
    out_dir = Path(args.out).resolve() if args.out else root / DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    passphrase, used_default = resolve_passphrase(args)

    print("root       : %s" % root)
    print("output     : %s" % out_dir)
    print("cipher     : SHA256-CTR keystream XOR + HMAC-SHA256, key PBKDF2-SHA256 %d iters" % PBKDF2_ITERS)
    print("passphrase : %s" % ("built-in default" if used_default else "supplied"))

    total = 0
    doomed = []
    for rel_dir, parts in parse_groups(args.group or list(DEFAULT_GROUPS)):
        files = collect_images(root, rel_dir, out_dir)
        if not files:
            print("skip       : %s has no images" % rel_dir)
            continue
        slug = rel_dir.replace("/", "__")
        chunks = split_even(files, parts)
        for index, chunk in enumerate(chunks, start=1):
            salt = os.urandom(16)
            master = derive_key(passphrase, salt)
            entries = []
            for rel in chunk:
                raw = (root / rel).read_bytes()
                entry = {"path": rel, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                entry.update(seal(raw, master, aad_for(rel, len(raw))))
                entries.append(entry)
            container = {
                "format": FORMAT,
                "version": VERSION,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "group": rel_dir,
                "part": index,
                "parts": len(chunks),
                "file_count": len(entries),
                "plain_bytes": sum(e["bytes"] for e in entries),
                "kdf": {"name": "pbkdf2-hmac-sha256", "iters": PBKDF2_ITERS, "salt": b64e(salt)},
                "files": entries,
            }
            blob = json.dumps(container, separators=(",", ":"), indent=None if not args.pretty else 2)
            target = out_dir / ("%s_part%d_of%d.json" % (slug, index, len(chunks)))
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(blob, encoding="ascii")
            os.replace(tmp, target)
            total += container["plain_bytes"]
            print("wrote      : %s  %d images  %.2f MiB plain" % (target.name, len(entries), container["plain_bytes"] / 1048576.0))
            if args.delete_sources:
                doomed.extend(chunk)
    if args.delete_sources:
        for rel in doomed:
            (root / rel).unlink()
        print("deleted    : %d source images" % len(doomed))
    print("done       : %.2f MiB encrypted into %s" % (total / 1048576.0, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
