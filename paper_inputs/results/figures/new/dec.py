#!/usr/bin/env python3
"""dec.py - rebuild the images from the json containers written by enc.py.

Run from this folder:  python dec.py
Every image goes back to the relative path stored inside the container, so the
original folder structure (figures/) is recreated automatically.  Use
--out OTHER_FOLDER to restore somewhere else instead of in place.

The passphrase rules match enc.py exactly: --passphrase TEXT, then the
environment variable MODAL_ENC_PASSPHRASE, then the built-in default.  Each blob
is verified with HMAC-SHA256 and its sha256 digest, so a wrong passphrase or a
corrupted json file fails loudly instead of writing broken images.

This script is standalone (it repeats the small codec of enc.py) so the json
files can be opened with only this one file present.

Examples
  python dec.py                    restore in place, keep existing files
  python dec.py --out restored     restore into the subfolder restored/
  python dec.py --force            overwrite files that already exist
  python dec.py --dry-run          decrypt and verify everything, write nothing
  python dec.py --list             show the stored image paths only
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FORMAT = "MODALNET-FIG-ENC"
VERSION = 1
DEFAULT_PASSPHRASE = "MODAL-NET-FIGURES"
PASS_ENV = "MODAL_ENC_PASSPHRASE"
PBKDF2_ITERS = 100000
DEFAULT_IN = "encrypted"


def b64d(text):
    return base64.b64decode(str(text).encode("ascii"))


def hmac256(key, *parts):
    return hmac.new(key, b"".join(parts), hashlib.sha256).digest()


def aad_for(rel_path, size):
    """Must match enc.py exactly: path and byte length of the blob."""
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


def sanitize(rel_path):
    """Return a safe relative path, refusing traversal and absolute paths."""
    parts = [p for p in str(rel_path).replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." or ":" in p for p in parts):
        raise SystemExit("unsafe path stored in container: %r" % rel_path)
    return "/".join(parts)


def build_parser():
    ap = argparse.ArgumentParser(description="decrypt json containers back into images")
    ap.add_argument("--in", dest="in_dir", help="folder with the json files (default: SCRIPT/encrypted)")
    ap.add_argument("--out", dest="out_dir", help="root folder to restore into (default: this script folder)")
    ap.add_argument("--passphrase", metavar="TEXT", help="key phrase, also read from env %s" % PASS_ENV)
    ap.add_argument("--force", action="store_true", help="overwrite files that already exist")
    ap.add_argument("--dry-run", action="store_true", help="decrypt and verify, write nothing")
    ap.add_argument("--list", action="store_true", help="only list the stored file paths")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(args.out_dir).resolve() if args.out_dir else SCRIPT_DIR
    in_dir = Path(args.in_dir).resolve() if args.in_dir else SCRIPT_DIR / DEFAULT_IN
    containers = sorted(in_dir.glob("*.json"))
    if not containers:
        raise SystemExit("no json containers found in %s (run enc.py first)" % in_dir)
    passphrase = args.passphrase or os.environ.get(PASS_ENV) or DEFAULT_PASSPHRASE
    print("containers : %d json files from %s" % (len(containers), in_dir))
    print("restore to : %s" % root)

    written = skipped = listed = 0
    for path in containers:
        header = json.loads(path.read_text(encoding="ascii"))
        if header.get("format") != FORMAT:
            print("skip       : %s is not a %s container" % (path.name, FORMAT))
            continue
        kdf = header["kdf"]
        master = derive_key(passphrase, b64d(kdf["salt"]), kdf.get("iters", PBKDF2_ITERS))
        print("opening    : %s  (group %s, part %s of %s)" % (path.name, header.get("group"), header.get("part", 1), header.get("parts", 1)))
        for entry in header["files"]:
            rel = sanitize(entry["path"])
            if args.list:
                listed += 1
                print("stored     : %s  (%d bytes)" % (rel, entry["bytes"]))
                continue
            data = open_blob(entry, master, aad_for(rel, entry["bytes"]))
            target = root / rel
            if target.exists() and not args.force and not args.dry_run:
                skipped += 1
                print("exists     : %s (use --force to overwrite)" % rel)
                continue
            if args.dry_run:
                written += 1
                print("verified   : %s  (%d bytes)" % (rel, len(data)))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".dec.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
            written += 1
            print("restored   : %s  (%d bytes)" % (rel, len(data)))
    if args.list:
        print("done       : %d stored files" % listed)
    else:
        print("done       : %d restored, %d skipped, HMAC-SHA256 and sha256 verified" % (written, skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
