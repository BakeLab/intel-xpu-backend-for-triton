"""Ahead-of-time builds of the native helper modules the XPU driver loads.

`compile_module_from_src` compiles `driver.c`, `arch_parser.c` and
`extension_utils.c` with the SYCL compiler the first time a process touches
the GPU, which costs seconds per module on a fresh cache. Building them once with the wheel removes that
cold-start cost. Each prebuilt module is recorded in a manifest with the hash
of its source and the exact variant flags; anything that does not match --
another source revision, an LTS driver, `INJECT_PYTORCH` -- is compiled at
runtime as before.

The modules link the SYCL and Level Zero runtimes by soname and carry no
RPATH, so they resolve through the loader's search path at runtime.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

MANIFEST = "manifest.json"
PREBUILT_DIR = Path(__file__).parent / "prebuilt"


def _source_hash(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def find(name: str, src: str, flags: list[str]) -> str | None:
    """Returns the prebuilt module for this exact source and variant, if any."""
    if os.environ.get("TRITON_INTEL_DISABLE_PREBUILT", "0") == "1":
        return None
    try:
        entries = json.loads((PREBUILT_DIR / MANIFEST).read_text())
    except (OSError, ValueError):
        return None
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    for entry in entries:
        if (entry["name"] == name and entry["ext_suffix"] == suffix and entry["flags"] == flags
                and entry["src_sha256"] == _source_hash(src)):
            path = PREBUILT_DIR / entry["file"]
            return str(path) if path.is_file() else None
    return None


# The variants worth building ahead of time: what a rolling (non-LTS) driver
# without `INJECT_PYTORCH` asks `compile_module_from_src` for.
VARIANTS = (
    ("spirv_utils", "driver.c", ["-DENABLE_EXPERIMENTAL_EVENTLESS_SUBMIT"]),
    ("arch_utils", "arch_parser.c", []),
    ("extension_utils_impl", "extension_utils.c", []),
)


def build(out_dir: str, compiler: str) -> list[dict]:
    """Builds every variant into `out_dir` with the SYCL `compiler` and writes the manifest.

    Runs at wheel-build time, so it must not import triton: the package and its
    runtime dependencies are not importable there.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    backend_dir = Path(__file__).parent
    # The search paths `CompilationHelper` derives from icpx and ZE_PATH.
    compiler_root = Path(compiler).resolve().parent.parent
    ze_root = os.getenv("LEVEL_ZERO_V1_SDK_PATH") or os.getenv("ZE_PATH", "/usr/local")
    include_dirs = [
        Path(ze_root) / "include", compiler_root / "include", compiler_root / "include" / "sycl",
        backend_dir / "include",
        sysconfig.get_paths()["include"]
    ]
    library_dirs = [compiler_root / "lib", backend_dir / "lib"]
    entries = []
    for name, source, flags in VARIANTS:
        src = (backend_dir / source).read_text()
        with tempfile.TemporaryDirectory() as tmp:
            src_path = os.path.join(tmp, "main.cpp")
            Path(src_path).write_text(src)
            so = os.path.join(tmp, f"{name}{suffix}")
            # What `_build` runs on XPU, minus the RPATH `compile_module_from_src` adds.
            cmd = [compiler, src_path, "-O3", "-shared", "-Wno-psabi", "-fPIC", "-lsycl", "-lze_loader"]
            cmd += [f"-L{d}" for d in library_dirs] + [f"-I{d}" for d in [*include_dirs, tmp]]
            cmd += ["-o", so, *flags, "-fsycl", "-fno-sycl-id-queries-fit-in-int"]
            subprocess.run(cmd, check=True)
            file = f"{name}-{_source_hash(src)[:16]}{suffix}"
            shutil.copyfile(so, out / file)
        entries.append(
            {"name": name, "file": file, "flags": flags, "ext_suffix": suffix, "src_sha256": _source_hash(src)})
    (out / MANIFEST).write_text(json.dumps(entries, indent=2) + "\n")
    return entries


if __name__ == "__main__":
    for entry in build(sys.argv[1], sys.argv[2]):
        print(f"prebuilt {entry['name']}: {entry['file']}")
