"""LAS/LAZ -> COPC (Cloud Optimized Point Cloud) for the browser, the way the PLA QC app does it: Untwine builds the
COPC octree out of core on every CPU core (82.5 M points in ~45 s, ~1.3 GB RAM, ~65 B/point of scratch disk), and
Potree streams it by level of detail with HTTP range requests, so every point of the survey can be shown."""
from __future__ import annotations
import glob, os, shutil, subprocess, tempfile, time
from pathlib import Path

SCRATCH_PER_POINT = 70  # bytes of Untwine scratch disk per point (measured ~65)

def untwine_exe() -> str | None:
    """UNTWINE_BIN, untwine on PATH, a conda environment that has it (conda install -c conda-forge untwine) or QGIS."""
    home = Path.home()
    for c in [os.getenv('UNTWINE_BIN'), shutil.which('untwine'),
              *sorted(glob.glob(str(home / 'miniforge3' / 'envs' / '*' / 'Library' / 'bin' / 'untwine.exe'))),
              *sorted(glob.glob(str(home / 'anaconda3' / 'envs' / '*' / 'Library' / 'bin' / 'untwine.exe'))),
              *sorted(glob.glob(r'C:\Program Files\QGIS*\apps\qgis\untwine.exe'), reverse=True),
              *sorted(glob.glob(str(home / 'miniforge3' / 'envs' / '*' / 'bin' / 'untwine')))]:
        if c and Path(c).is_file(): return str(c)
    return None

def point_count(path_or_stream) -> int:
    import laspy
    with (laspy.open(path_or_stream) if isinstance(path_or_stream, (str, Path)) else laspy.open(path_or_stream, closefd=False)) as r:
        return int(r.header.point_count)

def convert(src: Path, dst: Path, progress=None) -> int:
    """Write dst (.copc.laz) from a LAS/LAZ. Writes to a partial file first, so an interrupted run is never mistaken for
    a finished conversion. Returns the number of points."""
    exe = untwine_exe()
    if not exe: raise RuntimeError('Untwine is not installed on the GPU machine: conda install -c conda-forge untwine, or set UNTWINE_BIN')
    dst = Path(dst); dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.name[:-len('.copc.laz')] + '.partial.copc.laz'); partial.unlink(missing_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.untwine-', dir=dst.parent))
    flags = 0x4000 if os.name == 'nt' else 0  # BELOW_NORMAL_PRIORITY_CLASS: the machine stays responsive
    try:
        p = subprocess.Popen([exe, '-i', str(src), '-o', str(partial), '--temp_dir', str(work)], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, creationflags=flags)
        t0 = time.time(); out = []
        for line in p.stdout:  # untwine prints its stages
            out.append(line.rstrip())
            if progress and line.strip(): progress(f'Converting to COPC ({time.time() - t0:.0f} s): {line.strip()[:80]}')
        if p.wait() != 0 or not partial.is_file():
            raise RuntimeError(f'Untwine failed ({p.returncode}): ' + ' | '.join(out[-5:])[:500])
        os.replace(partial, dst)
    finally:
        shutil.rmtree(work, ignore_errors=True); partial.unlink(missing_ok=True)  # untwine leaves its scratch files behind
    return point_count(dst)
