"""Benchmark of the Voronoi overlap transforms. Run by hand, not by pytest.

    python tests/voronoi_benchmark.py                      # quick, onavg-ico32
    python tests/voronoi_benchmark.py --native onavg-ico128 --record

Each workload runs in its own process, so that its peak memory is its own.
A workload is a list of requests, and a request is a list of (source,
target) pairs that are asked for together. For every workload, the script
reports the time of each step, the time inside selected private functions,
the peak memory, and the digest of each output. Outputs with the same name
must have the same digest in every workload and, unless a commit changes
the results on purpose, across commits.

With ``--record``, the results are appended to
``data/voronoi_benchmark.json``, with the environment and input digests.
"""

import argparse
import datetime
import json
import os
import resource
import subprocess
import sys
import time
from collections import defaultdict

import voronoi_golden as vg

RECORD_FN = os.path.join(os.path.dirname(__file__), "data", "voronoi_benchmark.json")
LR = vg.LR
MARKER = "BENCHMARK_RESULT "
# The templates of each native space, from coarse to fine.
TEMPLATES = {
    "onavg-ico32": ["onavg-ico8", "onavg-ico16", "onavg-ico32"],
    "onavg-ico64": ["onavg-ico16", "onavg-ico32", "onavg-ico64"],
    "onavg-ico128": ["onavg-ico32", "onavg-ico64", "onavg-ico128"],
}


def workloads(templates):
    """The requests of each workload. ``"areas"`` asks for native cell areas."""
    to_templates = [("native", name) for name in templates]
    between = (templates[0], templates[1])
    return {
        "single": [to_templates[:1]],
        "batch": [to_templates],
        "separate": [[pair] for pair in to_templates],
        "batch_tt": [to_templates + [between]],
        "separate_tt": [[pair] for pair in to_templates + [between]],
        "partial": [to_templates[:2], [to_templates[2], between]],
        "areas": ["areas"],
    }


# Private functions whose total time is reported, as (owner, attribute).
# The owner is an attribute of the voronoi module, or None for the module
# itself. Entries that do not exist are skipped, and times can be nested.
STAGES = [
    (None, "_subdivided_graph"),
    (None, "dijkstra"),
    ("VoronoiSurface", "_label_points"),
    ("VoronoiSurface", "_boundary_distances"),
    ("VoronoiSurface", "_local_labels"),
    ("VoronoiSurface", "_decode"),
    ("VoronoiSurface", "_entries"),
    ("_kernel", "label_samples"),
]


# The next three functions are the only ones that use the API.


def make_surface(coords, faces, sphere):
    """Return the surface and its partitions by name."""
    from neuroboros.surface.voronoi import VoronoiSurface

    surface = VoronoiSurface(coords, faces, sphere)
    return surface, {"native": surface.native}


def add_partition(surface, partitions, name, points):
    partitions[name] = surface.partition(points)


def run_request(surface, partitions, request):
    """Return the outputs of one request by name."""
    if request == "areas":
        return {"areas native": partitions["native"].areas()}
    mats = surface.overlaps([(partitions[a], partitions[b]) for a, b in request])
    return {f"{a} to {b}": mat for (a, b), mat in zip(request, mats)}


def uses_kernel():
    from neuroboros.surface import voronoi

    return getattr(voronoi, "_kernel", None) is not None


def install_timers(stage_times):
    from neuroboros.surface import voronoi

    def timed(label, func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                stage_times[label] += time.perf_counter() - start

        return wrapper

    for owner_name, attribute in STAGES:
        owner = voronoi if owner_name is None else getattr(voronoi, owner_name, None)
        if hasattr(owner, attribute):
            label = attribute.lstrip("_")
            setattr(owner, attribute, timed(label, getattr(owner, attribute)))


def peak_memory_gb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # In bytes on macOS and in units of 1024 bytes on Linux.
    return peak * (1 if sys.platform == "darwin" else 1024) / 1e9


def run_workload(native, templates, workload):
    """Run one workload in this process and return its measurements."""
    import neuroboros as nb

    requests = workloads(templates)[workload]
    pairs = [pair for request in requests if request != "areas" for pair in request]
    names = {name for pair in pairs for name in pair} - {"native"}
    coords, faces = nb.geometry("midthickness", LR, native)
    sphere = nb.geometry("sphere.reg", LR, native, vertices_only=True)
    points = {
        name: nb.geometry("sphere.reg", LR, name, vertices_only=True)
        for name in templates
        if name in names
    }
    stage_times = defaultdict(float)
    install_timers(stage_times)

    steps, results = {}, {}
    step = time.perf_counter()
    surface, partitions = make_surface(coords, faces, sphere)
    steps["surface"] = time.perf_counter() - step
    step = time.perf_counter()
    for name, point_coords in points.items():
        add_partition(surface, partitions, name, point_coords)
    steps["partitions"] = time.perf_counter() - step
    steps["requests"] = []
    for request in requests:
        step = time.perf_counter()
        results.update(run_request(surface, partitions, request))
        steps["requests"].append(time.perf_counter() - step)
    # The measurements end here, so that they do not include the digests.
    total = steps["surface"] + steps["partitions"] + sum(steps["requests"])
    peak = peak_memory_gb()
    outputs = {name: vg.record(result) for name, result in results.items()}
    return dict(
        total=total,
        peak_memory_gb=peak,
        kernel=uses_kernel(),
        steps=steps,
        stages=dict(stage_times),
        outputs=outputs,
    )


def run_in_subprocess(args, workload):
    command = [sys.executable, __file__, "--child", workload]
    command += ["--native", args.native, "--templates", *args.templates]
    env = dict(os.environ, NEUROBOROS_VORONOI_KERNEL=str(int(args.kernel == "on")))
    out = subprocess.run(command, capture_output=True, text=True, env=env)
    lines = [line for line in out.stdout.splitlines() if line.startswith(MARKER)]
    if out.returncode or not lines:
        raise RuntimeError(f"Workload {workload} failed:\n{out.stdout}\n{out.stderr}")
    return json.loads(lines[-1][len(MARKER) :])


def input_digests(native, templates):
    case = dict(kind="overlap", native=native)
    digests = {}
    for name in templates:
        digests.update(vg.inputs(dict(case, source="native", target=name)))
    return digests


def describe(result):
    requests = " + ".join(f"{t:.1f}" for t in result["steps"]["requests"])
    stages = ", ".join(f"{k} {v:.1f}" for k, v in sorted(result["stages"].items()))
    return (
        f"{result['total']:7.1f} s  {result['peak_memory_gb']:5.2f} GB  "
        f"surface {result['steps']['surface']:.1f}, "
        f"partitions {result['steps']['partitions']:.1f}, requests {requests}"
        f"\n{'':15}[{stages}]"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--native", default="onavg-ico32", choices=list(TEMPLATES))
    parser.add_argument("--templates", nargs=3, help="Three spaces, coarse to fine.")
    parser.add_argument("--workloads", nargs="+", help="Default: all.")
    parser.add_argument("--kernel", default="on", choices=["on", "off"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--note", default="", help="Stored with the record.")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.templates = args.templates or TEMPLATES[args.native]

    if args.child:
        result = run_workload(args.native, args.templates, args.child)
        print(MARKER + json.dumps(result))
        return

    names = args.workloads or list(workloads(args.templates))
    results, digests = {}, {}
    for name in names:
        runs = [run_in_subprocess(args, name) for _ in range(args.repeats)]
        # Report the fastest run and keep the totals of all of them.
        results[name] = dict(
            min(runs, key=lambda run: run["total"]),
            totals=[run["total"] for run in runs],
        )
        print(f"{name:12} {describe(results[name])}", flush=True)
        for run in runs:
            for output, record in run["outputs"].items():
                expected = digests.setdefault(output, record)
                assert record == expected, f"{output} differs in {name}"
    print("Every output has the same digest in all workloads.")
    for output, record in digests.items():
        print(f"  {output}: sum {record['sum']!r}, sha256 {record['sha256'][:16]}")

    if args.record:
        source = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip()
        record = dict(
            source=source,
            date=datetime.date.today().isoformat(),
            note=args.note,
            kernel=all(result["kernel"] for result in results.values()),
            environment=vg.environment(),
            native=args.native,
            templates=args.templates,
            inputs=input_digests(args.native, args.templates),
            outputs=digests,
            workloads={
                k: dict(v, outputs=list(v["outputs"])) for k, v in results.items()
            },
        )
        records = []
        if os.path.exists(RECORD_FN):
            with open(RECORD_FN) as f:
                records = json.load(f)
        records.append(record)
        with open(RECORD_FN, "w") as f:
            json.dump(records, f, indent=1)
            f.write("\n")
        print(f"Recorded as {source} in {RECORD_FN}.")


if __name__ == "__main__":
    main()
