#!/usr/bin/env python3
"""Run the EmbedFlow v0.1.0 release gate without publishing anything.

The gate deliberately records unavailable optional interpreters/dependencies as
expected skips, while failing on code, registry, packaging, documentation, or
security problems.  It writes RELEASE_TEST_REPORT.md for human review.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import venv
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    output: str = ""
    duration: float = 0.0


CHECKS: list[Check] = []


def platform_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _record(name: str, status: str, detail: str = "", output: str = "", duration: float = 0.0) -> None:
    CHECKS.append(Check(name, status, detail, output[-4000:], duration))


def run_check(
    name: str,
    command: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    timeout: int = 300,
    required: bool = True,
) -> subprocess.CompletedProcess[str] | None:
    started = time.monotonic()
    merged = os.environ.copy()
    if env:
        merged.update(env)
    try:
        result = subprocess.run(command, cwd=cwd, env=merged, text=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        status = "FAIL" if required else "SKIP"
        _record(name, status, f"{type(exc).__name__}: {exc}", duration=time.monotonic() - started)
        return None
    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    _record(name, "PASS" if result.returncode == 0 else ("FAIL" if required else "SKIP"),
            f"exit={result.returncode}", output, time.monotonic() - started)
    return result


def python_env() -> dict[str, str]:
    env = {"PYTHONPATH": str(ROOT)}
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] += os.pathsep + os.environ["PYTHONPATH"]
    return env


def parse_test_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in ("passed", "failed", "skipped", "xfailed", "xpassed", "error"):
        match = re.search(rf"(\d+)\s+{key}", output)
        if match:
            counts[key] = int(match.group(1))
    return counts


def source_security_scan() -> tuple[list[str], list[str]]:
    secret_patterns = [
        ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
        ("access token", re.compile(r"\b(?:hf|ghp|github_pat)_[A-Za-z0-9_\-]{16,}\b")),
        ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
        ("bearer token", re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}")),
    ]
    path_patterns = [
        ("absolute Unix path", re.compile(r"/(?:home|Users|root)/[A-Za-z0-9_.\-]+")),
        ("Windows user path", re.compile(r"(?i)[A-Z]:\\Users\\")),
    ]
    secrets: list[str] = []
    paths: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in {".git", ".ruff_cache", ".pytest_cache", "__pycache__", "dist", "build", "embedflow.egg-info"} for part in path.parts) or path.name == "RELEASE_TEST_REPORT.md" or path.suffix in {".pyc", ".sqlite", ".db", ".npy", ".npz"}:
            continue
        try:
            if path.stat().st_size > 5_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        relative = str(path.relative_to(ROOT))
        for label, pattern in secret_patterns:
            if pattern.search(text):
                secrets.append(f"{relative}: {label}")
        for label, pattern in path_patterns:
            if pattern.search(text):
                paths.append(f"{relative}: {label}")
    return secrets, paths


def markdown_links() -> list[str]:
    missing: list[str] = []
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            target = target.strip().split("#", 1)[0].split("?", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            target = target.strip("<>")
            candidate = (path.parent / target).resolve()
            if not candidate.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")
    return missing


def free_port() -> int | None:
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except PermissionError:
        return None


def server_smoke(cwd: Path, *, config: str, demo: bool = True) -> bool | None:
    port = free_port()
    if port is None:
        return None
    command = [sys.executable, "-m", "embedflow", "serve", "--config", config, "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]
    if demo:
        command.append("--demo")
    process = subprocess.Popen(command, cwd=cwd, env={**os.environ, **python_env()}, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5) as response:
                    return response.status == 200
            except Exception:
                time.sleep(0.1)
        return False
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=5)


def wheel_audit(dist: Path) -> list[str]:
    problems: list[str] = []
    artifacts = sorted(dist.glob("*.whl"))
    if not artifacts:
        return ["no wheel produced"]
    for artifact in artifacts:
        if artifact.stat().st_size > 50_000_000:
            problems.append(f"wheel is unexpectedly large: {artifact.name}")
        with zipfile.ZipFile(artifact) as archive:
            names = archive.namelist()
            for name in names:
                lower = name.lower()
                if any(token in lower for token in ("runs/", "secrets", ".env", ".sqlite", ".db", ".safetensors", ".index")):
                    problems.append(f"forbidden package content: {name}")
                if archive.getinfo(name).file_size > 20_000_000:
                    problems.append(f"large wheel member: {name}")
    for artifact in sorted(dist.glob("*.tar.gz")):
        with tarfile.open(artifact, "r:gz") as archive:
            for member in archive.getmembers():
                lower = member.name.lower()
                if any(token in lower for token in ("runs/", ".env", ".sqlite", ".db", ".safetensors", ".index")):
                    problems.append(f"forbidden sdist content: {member.name}")
    return problems


def fresh_wheel_smoke(wheel: Path) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="embedflow-wheel-") as raw:
        root = Path(raw); env_dir = root / "venv"
        try:
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env_dir)
            py = env_dir / "bin" / "python"
            scripts = env_dir / "bin"
            install = subprocess.run([str(py), "-m", "pip", "install", "--no-deps", str(wheel)], cwd=root, text=True, capture_output=True, timeout=180)
            if install.returncode:
                return False, install.stdout + install.stderr
            for command in (["embedflow", "--help"], ["embedflow", "doctor", "--json"], ["embedflow", "registry", "verify"], ["embedflow", "demo", "--path", str(root / "demo"), "--no-serve"]):
                result = subprocess.run([str(scripts / command[0]), *command[1:]], cwd=root, text=True, capture_output=True, timeout=180)
                if result.returncode:
                    return False, (result.stdout or "") + (result.stderr or "")
            return True, "installed wheel and ran help, doctor, registry verify, and demo"
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)


def run_examples() -> None:
    with tempfile.TemporaryDirectory(prefix="embedflow-examples-") as raw:
        work = Path(raw)
        faiss_src = ROOT / "examples" / "faiss"
        faiss = work / "faiss"; shutil.copytree(faiss_src, faiss)
        commands = [
            [sys.executable, "-m", "embedflow", "init", "--config", "embedflow.yaml", "--build-index", "--queries", "queries.jsonl", "--demo"],
            [sys.executable, "-m", "embedflow", "analyze", "--config", "embedflow.yaml", "--queries", "queries.jsonl", "--demo", "--output-dir", "analysis"],
            [sys.executable, "-m", "embedflow", "status", "--config", "embedflow.yaml", "--demo"],
        ]
        for number, command in enumerate(commands, 1):
            if run_check(f"FAISS example command {number}", command, cwd=faiss, env=python_env()) is None:
                continue
        # Exercise the complete offline CLI surface against the same built
        # fixture. These commands deliberately avoid a network or paid model.
        cli_commands = [
            ("CLI registry list", [sys.executable, "-m", "embedflow", "registry", "list", "--json"]),
            ("CLI registry show", [sys.executable, "-m", "embedflow", "registry", "show", "--source", "Qwen/Qwen3-Embedding-4B", "--target", "Qwen/Qwen3-Embedding-8B", "--json"]),
            ("CLI registry match", [sys.executable, "-m", "embedflow", "registry", "match", "--config", "embedflow.yaml", "--json"]),
            ("CLI registry verify", [sys.executable, "-m", "embedflow", "registry", "verify", "--json"]),
            ("CLI doctor", [sys.executable, "-m", "embedflow", "doctor", "--config", "embedflow.yaml", "--json"]),
            ("CLI search", [sys.executable, "-m", "embedflow", "search", "what explains aurora", "--config", "embedflow.yaml", "--demo", "--top-k", "3"]),
            ("CLI prewarm", [sys.executable, "-m", "embedflow", "prewarm", "--config", "embedflow.yaml", "--demo", "--documents", "3"]),
            ("CLI audit-index", [sys.executable, "-m", "embedflow", "audit-index", "--config", "embedflow.yaml", "--demo", "--reference-index", "legacy.index", "--queries", "queries.jsonl", "--k", "10"]),
            ("CLI economics", [sys.executable, "-m", "embedflow", "economics", "--corpus-size", "100", "--docs-per-second", "10", "--gpu-price", "1", "--json"]),
            ("CLI export-target", [sys.executable, "-m", "embedflow", "export-target", "--config", "embedflow.yaml", "--demo", "--output-index", "target.index", "--batch-size", "16"]),
            ("CLI migrate", [sys.executable, "-m", "embedflow", "migrate", "--index", "legacy.index", "--documents", "documents.jsonl", "--old-model", "embedflow/demo-source", "--new-model", "embedflow/demo-target", "--backend", "faiss", "--config", "migrated.yaml", "--cache", "migrated_cache", "--state", "migrated_state.json", "--no-serve"]),
            ("CLI benchmark profiles", [sys.executable, "-m", "embedflow", "benchmark-profiles", "list", "--json"]),
        ]
        for name, command in cli_commands:
            run_check(name, command, cwd=faiss, env=python_env(), timeout=300)
        if importlib_available("qdrant_client"):
            run_check(
                "CLI Qdrant demo",
                [sys.executable, "-m", "embedflow", "demo", "--backend", "qdrant", "--path", "qdrant_cmd_demo", "--no-serve"],
                cwd=faiss,
                env=python_env(),
                timeout=300,
            )
            model_root = optional_external_root("EMBEDFLOW_MODEL_ROOT")
            if model_root and (model_root / "minilm_l6").is_dir() and (model_root / "qwen3_0_6b").is_dir():
                run_check(
                    "real local Qdrant model smoke",
                    [sys.executable, "scripts/real_qdrant_smoke.py", "--model-root", str(model_root)],
                    cwd=ROOT,
                    env=python_env(),
                    timeout=900,
                )
            else:
                _record("real local Qdrant model smoke", "SKIP", "set EMBEDFLOW_MODEL_ROOT to a directory containing minilm_l6/ and qwen3_0_6b/ snapshots")
        server_result = server_smoke(faiss, config="embedflow.yaml")
        _record("FAISS example server", "SKIP" if server_result is None else ("PASS" if server_result else "FAIL"),
                "sandbox disallows local sockets" if server_result is None else "")

        research_src = ROOT / "examples" / "research_analysis"
        research = work / "research_analysis"; shutil.copytree(research_src, research)
        commands = [
            [sys.executable, "-m", "embedflow", "init", "--config", "embedflow.yaml", "--build-index", "--queries", "queries.jsonl", "--demo"],
            [sys.executable, "-m", "embedflow", "evaluate", "--config", "embedflow.yaml", "--queries", "queries.jsonl", "--qrels", "qrels.json", "--k-values", "2,4,8", "--demo", "--output-dir", "results"],
        ]
        for number, command in enumerate(commands, 1):
            run_check(f"research example command {number}", command, cwd=research, env=python_env())

        if importlib_available("qdrant_client"):
            qdrant_src = ROOT / "examples" / "qdrant"
            qdrant = work / "qdrant"; shutil.copytree(qdrant_src, qdrant)
            commands = [
                [sys.executable, "build_index.py"],
                [sys.executable, "-m", "embedflow", "analyze", "--config", "embedflow.yaml", "--queries", "queries.jsonl", "--demo", "--output-dir", "analysis"],
            ]
            for number, command in enumerate(commands, 1):
                run_check(f"Qdrant example command {number}", command, cwd=qdrant, env=python_env())
            server_result = server_smoke(qdrant, config="embedflow.yaml")
            _record("Qdrant example server", "SKIP" if server_result is None else ("PASS" if server_result else "FAIL"),
                    "sandbox disallows local sockets" if server_result is None else "")
        else:
            _record("Qdrant example", "SKIP", "qdrant-client is not installed")


def importlib_available(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def optional_external_root(environment_name: str) -> Path | None:
    """Resolve an explicitly supplied external checkout or model directory.

    The public checkout must not depend on the maintainer's sibling-directory
    layout.  External research artifacts and downloaded model snapshots are
    opt-in inputs to this gate via environment variables.
    """

    value = os.environ.get(environment_name)
    if not value:
        return None
    candidate = Path(value).expanduser().resolve()
    return candidate if candidate.exists() else None


def main() -> int:
    os.chdir(ROOT)
    # When invoked as ``python scripts/release_gate.py`` Python initially puts
    # ``scripts/`` (rather than the repository root) on sys.path. Ensure the
    # in-process registry/report checks use this checkout, not an unrelated
    # globally installed EmbedFlow package.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    test_result = run_check("public pytest suite", [sys.executable, "-m", "pytest", "-q"], env=python_env(), timeout=600)
    test_counts = parse_test_counts(((test_result.stdout if test_result else "") or "") + ((test_result.stderr if test_result else "") or ""))

    ruff = shutil.which("ruff")
    if ruff:
        run_check("ruff", [ruff, "check", "."], env=python_env())
    else:
        run_check("ruff", [sys.executable, "-m", "ruff", "check", "."], env=python_env(), required=False)
    run_check("compileall", [sys.executable, "-m", "compileall", "-q", "embedflow", "src", "scripts"], env=python_env())
    registry_result: dict[str, object] | None = None
    # Registry artifact paths are canonical relative identifiers and may span
    # more than one retained research checkout.  Keep their optional base
    # separate from EMBEDFLOW_RESEARCH_ROOT, which points at the checkout used
    # for the legacy regression suite.
    provenance_root = optional_external_root("EMBEDFLOW_PROVENANCE_ROOT")
    registry_command = run_check(
        "registry verification",
        [
            sys.executable,
            "-c",
            "import os, pathlib; from embedflow.registry import verify_registry; root=os.environ.get('EMBEDFLOW_PROVENANCE_ROOT'); r=verify_registry(provenance_root=pathlib.Path(root) if root else None); print(r); raise SystemExit(0 if r['ok'] else 1)",
        ],
        env=python_env(),
    )
    # Re-read the structured result in-process for the generated report. The
    # subprocess above remains authoritative for the gate and verifies the
    # public import path independently of this script.
    try:
        from embedflow.registry import verify_registry

        registry_result = verify_registry(provenance_root=provenance_root)
    except Exception as exc:  # pragma: no cover - the subprocess records this
        registry_result = {"ok": False, "error": str(exc)}
    if registry_command is None and registry_result.get("ok"):
        registry_result["ok"] = False

    research = optional_external_root("EMBEDFLOW_RESEARCH_ROOT")
    if research and (research / "tests").exists() and (research / "tests_mvp").exists():
        run_check("research regression suite", [sys.executable, "-m", "pytest", "-q", "tests", "tests_mvp"], cwd=research, env={"PYTHONPATH": str(research)}, timeout=900)
    else:
        _record("research regression suite", "SKIP", "set EMBEDFLOW_RESEARCH_ROOT to a checkout containing tests/ and tests_mvp/")

    build = run_check("package build", [sys.executable, "-m", "build", "--no-isolation"], env=python_env(), timeout=600)
    dist = ROOT / "dist"
    package_problems = wheel_audit(dist) if build and build.returncode == 0 else ["package build did not complete"]
    _record("package artifact audit", "PASS" if not package_problems else "FAIL", "; ".join(package_problems))
    wheels = sorted(dist.glob("*.whl"))
    if wheels:
        ok, detail = fresh_wheel_smoke(wheels[-1])
        _record("fresh wheel install", "PASS" if ok else "FAIL", detail)
    else:
        _record("fresh wheel install", "FAIL", "no wheel available")

    run_examples()
    docs_missing = markdown_links()
    _record("documentation links", "PASS" if not docs_missing else "FAIL", "; ".join(docs_missing))

    secrets, paths = source_security_scan()
    _record("security scan", "PASS" if not secrets and not paths else "FAIL", "; ".join(secrets + paths))
    placeholders = []
    for path in (ROOT / "CITATION.cff", ROOT / "pyproject.toml", ROOT / "README.md"):
        if path.exists() and "REPLACE_ME" in path.read_text(encoding="utf-8"):
            placeholders.append(str(path.relative_to(ROOT)))
    license_path = ROOT / "LICENSE"
    if license_path.exists():
        license_text = license_path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"(?im)^\s*copyright\s+(?:\(c\)\s*)?\d{4}", license_text):
            placeholders.append("LICENSE (copyright holder/year confirmation required)")
    _record("manual metadata audit", "FAIL" if placeholders else "PASS", "unresolved placeholders: " + ", ".join(placeholders) if placeholders else "none")

    interpreters: dict[str, str] = {}
    for version in ("3.10", "3.11", "3.12"):
        candidate = sys.executable if platform_version() == version else shutil.which(f"python{version}")
        if candidate:
            dependency_probe = subprocess.run([candidate, "-c", "import pytest"], cwd=ROOT, env={**os.environ, **python_env()}, text=True, capture_output=True)
            if dependency_probe.returncode != 0:
                _record(f"Python {version} smoke", "SKIP", "pytest is not installed in this interpreter")
                interpreters[version] = "SKIP"
                continue
            result = run_check(f"Python {version} smoke", [candidate, "-m", "pytest", "-q"], env=python_env(), timeout=600)
            interpreters[version] = "PASS" if result and result.returncode == 0 else "FAIL"
        else:
            _record(f"Python {version} smoke", "SKIP", "interpreter unavailable")
            interpreters[version] = "SKIP"

    failures = [check for check in CHECKS if check.status == "FAIL"]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip() or "unknown"
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout.strip())
    commit_label = f"{commit} (working tree has uncommitted changes)" if dirty else commit
    report_lines = [
        "# EmbedFlow v0.1.0 Release Test Report", "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- Python running gate: {sys.version.split()[0]}",
        "- Repository: `embedflow` (local release checkout)",
        f"- Git commit: {commit_label}",
        "", "## Gate", "", f"**RELEASE GATE: {'FAIL' if failures else 'PASS'}**", "",
        "## Check matrix", "", "| Check | Status | Detail | Seconds |", "|---|---|---|---:|",
    ]
    for check in CHECKS:
        report_lines.append(f"| {check.name} | {check.status} | {check.detail.replace('|', '/')[:500]} | {check.duration:.2f} |")
    registry_rows = registry_result.get("record_count", "unknown") if registry_result else "unknown"
    registry_profiles = registry_result.get("profile_count", "unknown") if registry_result else "unknown"
    registry_summaries = registry_result.get("summary_count", "unknown") if registry_result else "unknown"
    registry_artifacts = registry_result.get("provenance_artifacts_checked", "unknown") if registry_result else "unknown"
    registry_semantics = registry_result.get("semantic_values_checked", "unknown") if registry_result else "unknown"
    report_lines.extend(["", "## Test counts", "", "```json", json.dumps(test_counts, indent=2, sort_keys=True), "```", "",
                         "## Python matrix", "", *[f"- Python {version}: **{status}**" for version, status in interpreters.items()],
                         "", "## Registry", "", f"Core registry rows: {registry_rows}", f"Benchmark profiles: {registry_profiles}", f"Research summaries: {registry_summaries}", f"Retained artifacts verified: {registry_artifacts}", f"Semantic values cross-checked: {registry_semantics}", "Provenance and packaged checksums are validated by `registry verify`.",
                         "", "## Security", "", "The source scan checks credentials, private keys, absolute local paths, and runtime artifacts. Manual metadata placeholders are a separate release blocker.",
                         "", "## Notes", "", "Expected skips are unavailable Python interpreters or optional Qdrant dependencies. No GitHub or external deployment commands are executed by this script.", ""])
    (ROOT / "RELEASE_TEST_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print("RELEASE GATE: " + ("FAIL" if failures else "PASS"))
    for check in CHECKS:
        print(f"{check.status:5} {check.name}: {check.detail}")
    print(f"Report: {ROOT / 'RELEASE_TEST_REPORT.md'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
