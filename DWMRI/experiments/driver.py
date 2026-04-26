import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone

import yaml


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json_dump(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_src_cwd(manifest_path):
    d = os.path.abspath(os.path.join(os.path.dirname(manifest_path), "..", "src"))
    return d if os.path.isdir(d) else None


def _default_repo_root(manifest_path):
    return os.path.abspath(os.path.join(os.path.dirname(manifest_path), ".."))


def _event_path(output_root):
    return os.path.join(output_root, "driver_events.jsonl")


def _state_path(output_root):
    return os.path.join(output_root, "driver_state.json")


def _summary_path(output_root):
    return os.path.join(output_root, "driver_summary.json")


def _append_event(output_root, event):
    event = dict(event)
    event.setdefault("timestamp_utc", _now_iso())
    event_file = _event_path(output_root)
    os.makedirs(os.path.dirname(event_file), exist_ok=True)
    with open(event_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _load_state(output_root, manifest_jobs):
    path = _state_path(output_root)
    manifest_job_ids = [job["id"] for job in manifest_jobs]
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {"created_at_utc": _now_iso(), "jobs": {}, "history": []}

    state.setdefault("created_at_utc", _now_iso())
    state.setdefault("jobs", {})
    state.setdefault("history", [])

    for job in manifest_jobs:
        if job["id"] not in state["jobs"]:
            state["jobs"][job["id"]] = {
                "status": "pending",
                "attempts": 0,
                "last_updated_utc": _now_iso(),
            }
    state["manifest_job_ids"] = manifest_job_ids
    state["last_loaded_utc"] = _now_iso()
    return state


def _save_state(output_root, state):
    _atomic_json_dump(_state_path(output_root), state)


def _build_command(job, exp_id, output_root, registry_path):
    cmd = list(job["command"])
    if job.get("append_registry_flags", True):
        cmd += [
            "--exp-id",
            exp_id,
            "--job-id",
            job["id"],
            "--recipe",
            job.get("recipe", job["id"]),
        ]
        cmd += ["--output-root", output_root, "--registry-path", registry_path]
    return cmd


def _resolve_cwd(job, manifest_path):
    cwd = job.get("cwd")
    if cwd == "repo_root":
        return _default_repo_root(manifest_path)
    if cwd is None or cwd == "null":
        return _default_src_cwd(manifest_path) or os.getcwd()
    return cwd


def _select_jobs(manifest_jobs, state, resume, retry_failed):
    selected = []
    skipped = []
    for job in manifest_jobs:
        entry = state["jobs"].get(job["id"], {})
        status = entry.get("status", "pending")
        if not resume:
            selected.append(job)
            continue
        if status == "succeeded":
            skipped.append({"id": job["id"], "reason": "already_succeeded"})
            continue
        if status in {"failed", "interrupted"} and not retry_failed:
            skipped.append({"id": job["id"], "reason": f"previous_{status}"})
            continue
        selected.append(job)
    return selected, skipped


def _mark_job_state(state, job_id, **fields):
    entry = state["jobs"].setdefault(job_id, {"status": "pending", "attempts": 0})
    entry.update(fields)
    entry["last_updated_utc"] = _now_iso()


def run_job(job, exp_id, output_root, registry_path, manifest_path):
    cmd = _build_command(job, exp_id, output_root, registry_path)
    cwd = _resolve_cwd(job, manifest_path)
    log_path = os.path.join(output_root, "runs", f"{job['id']}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n[{_now_iso()}] driver starting attempt with command: {cmd}\n")
        proc = subprocess.run(cmd, cwd=cwd, stdout=logf, stderr=logf)
    return {
        "id": job["id"],
        "exit_code": proc.returncode,
        "duration_s": time.time() - t0,
        "log_path": log_path,
        "cwd": cwd,
        "command": cmd,
    }


def _install_interrupt_handlers():
    interrupted = {"flag": False, "signal": None}
    original = {}

    def _handler(signum, _frame):
        interrupted["flag"] = True
        interrupted["signal"] = signum
        raise KeyboardInterrupt()

    for sig in (signal.SIGINT, signal.SIGTERM):
        original[sig] = signal.getsignal(sig)
        signal.signal(sig, _handler)
    return interrupted, original


def _restore_interrupt_handlers(original):
    for sig, handler in original.items():
        signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description="Sequential experiment driver")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip succeeded jobs from existing state")
    parser.add_argument("--reset-state", action="store_true", help="Discard existing driver state and start fresh")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="When resuming, include previously failed/interrupted jobs",
    )
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    manifest = load_manifest(args.manifest)
    jobs = manifest.get("jobs", [])
    job_ids = [job["id"] for job in jobs]
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("Manifest contains duplicate job ids")

    if args.reset_state and os.path.exists(_state_path(args.output_root)):
        os.remove(_state_path(args.output_root))

    state = _load_state(args.output_root, jobs)
    missing_in_manifest = [jid for jid in state["jobs"].keys() if jid not in set(job_ids)]
    for jid in missing_in_manifest:
        _append_event(
            args.output_root,
            {"event": "manifest_mismatch", "job_id": jid, "details": "state job not in current manifest"},
        )

    selected_jobs, skipped_jobs = _select_jobs(
        jobs,
        state=state,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )

    for skipped in skipped_jobs:
        _append_event(args.output_root, {"event": "job_skipped", "job_id": skipped["id"], "reason": skipped["reason"]})

    _save_state(args.output_root, state)
    run_started_at = time.time()
    run_started_iso = _now_iso()
    summary_rows = []
    interrupted, original_handlers = _install_interrupt_handlers()
    current_job_id = None

    try:
        for job in selected_jobs:
            current_job_id = job["id"]
            prev_attempts = int(state["jobs"].get(current_job_id, {}).get("attempts", 0))
            _mark_job_state(
                state,
                current_job_id,
                status="running",
                attempts=prev_attempts + 1,
                started_at_utc=_now_iso(),
                command=_build_command(job, args.exp_id, args.output_root, args.registry_path),
            )
            _save_state(args.output_root, state)
            _append_event(args.output_root, {"event": "job_started", "job_id": current_job_id})

            result = run_job(
                job,
                exp_id=args.exp_id,
                output_root=args.output_root,
                registry_path=args.registry_path,
                manifest_path=manifest_path,
            )
            status = "succeeded" if result["exit_code"] == 0 else "failed"
            _mark_job_state(
                state,
                current_job_id,
                status=status,
                finished_at_utc=_now_iso(),
                exit_code=result["exit_code"],
                duration_s=result["duration_s"],
                log_path=result["log_path"],
                cwd=result["cwd"],
            )
            _save_state(args.output_root, state)
            _append_event(
                args.output_root,
                {
                    "event": "job_finished",
                    "job_id": current_job_id,
                    "status": status,
                    "exit_code": result["exit_code"],
                    "duration_s": result["duration_s"],
                },
            )
            summary_rows.append(result)
            if status == "failed" and args.fail_fast:
                break
    except KeyboardInterrupt:
        if current_job_id is not None:
            _mark_job_state(
                state,
                current_job_id,
                status="interrupted",
                interrupted_at_utc=_now_iso(),
                interrupt_signal=interrupted["signal"],
            )
            _save_state(args.output_root, state)
            _append_event(
                args.output_root,
                {
                    "event": "job_interrupted",
                    "job_id": current_job_id,
                    "signal": interrupted["signal"],
                },
            )
        raise
    finally:
        _restore_interrupt_handlers(original_handlers)
        status_counts = {}
        for entry in state["jobs"].values():
            status = entry.get("status", "pending")
            status_counts[status] = status_counts.get(status, 0) + 1
        driver_summary = {
            "exp_id": args.exp_id,
            "manifest_path": manifest_path,
            "started_at_utc": run_started_iso,
            "ended_at_utc": _now_iso(),
            "duration_s": time.time() - run_started_at,
            "resume_enabled": bool(args.resume),
            "retry_failed": bool(args.retry_failed),
            "fail_fast": bool(args.fail_fast),
            "selected_jobs": [job["id"] for job in selected_jobs],
            "skipped_jobs": skipped_jobs,
            "status_counts": status_counts,
            "results": summary_rows,
        }
        _atomic_json_dump(_summary_path(args.output_root), driver_summary)


if __name__ == "__main__":
    main()
