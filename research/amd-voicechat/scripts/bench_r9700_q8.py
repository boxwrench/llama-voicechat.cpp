#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path


PERCEPTION_RE = re.compile(r"perception: (\d+) frames .* in (\d+) ms")
DRAIN_RE = re.compile(r"voicechat-tts: drained (\d+) frames .* in (\d+) ms")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


class Service:
    def __init__(self, command: list[str], env: dict[str, str], stderr_path: Path):
        self.events: queue.Queue[tuple[float, dict]] = queue.Queue()
        self.stderr_metrics: queue.Queue[tuple[str, int, int]] = queue.Queue()
        self.stderr_file = stderr_path.open("w", encoding="utf-8")
        self.start = time.monotonic()
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stdout_thread.start()
        self.stderr_thread.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            when = time.monotonic()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                event = {"kind": "invalid_json", "line": line.rstrip("\n")}
            self.events.put((when, event))

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.stderr_file.write(line)
            self.stderr_file.flush()
            perception = PERCEPTION_RE.search(line)
            if perception:
                self.stderr_metrics.put(("perception", int(perception.group(1)), int(perception.group(2))))
            drain = DRAIN_RE.search(line)
            if drain:
                self.stderr_metrics.put(("drain", int(drain.group(1)), int(drain.group(2))))

    def send(self, command: dict) -> float:
        assert self.process.stdin is not None
        submitted = time.monotonic()
        self.process.stdin.write(json.dumps(command, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        return submitted

    def wait_for(self, kind: str, timeout: float = 300.0) -> tuple[float, dict]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {kind}")
            when, event = self.events.get(timeout=remaining)
            if event.get("kind") == "error":
                raise RuntimeError(event.get("message", "VoiceChat service error"))
            if event.get("kind") == kind:
                return when, event

    def reset(self) -> None:
        self.send({"cmd": "reset"})
        self.wait_for("reset")

    def system(self, text: str) -> None:
        self.send({"cmd": "system", "text": text})
        self.wait_for("system")

    def close(self) -> None:
        if self.process.poll() is None:
            self.send({"cmd": "quit"})
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.stderr_file.close()


class Telemetry:
    def __init__(self, process: subprocess.Popen, sysfs_device: Path, interval_ms: int):
        self.process = process
        self.sysfs_device = sysfs_device
        self.interval = interval_ms / 1000
        self.rows: list[dict] = []
        self.stop_event = threading.Event()
        self.start = time.monotonic()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    @staticmethod
    def _integer(path: Path) -> int | None:
        try:
            return int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return None

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            status = Path(f"/proc/{self.process.pid}/status")
            rss_kib = None
            try:
                for line in status.read_text(encoding="ascii").splitlines():
                    if line.startswith("VmRSS:"):
                        rss_kib = int(line.split()[1])
                        break
            except (OSError, ValueError):
                pass
            self.rows.append({
                "elapsed_ms": f"{(time.monotonic() - self.start) * 1000:.3f}",
                "vram_used_bytes": self._integer(self.sysfs_device / "mem_info_vram_used"),
                "gpu_busy_percent": self._integer(self.sysfs_device / "gpu_busy_percent"),
                "process_rss_kib": rss_kib,
            })
            self.stop_event.wait(self.interval)

    def close(self, path: Path) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        with path.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=["elapsed_ms", "vram_used_bytes", "gpu_busy_percent", "process_rss_kib"])
            writer.writeheader()
            writer.writerows(self.rows)


def run_turn(service: Service, audio: Path, output: Path, index: int, warmup: bool, system: str, tool_response: str) -> dict:
    if index > 0:
        service.reset()
    if system:
        service.system(system)

    submitted = service.send({"cmd": "turn", "audio": str(audio), "out": str(output)})
    first_text = None
    audio_ready = None
    wav_write_ms = None
    tool_calls = []
    result = None
    while result is None:
        when, event = service.events.get(timeout=300)
        kind = event.get("kind")
        if kind == "error":
            raise RuntimeError(event.get("message", "VoiceChat service error"))
        if kind == "assistant_text_delta" and first_text is None:
            first_text = when
        elif kind == "tool_call":
            tool_calls.append(event.get("text", ""))
            if tool_response:
                service.send({"cmd": "tool_response", "text": tool_response})
            else:
                service.send({"cmd": "tool_skip"})
        elif kind == "audio":
            audio_ready = when
            wav_write_ms = event.get("ms")
        elif kind == "turn_end":
            result = event
            completed = when

    stage = {}
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and len(stage) < 2:
        try:
            name, frames, milliseconds = service.stderr_metrics.get(timeout=0.05)
            stage[name] = (frames, milliseconds)
        except queue.Empty:
            pass

    perception_frames, perception_ms = stage.get("perception", (None, None))
    drain_frames, drain_ms = stage.get("drain", (None, None))
    output_duration = wav_seconds(output)
    total_ms = (completed - submitted) * 1000
    return {
        "run": index,
        "warmup": int(warmup),
        "input": audio.name,
        "input_seconds": f"{wav_seconds(audio):.6f}",
        "perception_frames": perception_frames,
        "perception_ms": perception_ms,
        "timeline_ms": result.get("ms"),
        "drain_frames": drain_frames,
        "drain_ms": drain_ms,
        "wav_write_ms": wav_write_ms,
        "first_text_ms": "" if first_text is None else f"{(first_text - submitted) * 1000:.3f}",
        "audio_ready_ms": "" if audio_ready is None else f"{(audio_ready - submitted) * 1000:.3f}",
        "total_ms": f"{total_ms:.3f}",
        "output_seconds": f"{output_duration:.6f}",
        "total_s2s_rtf": f"{total_ms / 1000 / output_duration:.6f}",
        "spoken_tokens": result.get("spoken"),
        "timeline_frame": result.get("t"),
        "text": result.get("text", ""),
        "tool_calls": json.dumps(tool_calls, separators=(",", ":")),
        "output_sha256": sha256(output),
        "exit_status": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic R9700 Q8 VoiceChat baseline")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--sample-ms", type=int, default=0)
    parser.add_argument("--sysfs-device", type=Path, default=Path("/sys/class/drm/card2/device"))
    parser.add_argument("--system-file", type=Path)
    parser.add_argument("--tool-response-file", type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    audio = args.audio.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not audio.is_file():
        parser.error(f"audio input does not exist: {audio}")
    system = args.system_file.read_text(encoding="utf-8").strip() if args.system_file else ""
    tool_response = args.tool_response_file.read_text(encoding="utf-8").strip() if args.tool_response_file else ""

    runtime = repo / "models/voicechat-q8/runtime"
    command = [
        str(repo / "build/hip-gfx1201/bin/llama-voicechat"),
        "-m", str(runtime / "nemotron_voicechat_11b-stt-llm-Q8_0.gguf"),
        "--mmproj", str(runtime / "mmproj-voicechat-perception-Q8_0.gguf"),
        "--tts", str(runtime / "voicechat-tts-Q8_0.gguf"),
        "--serve", "--no-warmup", "--device", "ROCm0", "--split-mode", "none",
        "--gpu-layers", "all", "--temp", "0", "--seed", "42",
    ]
    env = os.environ.copy()
    env.update({
        "ROCR_VISIBLE_DEVICES": "1",
        "GGML_CUDA_DISABLE_GRAPHS": "1",
        "VC_NO_BARGE": "1",
        "VC_FORCE_BOS": "1",
    })

    service = Service(command, env, output_dir / "service.stderr.txt")
    telemetry = Telemetry(service.process, args.sysfs_device, args.sample_ms) if args.sample_ms > 0 else None
    rows = []
    try:
        ready_at, ready = service.wait_for("ready")
        (output_dir / "ready.json").write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
        (output_dir / "cold-load-ms.txt").write_text(f"{(ready_at - service.start) * 1000:.3f}\n", encoding="utf-8")
        total = args.warmups + args.runs
        for index in range(total):
            warmup = index < args.warmups
            output = output_dir / f"turn-{index:03d}.wav"
            row = run_turn(service, audio, output, index, warmup, system, tool_response)
            rows.append(row)
            print(f"turn {index + 1}/{total}: {row['total_ms']} ms", file=sys.stderr)
    finally:
        service.close()
        if telemetry:
            telemetry.close(output_dir / "telemetry.csv")

    fields = list(rows[0])
    with (output_dir / "raw-runs.csv").open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    measured = [row for row in rows if not row["warmup"]]
    summary_fields = ["total_ms", "first_text_ms", "audio_ready_ms", "perception_ms", "timeline_ms", "drain_ms", "total_s2s_rtf"]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as dst:
        writer = csv.writer(dst)
        writer.writerow(["metric", "count", "mean", "p50", "p95", "min", "max"])
        for field in summary_fields:
            values = [float(row[field]) for row in measured if row[field] not in (None, "")]
            writer.writerow([
                field,
                len(values),
                f"{sum(values) / len(values):.3f}",
                f"{percentile(values, 0.50):.3f}",
                f"{percentile(values, 0.95):.3f}",
                f"{min(values):.3f}",
                f"{max(values):.3f}",
            ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
