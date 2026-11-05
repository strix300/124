"""Измеряет полный HTTP pipeline и GPU без изменения настроек или правил проверки."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

from create_example import create_example
from smoke import smoke


def benchmark(api_url: str, output_dir: Path) -> None:
    """Сохраняет ответы, стадии и выборки GPU; сбои не считаются успешным замером."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / "example.pdf"
    create_example(pdf)
    samples_path = output_dir / "gpu.csv"
    command = [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
        "--loop-ms=250",
    ]
    subprocess.run(["nvidia-smi", "--id=0"], check=True, capture_output=True, timeout=10)
    with samples_path.open("w", encoding="utf-8") as samples:
        monitor = subprocess.Popen(command, stdout=samples, stderr=subprocess.PIPE, text=True)
        try:
            smoke(pdf, api_url, output_dir / "answers.json", timeout=1300, warm=True)
        finally:
            failed = monitor.poll() is not None
            monitor.terminate()
            _, error = monitor.communicate(timeout=10)
        if failed:
            raise RuntimeError(f"Монитор GPU завершился раньше измерений: {error}")
    with samples_path.open(encoding="utf-8") as source:
        rows = list(csv.reader(source))
    if not rows:
        raise RuntimeError("Не получены измерения GPU.")
    summary = {
        "sample_interval_ms": 250,
        "samples": len(rows),
        "peak_utilization_percent": max(float(row[1]) for row in rows),
        "peak_vram_mib": max(float(row[2]) for row in rows),
        "total_vram_mib": float(rows[0][3]),
        "scope": "Вся GPU, включая Windows и другие процессы; выборочный максимум.",
    }
    (output_dir / "gpu-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://api:8000")
    parser.add_argument("--output-dir", type=Path, default=Path(".data/gpu-benchmark"))
    arguments = parser.parse_args()
    benchmark(arguments.api_url, arguments.output_dir)
