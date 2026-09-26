"""Run the C engine (host/tinyai) on many questions, split across CPU cores."""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor


def _run(binary, model, flags, chunk):
    out = subprocess.run([binary, model, *flags], input="\n".join(chunk) + "\n", capture_output=True,
                         text=True, check=True).stdout
    lines = out.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # only the final newline; empty answers stay as rows
    assert len(lines) == len(chunk), f"{len(lines)} answers for {len(chunk)} questions"
    return lines


def ask(binary, model, questions, *flags, jobs=None):
    """One output line per question, in order: the engine's answer, or with
    --gate the routed fact key ("=answer" if computed, "-" if refused)."""
    questions = list(questions)
    if not questions:
        return []
    jobs = max(1, min(jobs or os.cpu_count() or 1, len(questions) // 200 or 1))
    size = -(-len(questions) // jobs)
    chunks = [questions[i:i + size] for i in range(0, len(questions), size)]
    with ThreadPoolExecutor(len(chunks)) as pool:
        parts = pool.map(lambda c: _run(binary, model, flags, c), chunks)
    return [line for part in parts for line in part]
