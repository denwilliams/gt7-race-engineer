import signal
import subprocess
import sys
import time


def test_single_ctrl_c_exits_promptly():
    p = subprocess.Popen(
        [sys.executable, "-m", "gt7_engineer.cli", "run", "--simulate", "--no-audio", "--no-llm", "--laps", "50"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    time.sleep(2.5)
    t = time.time()
    p.send_signal(signal.SIGINT)
    assert p.wait(timeout=5) == 0
    assert time.time() - t < 2
    assert "Stopped." in p.stdout.read()
