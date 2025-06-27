import os
import psutil
import threading
import time
import sys

def memory_watchdog(limit_gb=28, check_interval=1):
    """
    Monitor the memory usage of the current process. Kill it if usage exceeds limit.
    """
    process = psutil.Process(os.getpid())
    limit_bytes = limit_gb * 1024 ** 3

    while True:
        try:
            mem = process.memory_info().rss  # Resident Set Size (RAM usage)
            if mem > limit_bytes:
                print(f"[Memory Watchdog] Exceeded {limit_gb} GB. Killing process.")
                sys.stdout.flush()
                os.kill(os.getpid(), 9)  # Send SIGKILL
        except Exception as e:
            print(f"[Memory Watchdog] Error: {e}")
        time.sleep(check_interval)

def start_memory_watchdog(limit_gb=28, check_interval=1):
    t = threading.Thread(target=memory_watchdog, args=(limit_gb, check_interval), daemon=True)
    t.start()
