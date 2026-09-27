"""Record memory of each visible PPU while an experiment runs."""

import csv
import subprocess
import sys
import time
from datetime import datetime, timezone

with open(sys.argv[1], 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time_utc', 'ppu', 'memory_mib'])
    while True:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True,
        )
        now = datetime.now(timezone.utc).isoformat()
        for line in result.stdout.splitlines():
            index, memory = [int(x.strip()) for x in line.split(',')]
            writer.writerow([now, index, memory])
        f.flush()
        time.sleep(2)
