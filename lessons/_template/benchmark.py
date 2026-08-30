import json
from time import perf_counter

from inference_lab import implementation_status

started = perf_counter()
status = implementation_status()
duration = perf_counter() - started
print(
    json.dumps(
        {
            "lesson_id": "00-template",
            "implementation_status": status,
            "duration_seconds": duration,
        }
    )
)
