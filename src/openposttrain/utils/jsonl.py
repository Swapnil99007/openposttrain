import json
from pathlib import Path
from typing import Any


def write_jsonl(records: list[dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
