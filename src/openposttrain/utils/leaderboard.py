from pathlib import Path
from typing import Any

import pandas as pd


def append_to_leaderboard(
    leaderboard_path: str,
    row: dict[str, Any],
) -> None:
    path = Path(leaderboard_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_row_df = pd.DataFrame([row])

    if path.exists():
        existing_df = pd.read_csv(path)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        updated_df = new_row_df

    updated_df.to_csv(path, index=False)
