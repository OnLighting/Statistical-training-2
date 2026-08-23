from pathlib import Path
import sys

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from data_preprocessing import normalize_export_fields


def test_normalize_export_fields_rebuilds_date_and_clock_from_clean_timestamps():
    data = pd.DataFrame(
        {
            "date_raw": [45698, "13/09/2025", None],
            "time_raw": ["0700", 5.87, None],
            "base_date": pd.to_datetime(["2025-02-10", "2025-09-13", "2026-02-20"]),
            "timestamp": pd.to_datetime(
                ["2025-02-10 07:00", "2025-09-13 17:00", "2026-02-21 05:00"]
            ),
        }
    )

    normalized = normalize_export_fields(data)

    assert normalized["date_raw"].tolist() == [
        "2025-02-10",
        "2025-09-13",
        "2026-02-20",
    ]
    assert normalized["time_raw"].tolist() == ["07:00", "17:00", "05:00"]
    assert data["date_raw"].tolist() == [45698, "13/09/2025", None]
