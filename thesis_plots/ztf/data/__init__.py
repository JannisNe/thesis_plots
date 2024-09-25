from pathlib import Path
import json


data_dir = Path(__file__).parent


def load_lines():
    file = data_dir / "lines.json"
    with file.open("r") as f:
        return json.load(f)


band_colors = {
    "g": "C2", "r": "C3", "i": "C5"
}