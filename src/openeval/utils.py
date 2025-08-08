from __future__ import annotations

import os
import random
import hashlib
from pathlib import Path
from typing import Optional


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


def hash_file(path: Path | str, *, algo: str = "sha256", chunk_size: int = 1 << 20) -> str:
    p = Path(path)
    h = hashlib.new(algo)
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
