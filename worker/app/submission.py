"""Turn raw test predictions into the competition's submission format."""

import numpy as np
import pandas as pd

from .metrics import top_k
from .profiling import CompetitionData


def build(data: CompetitionData, preds: np.ndarray, fmt: str | None = None) -> pd.DataFrame:
    fmt = fmt or data.submission_format
    out = pd.DataFrame({data.id_col: data.test[data.id_col].values})
    target_cols = list(data.sample.columns[1:])

    if fmt == "value":
        out[target_cols[0]] = preds
    elif fmt == "proba":
        out[target_cols[0]] = preds[:, 1]
    elif fmt == "label":
        out[target_cols[0]] = data.classes[preds.argmax(axis=1)]
    elif fmt == "topk":
        labels = data.classes[top_k(preds, data.topk)].astype(str)
        out[target_cols[0]] = [" ".join(row) for row in labels]
    elif fmt == "proba_per_class":
        class_index = {str(c): i for i, c in enumerate(data.classes)}
        missing = [c for c in target_cols if c not in class_index]
        if missing:
            raise ValueError(f"Sample submission columns {missing} do not match target classes")
        for col in target_cols:
            out[col] = preds[:, class_index[col]]
    else:
        raise ValueError(f"Unknown submission format '{fmt}'")

    return out
