import glob
import pandas as pd
from pathlib import Path


class Dataset:
    def __init__(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)
        # print(str(path) + "**/*.csv")
        self.datafiles = glob.glob(str(path / "**" / "*.csv"), recursive=True)

    def __getitem__(self, index: int) -> dict[str, pd.DataFrame]:
        if index >= len(self.datafiles):
            raise IndexError("Index out of range")
        data = pd.read_csv(self.datafiles[index])
        data.timestamp = pd.to_datetime(data.timestamp, unit='ms')
        return {'csv_path': self.datafiles[index], 'time_series': data}

    def __len__(self) -> int:
        return len(self.datafiles)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
