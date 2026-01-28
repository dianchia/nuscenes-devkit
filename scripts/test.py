import logging
from typing import Any

from nuscenes._lib import Tables


class NuScenes:
    """A minimal wrapper around the nuScenes tables.
    This is just to make sure that the tables are loaded correctly, and other functionality is working.
    """

    def __init__(self, version: str, dataroot: str, verbose: bool = False):
        self._logger = logging.getLogger("nuscenes")
        self._logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        self._tables = Tables(version, dataroot)

    def get(self, table_name: str, token: str) -> dict[str, Any]:
        return self._tables.get(table_name, token)

    def field2token(self, table_name: str, field: str, query: Any) -> list[str]:
        return [m["token"] for m in getattr(self._tables, table_name) if m[field] == query]

    @property
    def scene(self) -> list[dict[str, Any]]:
        return self._tables.scene

    @property
    def sample(self) -> list[dict[str, Any]]:
        return self._tables.sample

    @property
    def sensor(self) -> list[dict[str, Any]]:
        return self._tables.sensor


def main():
    nusc = NuScenes("v1.0-mini", "data/")

    scene = nusc.scene[0]
    print(scene)

    sample = nusc.get("sample", scene["first_sample_token"])
    print(sample)

    sensors = nusc.sensor[:3]
    print(sensors)

    sample = nusc.sample["abc"]  # This should fail
    print(sample)


if __name__ == "__main__":
    main()
