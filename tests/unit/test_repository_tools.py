from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from tools import check_wheel_contents


def test_wheel_check_rejects_archive_module(tmp_path: Path) -> None:
    wheel = tmp_path / "soptx-test.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("soptx/__init__.py", "")
        archive.writestr(
            "soptx/optimization/mma_optimizer_backup.py",
            "",
        )

    assert check_wheel_contents.main([str(wheel)]) == 1
