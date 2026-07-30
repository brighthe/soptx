from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from tools import check_wheel_contents, generate_repository_inventory


def test_wheel_check_rejects_archive_module(tmp_path: Path) -> None:
    wheel = tmp_path / "soptx-test.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("soptx/__init__.py", "")
        archive.writestr(
            "soptx/optimization/mma_optimizer_backup.py",
            "",
        )

    assert check_wheel_contents.main([str(wheel)]) == 1


def test_inventory_skips_generated_directories(tmp_path: Path) -> None:
    included = tmp_path / "module.py"
    excluded = tmp_path / "outputs" / "generated.py"
    included.write_text("", encoding="utf-8")
    excluded.parent.mkdir()
    excluded.write_text("", encoding="utf-8")

    paths = generate_repository_inventory.python_files_under(tmp_path)

    assert paths == [included]
