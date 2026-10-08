"""``soptx.core.measure`` 阶段耗时与内存记录的测试."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from soptx.core import measure
from soptx.core.profiling import _status_kib

pytestmark = pytest.mark.skipif(not os.path.exists('/proc/self/clear_refs'),
                                reason='依赖 Linux 的 /proc 接口')


def test_records_time_and_memory_fields() -> None:
    records = {}
    with measure(records, 'sleep'):
        time.sleep(0.05)

    record = records['sleep']
    assert set(record) == {'seconds', 'peak_gib', 'net_gib', 'after_gib', 'major_faults', 'swap_gib'}
    assert isinstance(record['major_faults'], int) and record['major_faults'] >= 0
    assert record['swap_gib'] >= 0.0
    assert record['seconds'] >= 0.05
    assert record['net_gib'] >= 0.0
    assert record['peak_gib'] >= record['after_gib'] > 0.0


def test_peak_reflects_temporary_allocation() -> None:
    records = {}
    with measure(records, 'allocate'):
        block = np.ones(256 * 2 ** 20 // 8)        # 256 MiB, 写满以确保页面驻留
        del block

    # 数组在阶段内已释放, 但阶段峰值须计入这 256 MiB
    assert records['allocate']['net_gib'] >= 0.2


def test_label_prints_one_line_per_stage(capsys) -> None:
    with measure({}, '网格', label='准备'):
        pass

    output = capsys.readouterr().out
    assert output.startswith('[准备/网格] ') and '峰值' in output and output.count('\n') == 1


def test_no_record_when_block_raises() -> None:
    records = {}
    with pytest.raises(ValueError):
        with measure(records, 'broken'):
            raise ValueError('boom')
    assert 'broken' not in records


def test_missing_status_field_raises() -> None:
    with pytest.raises(RuntimeError, match='NoSuchField'):
        _status_kib('NoSuchField')
