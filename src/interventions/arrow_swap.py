"""Swap which .arrow file acts as the test arrow (e.g.
'vqa_slack_test.arrow', 'vqa_vqa_rad_test.arrow') inside an M3AE/LaPA
finetune_arrows directory, with automatic backup/restore. Extracted from
logic that used to be hardcoded in src/run_condition_m3ae.sh (the C0/C1/C2
cases) into a helper reusable by ANY condition (including the six C4
variants) and ANY dataset supported by M3AE (VQA-RAD uses the same file
naming convention as SLAKE), and testable without GPU/torch."""
from __future__ import annotations

import contextlib
import os
import shutil

DEFAULT_TEST_ARROW_FILENAME = "vqa_slack_test.arrow"


@contextlib.contextmanager
def swap_test_arrow(m3ae_dir: str, condition_arrow_path: str,
                     test_arrow_filename: str = DEFAULT_TEST_ARROW_FILENAME):
    """Copy `condition_arrow_path` over
    `<m3ae_dir>/data/finetune_arrows/<test_arrow_filename>` for the duration
    of the `with` block, keeping the original contents in an `.orig_backup`
    file (created once, reused by later calls) and restoring it on exit —
    including on exception. Returns the path of the swapped file.
    `test_arrow_filename` defaults to the SLAKE name for backward
    compatibility — pass it explicitly for other datasets (e.g.
    'vqa_vqa_rad_test.arrow')."""
    arrow_dir = os.path.join(m3ae_dir, "data", "finetune_arrows")
    target = os.path.join(arrow_dir, test_arrow_filename)
    backup = target + ".orig_backup"

    if not os.path.isfile(target):
        raise FileNotFoundError(f"{target} does not exist")
    if not os.path.isfile(condition_arrow_path):
        raise FileNotFoundError(f"{condition_arrow_path} does not exist")

    if not os.path.isfile(backup):
        shutil.copy2(target, backup)

    # C0 is the target file itself (condition_arrow_path == target) — copying
    # a file over itself raises shutil.SameFileError. Only copy when the two
    # are in fact different files.
    if not os.path.samefile(condition_arrow_path, target):
        shutil.copy2(condition_arrow_path, target)
    try:
        yield target
    finally:
        shutil.copy2(backup, target)
