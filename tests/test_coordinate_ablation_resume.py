from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from experiments import run_coordinate_ablation as ablation


class CoordinateAblationResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory(prefix="abc_resume_test_")
        self.addCleanup(directory.cleanup)
        self.out_dir = Path(directory.name)
        stem = "coord_ablation_resume_test"
        self.csv_path = self.out_dir / f"{stem}_trials.csv"
        self.metadata_path = self.out_dir / f"{stem}_metadata.json"
        self.summary_path = self.out_dir / f"{stem}_summary.json"

    @staticmethod
    def _record(replicate: int) -> dict[str, float | int]:
        record = {field: 0.5 for field in ablation.FIELDS}
        record.update(
            replicate=replicate, valid=1.0, fitted_sigma=0.42, selected_width=8.0
        )
        return record

    def _write_trials(self, fields: tuple[str, ...], count: int = 2) -> None:
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            for replicate in range(count):
                record = self._record(replicate)
                writer.writerow([record.get(field, 0.5) for field in fields])

    def _args(self, *, replicates: int = 3, overwrite: bool = False):
        argv = [
            "run_coordinate_ablation.py",
            "--condition-tag", "resume_test",
            "--out-dir", str(self.out_dir),
            "--replicates", str(replicates),
            "--batch-size", "2",
        ]
        if overwrite:
            argv.append("--overwrite")
        with patch.object(sys, "argv", argv):
            return ablation.parse_args()

    def _write_checkpoint(self, args, completed: int) -> None:
        metadata = ablation._metadata(args)
        metadata["completed_replicates"] = completed
        self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        self.summary_path.write_text(
            json.dumps({"n": completed}), encoding="utf-8"
        )

    def _run_with_fake_trials(self, args) -> Mock:
        # Only mock the costly experiment, leaving CSV/metadata/resume real.
        row = np.asarray([self._record(0)[field] for field in ablation.FIELDS[1:]])
        trial = Mock(side_effect=lambda keys: np.tile(row, (len(keys), 1)))
        with (
            patch.object(ablation, "parse_args", return_value=args),
            patch.object(ablation, "make_batched_trial", return_value=trial),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            ablation.main()
        return trial

    def test_legacy_schema_is_rejected_without_modifying_file(self) -> None:
        legacy_fields = tuple(field for field in ablation.FIELDS if field != "valid")
        for count in (0, 500):
            with self.subTest(rows=count):
                self._write_trials(legacy_fields, count)
                before = self.csv_path.read_bytes()
                with self.assertRaisesRegex(ValueError, "incompatible.*schema"):
                    ablation._csv_row_count(self.csv_path, overwrite=False)
                self.assertEqual(self.csv_path.read_bytes(), before)

    def test_same_width_schema_changes_are_rejected(self) -> None:
        renamed = (*ablation.FIELDS[:-1], "unrecognized_energy")
        reordered = (ablation.FIELDS[1], ablation.FIELDS[0], *ablation.FIELDS[2:])
        duplicated = (*ablation.FIELDS[:-1], ablation.FIELDS[-2])
        for case, fields in (("renamed", renamed), ("reordered", reordered), ("duplicated", duplicated)):
            with self.subTest(case=case):
                self._write_trials(fields)
                before = self.csv_path.read_bytes()
                with self.assertRaisesRegex(ValueError, "incompatible.*schema"):
                    ablation._csv_row_count(self.csv_path, overwrite=False)
                self.assertEqual(self.csv_path.read_bytes(), before)

    def test_current_schema_counts_complete_rows_without_writing(self) -> None:
        for tail in (b"", b"2,1.0,0."):
            with self.subTest(tail=tail):
                self._write_trials(ablation.FIELDS)
                with self.csv_path.open("ab") as handle:
                    handle.write(tail)
                before = self.csv_path.read_bytes()
                self.assertEqual(
                    ablation._csv_row_count(self.csv_path, overwrite=False), 2
                )
                self.assertEqual(self.csv_path.read_bytes(), before)

    def test_missing_empty_or_current_header_only_csv_starts_at_zero(self) -> None:
        self.assertEqual(ablation._csv_row_count(self.csv_path, overwrite=False), 0)
        self.csv_path.touch()
        self.assertEqual(ablation._csv_row_count(self.csv_path, overwrite=False), 0)
        self._write_trials(ablation.FIELDS, count=0)
        self.assertEqual(ablation._csv_row_count(self.csv_path, overwrite=False), 0)

    def test_main_rejects_legacy_before_mutating_any_outputs(self) -> None:
        args = self._args()
        self._write_trials(tuple(field for field in ablation.FIELDS if field != "valid"))
        self._write_checkpoint(args, completed=2)
        before = {
            path: path.read_bytes()
            for path in (self.csv_path, self.metadata_path, self.summary_path)
        }
        with (
            patch.object(ablation, "parse_args", return_value=args),
            patch.object(
                ablation, "make_batched_trial",
                side_effect=AssertionError("Must reject before building the experiment"),
            ) as make_trial,
            patch.object(ablation, "_truncate_csv", wraps=ablation._truncate_csv) as truncate,
            self.assertRaisesRegex(ValueError, "incompatible.*schema"),
        ):
            ablation.main()
        make_trial.assert_not_called()
        truncate.assert_not_called()
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_main_resumes_current_schema_after_partial_tail(self) -> None:
        args = self._args()
        self._write_trials(ablation.FIELDS)
        intact_prefix = self.csv_path.read_bytes()
        with self.csv_path.open("ab") as handle:
            handle.write(b"2,1.0,0.")
        # CSV flush precedes metadata: resume from its two intact records.
        self._write_checkpoint(args, completed=1)
        trial = self._run_with_fake_trials(args)
        trial.assert_called_once()
        self.assertTrue(self.csv_path.read_bytes().startswith(intact_prefix))
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["replicate"] for row in rows], ["0", "1", "2"])
        self.assertEqual(rows[-1]["valid"], "1.0")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        summary = json.loads(self.summary_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["completed_replicates"], 3)
        self.assertEqual(summary["n"], 3)

    def test_main_creates_new_current_schema_csv(self) -> None:
        self._run_with_fake_trials(self._args(replicates=2))
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(tuple(reader.fieldnames), ablation.FIELDS)
        self.assertEqual([row["replicate"] for row in rows], ["0", "1"])

    def test_main_explicit_overwrite_can_replace_legacy_schema(self) -> None:
        self._write_trials(tuple(field for field in ablation.FIELDS if field != "valid"))
        self._write_checkpoint(self._args(), completed=2)
        self._run_with_fake_trials(self._args(replicates=1, overwrite=True))
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(tuple(reader.fieldnames), ablation.FIELDS)
        self.assertEqual([row["replicate"] for row in rows], ["0"])
        self.assertEqual(rows[0]["valid"], "1.0")


if __name__ == "__main__":
    unittest.main()
