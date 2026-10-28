"""
DOP-BIN 2.0 ADC reading (file version 6, multiplexing_mode 2, per-frame tx_map)
and the legacy behaviour it must keep.

Run from the repository root:  python -m unittest discover -s tests -v
"""

import contextlib
import io
import os
import shutil
import tempfile
import unittest

import numpy as np

from python_parser import parse_dopplium, parse_dopplium_raw, parse_tx_map, split_pattern_slots
from python_parser.parse_dopplium_header import parse_file_header

from dop_bin_fixtures import frame_samples, write_adc_file

TDMA = [1, 2, 0, 0]
TX1 = [1, 0, 0, 0]
TX2 = [2, 0, 0, 0]
TX1122 = [1, 1, 2, 2]
NO_TX = [0, 0, 0, 0]
INVALID = [1, 0, 2, 0]

C, R, N = 8, 4, 8


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="dopbin2_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def parse(self, path, **kwargs):
        return parse_dopplium_raw(path, verbose=0, **kwargs)


class TxMapFieldTests(unittest.TestCase):
    def test_pattern_ends_at_first_zero(self):
        self.assertEqual(parse_tx_map(TDMA), (1, 2))
        self.assertEqual(parse_tx_map(TX1), (1,))
        self.assertEqual(parse_tx_map(TX2), (2,))
        self.assertEqual(parse_tx_map(TX1122), (1, 1, 2, 2))
        self.assertEqual(parse_tx_map(bytes([3, 255, 0, 0])), (3, 255))

    def test_all_zero_means_no_transmitter_information(self):
        self.assertEqual(parse_tx_map(NO_TX), ())

    def test_entries_after_terminating_zero_are_rejected(self):
        for raw in (INVALID, [0, 1, 0, 0], [1, 2, 0, 3], [0, 0, 0, 1]):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "terminating 0"):
                parse_tx_map(raw)


class Mode2FileTests(_TempDirCase):
    FRAMES = [TDMA, TX1, TDMA, TX2, TX1122, NO_TX]

    def test_mixed_patterns_decode_in_transmission_order(self):
        path = self.path("mixed.dop")
        expected = write_adc_file(path, frames=self.FRAMES, n_chirps_on_wire=C, n_rx=R, n_samples=N)
        data, headers = self.parse(path)

        self.assertEqual(headers["file"].version, 6)
        self.assertEqual(headers["file"].frame_header_size, 32)
        self.assertEqual(headers["body"].config_version, 4)
        self.assertEqual(headers["body"].multiplexing_mode, 2)
        self.assertEqual(headers["body"].n_transmitters, 2)
        self.assertEqual(data.shape, (N, C, R, len(self.FRAMES)))
        for f, samples in enumerate(expected):
            # samples is (chirp, rx, sample); data is [samples, chirps, receivers, frames]
            np.testing.assert_array_equal(data[:, :, :, f], np.transpose(samples, (2, 0, 1)))

        self.assertEqual([fr.tx_map for fr in headers["frame"]],
                         [(1, 2), (1,), (1, 2), (2,), (1, 1, 2, 2), ()])
        self.assertEqual([fr.frame_number for fr in headers["frame"]], list(range(100, 106)))
        self.assertTrue(all(fr.frame_header_size == 32 for fr in headers["frame"]))

    def test_dispatcher_routes_version_6(self):
        path = self.path("dispatch.dop")
        write_adc_file(path, frames=[TDMA, TX1], n_chirps_on_wire=C, n_rx=R, n_samples=N)
        data, headers = parse_dopplium(path, verbose=0)
        self.assertEqual(data.shape, (N, C, R, 2))
        self.assertEqual(headers["frame"][1].tx_map, (1,))

    def test_verbose_summary_lists_patterns(self):
        path = self.path("verbose.dop")
        write_adc_file(path, frames=[TDMA, TX1, TDMA, NO_TX], n_chirps_on_wire=C, n_rx=R, n_samples=N)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            parse_dopplium_raw(path, verbose=2)
        text = out.getvalue()
        self.assertIn("MultiplexingMode=TDM, per-frame tx_map (2)", text)
        self.assertIn("tx_map patterns (frames): [1, 2] x2, [1] x1, [] x1", text)
        self.assertIn("(8, 8, 4, 4)  [samples, chirps, receivers, frames]", text)

    def test_start_frame_uses_32_byte_stride(self):
        path = self.path("stride.dop")
        write_adc_file(path, frames=self.FRAMES, n_chirps_on_wire=C, n_rx=R, n_samples=N)
        full, _ = self.parse(path)
        part, headers = self.parse(path, start_frame=3, max_frames=2)
        np.testing.assert_array_equal(part, full[..., 3:5])
        self.assertEqual([fr.tx_map for fr in headers["frame"]], [(2,), (1, 1, 2, 2)])
        self.assertEqual([fr.frame_number for fr in headers["frame"]], [103, 104])

    def test_split_pattern_slots(self):
        path = self.path("slots.dop")
        write_adc_file(path, frames=self.FRAMES, n_chirps_on_wire=C, n_rx=R, n_samples=N)
        data, headers = self.parse(path)

        for f, fr in enumerate(headers["frame"]):
            if not fr.tx_map:
                with self.assertRaisesRegex(ValueError, "no transmitter information"):
                    split_pattern_slots(data[..., f], fr.tx_map)
                continue
            L = len(fr.tx_map)
            slots = split_pattern_slots(data[..., f], fr.tx_map)
            self.assertEqual(slots.shape, (N, C // L, L, R))
            for m in range(C // L):
                for k in range(L):
                    np.testing.assert_array_equal(slots[:, m, k, :], data[:, m * L + k, :, f])

        # TDMA frame: slot 0 holds the even (TX1) chirps, slot 1 the odd (TX2) chirps.
        tdma = split_pattern_slots(data[..., 0], headers["frame"][0].tx_map)
        np.testing.assert_array_equal(tdma[:, :, 0, :], data[:, 0::2, :, 0])
        np.testing.assert_array_equal(tdma[:, :, 1, :], data[:, 1::2, :, 0])

        with self.assertRaisesRegex(ValueError, "does not divide"):
            split_pattern_slots(np.zeros((N, 6, R)), (1, 1, 2, 2))

    def test_invalid_tx_map_frame_is_rejected(self):
        path = self.path("invalid.dop")
        write_adc_file(path, frames=[TDMA, TX1, TX2, INVALID, TDMA], n_chirps_on_wire=C, n_rx=R, n_samples=N)
        with self.assertRaisesRegex(ValueError, r"Frame 4: invalid tx_map \[1, 0, 2, 0\]"):
            self.parse(path)
        # The frames around it still read, at the right offsets.
        before, headers = self.parse(path, max_frames=3)
        self.assertEqual([fr.tx_map for fr in headers["frame"]], [(1, 2), (1,), (2,)])
        after, headers = self.parse(path, start_frame=4)
        self.assertEqual(headers["frame"][0].tx_map, (1, 2))
        np.testing.assert_array_equal(after[..., 0], np.transpose(frame_samples(4, C, R, N), (2, 0, 1)))

    def test_pattern_length_must_divide_chirps(self):
        path = self.path("length3.dop")
        write_adc_file(path, frames=[TDMA, [1, 2, 1, 0]], n_chirps_on_wire=C, n_rx=R, n_samples=N)
        with self.assertRaisesRegex(ValueError, r"Frame 2: tx_map \[1, 2, 1\] has length 3"):
            self.parse(path)

    def test_mode_2_needs_version_6_and_config_version_4(self):
        cases = {
            "file version 5": dict(version=5),
            "config_version 3": dict(config_version=3),
            "24-byte payload header": dict(payload_header_size=24),
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                path = self.path("bad_gate.dop")
                frames = [None, None] if overrides.get("payload_header_size") == 24 else [TDMA, TX1]
                write_adc_file(path, frames=frames, n_chirps_on_wire=C, n_rx=R, n_samples=N, **overrides)
                with self.assertRaisesRegex(ValueError, "multiplexing_mode 2 requires"):
                    self.parse(path)

    def test_n_chirps_per_frame_must_count_every_chirp(self):
        path = self.path("per_tx_count.dop")
        write_adc_file(path, frames=[TDMA], n_chirps_on_wire=C, n_chirps_field=C // 2, n_rx=R, n_samples=N)
        with self.assertRaisesRegex(ValueError, "does not hold n_chirps_per_frame=4"):
            self.parse(path)

    def test_unknown_multiplexing_mode_is_unsupported(self):
        path = self.path("mode3.dop")
        write_adc_file(path, frames=[TDMA], n_chirps_on_wire=C, n_rx=R, n_samples=N,
                       multiplexing_mode=3, payload_header_size=32)
        with self.assertRaisesRegex(NotImplementedError, "multiplexing_mode=3"):
            self.parse(path)

    def test_unknown_file_version_is_rejected(self):
        path = self.path("v7.dop")
        write_adc_file(path, frames=[TDMA], n_chirps_on_wire=C, n_rx=R, n_samples=N, version=7)
        with self.assertRaisesRegex(ValueError, "Unsupported file version: 7"):
            parse_dopplium(path, verbose=0)

    def test_complex_and_by_chirp_layouts(self):
        for sample_type, data_order in ((1, 0), (0, 1), (1, 1)):
            with self.subTest(sample_type=sample_type, data_order=data_order):
                path = self.path(f"layout_{sample_type}_{data_order}.dop")
                expected = write_adc_file(path, frames=[TDMA, TX2], n_chirps_on_wire=C, n_rx=R, n_samples=N,
                                          sample_type=sample_type, data_order=data_order)
                data, headers = self.parse(path)
                self.assertEqual(data.shape, (N, C, R, 2))
                for f, samples in enumerate(expected):
                    want = np.transpose(samples, (2, 0, 1)).astype(np.float32)
                    if sample_type == 1:
                        want = want - 1j * want
                    np.testing.assert_array_equal(data[..., f], want)
                self.assertEqual([fr.tx_map for fr in headers["frame"]], [(1, 2), (2,)])

    def test_big_endian(self):
        path = self.path("be.dop")
        expected = write_adc_file(path, frames=[TX1122, TX1], n_chirps_on_wire=C, n_rx=R, n_samples=N, ep=">")
        data, headers = self.parse(path)
        np.testing.assert_array_equal(data[..., 1], np.transpose(expected[1], (2, 0, 1)))
        self.assertEqual([fr.tx_map for fr in headers["frame"]], [(1, 1, 2, 2), (1,)])


class LegacyFileTests(_TempDirCase):
    """Files with file version <= 5 and 24-byte payload headers decode as before."""

    def assert_mimo_grouping(self, data, expected, n_tx, chirps_per_tx):
        self.assertEqual(data.shape, (N, chirps_per_tx, n_tx * R, len(expected)))
        for f, samples in enumerate(expected):
            for c in range(chirps_per_tx * n_tx):
                tx, c_tx = c % n_tx, c // n_tx
                for r in range(R):
                    np.testing.assert_array_equal(data[:, c_tx, tx * R + r, f], samples[c, r, :])

    def test_mimo_versions_4_and_5(self):
        for version in (4, 5):
            with self.subTest(version=version):
                path = self.path(f"legacy_v{version}.dop")
                expected = write_adc_file(path, frames=[None] * 3, n_chirps_on_wire=C, n_chirps_field=C // 2,
                                          n_rx=R, n_samples=N, n_tx=2, version=version, config_version=3,
                                          multiplexing_mode=0)
                data, headers = parse_dopplium(path, verbose=0)
                self.assertEqual(headers["file"].version, version)
                self.assertEqual(headers["body"].multiplexing_mode, 0)
                self.assert_mimo_grouping(data, expected, n_tx=2, chirps_per_tx=C // 2)
                self.assertTrue(all(fr.tx_map is None for fr in headers["frame"]))

    def test_version_5_header_fields(self):
        path = self.path("legacy_v5_header.dop")
        write_adc_file(path, frames=[None] * 3, n_chirps_on_wire=C, n_rx=R, n_samples=N, n_tx=1,
                       version=5, config_version=3, multiplexing_mode=0)
        fh, _ = parse_file_header(path)
        self.assertEqual((fh.version, fh.file_header_size, fh.frame_header_size), (5, 128, 24))
        self.assertEqual(fh.total_frames_written, 3)
        self.assertEqual(fh.total_payload_bytes, 3 * C * R * N * 2)
        self.assertEqual(fh.reserved1, 0)
        self.assertEqual(fh.node_id, "test-node")
        self.assertEqual((fh.latitude_e7, fh.longitude_e7, fh.height_cm), (415000000, -37000000, 1200))

    def test_version_2_single_tx(self):
        path = self.path("legacy_v2.dop")
        expected = write_adc_file(path, frames=[None] * 2, n_chirps_on_wire=C, n_rx=R, n_samples=N, n_tx=1,
                                  version=2, config_version=2, multiplexing_mode=0)
        data, headers = parse_dopplium(path, verbose=0)
        self.assertEqual(headers["file"].version, 2)
        self.assertEqual(data.shape, (N, C, R, 2))
        for f, samples in enumerate(expected):
            np.testing.assert_array_equal(data[..., f], np.transpose(samples, (2, 0, 1)))

    def test_beamforming_mode_keeps_file_wide_grouping(self):
        path = self.path("legacy_bf.dop")
        expected = write_adc_file(path, frames=[None] * 2, n_chirps_on_wire=C, n_chirps_field=C // 2,
                                  n_rx=R, n_samples=N, n_tx=2, version=5, config_version=3,
                                  multiplexing_mode=1)
        data, headers = self.parse(path)
        self.assertEqual(headers["body"].multiplexing_mode, 1)
        self.assert_mimo_grouping(data, expected, n_tx=2, chirps_per_tx=C // 2)


if __name__ == "__main__":
    unittest.main()
