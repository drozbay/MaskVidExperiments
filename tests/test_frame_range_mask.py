"""Functional tests for Frame Range Mask."""
import importlib.util
import os
import sys

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(PACK)))  # ComfyUI root

import torch

spec = importlib.util.spec_from_file_location(
    "nodes_frame_range_mask", os.path.join(PACK, "nodes_frame_range_mask.py"))
frm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frm)

Node = frm.MVEx_FrameRangeMaskNode
parse = frm.parse_frame_ranges


failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except Exception as e:
        failures.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


def masked(text, frames=10):
    out = Node.execute(4, 3, frames, text).result[0]
    assert out.shape == (frames, 3, 4), out.shape
    assert set(out.unique().tolist()) <= {0.0, 1.0}, out.unique()
    flat = out.reshape(frames, -1)
    assert (flat.amin(1) == flat.amax(1)).all(), "frames must be solid"
    return [i for i, v in enumerate(flat.amax(1).tolist()) if v == 1.0]


def test_matches_python_slicing():
    reference = list(range(10))
    for text, expected in (("2:5", reference[2:5]),
                           ("0:24", reference[0:24]),
                           ("5:", reference[5:]),
                           (":3", reference[:3]),
                           (":", reference[:]),
                           ("5:-1", reference[5:-1]),
                           ("-3:-1", reference[-3:-1]),
                           ("2:-3", reference[2:-3]),
                           ("::2", reference[::2]),
                           ("1:8:3", reference[1:8:3]),
                           ("8:2:-1", reference[8:2:-1])):
        assert masked(text) == sorted(expected), text


def test_stop_is_excluded():
    assert parse("0:24", 81) == list(range(24))
    assert masked("0:0") == []


def test_end_keyword_includes_last_frame():
    assert masked("5:end") == [5, 6, 7, 8, 9]
    assert masked("5:") == [5, 6, 7, 8, 9]
    assert masked("0:end:4") == [0, 4, 8]


def test_single_frames():
    assert masked("0,3,9") == [0, 3, 9]
    assert masked("-1") == [9]


def test_multiple_ranges_and_overlap():
    assert masked("0:2,5:7") == [0, 1, 5, 6]
    assert masked("0:4,2:6") == [0, 1, 2, 3, 4, 5]


def test_whitespace_tolerated():
    assert masked(" 2 : 5 ,\t7 ") == [2, 3, 4, 7]
    assert masked("0:1\n4,7:9") == [0, 4, 7, 8]
    assert masked("2:5,") == [2, 3, 4]


def test_clamps_out_of_range_ranges():
    assert masked("8:200") == [8, 9]
    assert masked("50:60") == []
    assert masked("-99:2") == [0, 1]


def test_empty_ranges_select_nothing():
    assert masked("3:1") == []
    assert masked("-2:-4") == []


def test_empty_list_is_empty_mask():
    assert masked("") == []
    assert masked("  \n ") == []


def test_rejects_bad_input():
    for text in ("0:zz", "a:b", "1:2:3:4", "0:1:0", "end:5", "end", "0-5", "0..5", "1;2"):
        try:
            parse(text, 10)
        except ValueError:
            continue
        raise AssertionError(f"accepted {text!r}")


def test_rejects_single_frame_out_of_range():
    for text in ("10", "-11"):
        try:
            parse(text, 10)
        except ValueError:
            continue
        raise AssertionError(f"accepted {text!r}")


def test_default_widget_value_parses():
    schema = Node.define_schema()
    ranges = next(i for i in schema.inputs if i.id == "ranges")
    assert parse(ranges.default, 81) == list(range(24))
    assert parse(ranges.placeholder, 200) == list(range(24)) + [40] + list(range(100, 200))


def test_resolution_and_dtype():
    out = Node.execute(64, 32, 5, "1:2").result[0]
    assert out.shape == (5, 32, 64), out.shape
    assert out.dtype == torch.float32, out.dtype


for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
    check(name, fn)

if failures:
    print(f"\n{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("\nall tests passed")
