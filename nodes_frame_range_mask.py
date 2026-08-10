"""Solid video mask built from a frame range list.

Selecting which frames of a clip to inpaint usually means building the mask
elsewhere and hoping the frame count lines up. This node writes it directly:
a batch of fully masked or fully empty frames at the resolution you ask for.
"""

import re

import torch

from comfy_api.latest import io

CATEGORY = "MaskVidExperiments"


_NUMBER = re.compile(r"^-?\d+$")

SYNTAX = ("expected comma-separated frames or Python-style slices, "
          "e.g. 0:24, 40, 100:end")


def _bound(token, stop_position):
    if token == "":
        return None
    if token == "end":
        if not stop_position:
            raise ValueError(f"'end' can only be used as the end of a range: {SYNTAX}")
        return None
    if not _NUMBER.match(token):
        raise ValueError(f"'{token}' is not a frame number: {SYNTAX}")
    return int(token)


def parse_frame_ranges(text, frames):
    """Frame indices selected by a range list, using Python slice semantics:
    the stop frame is excluded, negative numbers count back from the end, and
    out-of-range endpoints clamp. 'end' is accepted in place of an omitted
    stop. A single frame outside the batch raises, as list indexing would."""
    selected = set()
    for segment in re.split(r"[,\n]", text):
        segment = "".join(segment.split())
        if not segment:
            continue
        if _NUMBER.match(segment):
            frame = int(segment)
            if not -frames <= frame < frames:
                raise ValueError(f"frame {frame} is outside a batch of {frames} frames")
            selected.add(frame % frames)
            continue
        parts = segment.split(":")
        if len(parts) > 3:
            raise ValueError(f"could not read frame range '{segment}': {SYNTAX}")
        start, stop = _bound(parts[0], False), _bound(parts[1], True)
        step = _bound(parts[2], False) if len(parts) == 3 else None
        if step == 0:
            raise ValueError(f"frame range '{segment}' has a step of zero")
        selected.update(range(*slice(start, stop, step).indices(frames)))
    return sorted(selected)


class MVEx_FrameRangeMaskNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MVEx_FrameRangeMask",
            display_name="MVEx Frame Range Mask",
            category=CATEGORY,
            description="Builds a batch of solid masks at the given resolution, fully masked (1) on the listed frames and empty (0) on all the others. Ranges follow Python slice syntax, so 0:24 is frames 0 to 23.",
            inputs=[
                io.Int.Input("width", default=832, min=1, max=16384, step=8,
                             tooltip="Mask width in pixels."),
                io.Int.Input("height", default=480, min=1, max=16384, step=8,
                             tooltip="Mask height in pixels."),
                io.Int.Input("frames", default=81, min=1, max=16384,
                             tooltip="Frames in the batch."),
                io.String.Input("ranges", default="0:24", multiline=True,
                                placeholder="0:24, 40, 100:end",
                                tooltip="Frames to mask, comma-separated. Ranges are Python slices written start:stop, with the stop frame excluded, so 0:24 masks frames 0 to 23. Leave a side out to run to the start or the end (5:, :3), or write 5:end for the same thing. Negative numbers count back from the end, so -1 is the last frame and 5:-1 stops one short of it. A bare number masks a single frame, an optional third field steps (0::2 is every other frame), frame numbers start at 0, and an empty list gives an empty mask."),
            ],
            outputs=[io.Mask.Output(display_name="masks", tooltip="One mask per frame.")],
        )

    @classmethod
    def execute(cls, width, height, frames, ranges) -> io.NodeOutput:
        out = torch.zeros((frames, height, width))
        out[parse_frame_ranges(ranges, frames)] = 1.0
        return io.NodeOutput(out)
