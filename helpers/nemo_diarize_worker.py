import os
# Defaults only: GPU 0 fits a single-GPU workstation (a hardcoded "1" left
# this worker on CPU there). A multi-GPU box sets these in its environment.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Diarizes a list of audio windows in an isolated process. Run as "
            "a throwaway subprocess by nemo_server.py: calling "
            "SortformerEncLabelModel.diarize() and ASRModel.transcribe() in "
            "the same process hard-crashes it (a native CUDA abort, not a "
            "catchable exception) on this torch/nemo_toolkit/Windows stack, "
            "reproduced and isolated separately — repeated calls to either "
            "model alone are stable indefinitely, but alternating between "
            "them is not. Running the diarizer here, in a process that never "
            "touches the ASR model, sidesteps that entirely."
        )
    )
    ap.add_argument("--diar-model-name", required=True)
    ap.add_argument(
        "--window",
        action="append",
        default=[],
        dest="windows",
        help="Path to a window audio file; repeat in order, one per window",
    )
    ap.add_argument(
        "--window-list",
        type=Path,
        default=None,
        help=(
            "File with one window path per line (appended after any "
            "--window); avoids Windows' 32767-char command-line limit on "
            "long videos"
        ),
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path: a list of raw diarize() segment lists, one per window",
    )
    args = ap.parse_args()
    if args.window_list:
        args.windows += _read_window_list(args.window_list)
    if not args.windows:
        ap.error("no windows given (use --window and/or --window-list)")

    import numpy as np
    if not hasattr(np, "sctypes"):
        # NumPy >=2.0 removed np.sctypes; some installed nemo_toolkit builds'
        # AudioSegment loader (segment.py::_convert_samples_to_float32) still
        # reads it. Restore the exact pre-2.0 shape rather than pinning
        # numpy<2.0, which conflicts with pyannote-core/librosa/scipy's own
        # numpy>=2.0 requirement.
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [bool, object, bytes, str, np.void],
        }

    from nemo.collections.asr.models import SortformerEncLabelModel

    import os
    try:
        diar_model = SortformerEncLabelModel.from_pretrained(args.diar_model_name)
    except Exception as e:
        if "NVIDIA driver" in str(e) or "too old" in str(e) or "CUDA" in str(e):
            print(f"[Warning] CUDA driver incompat: {e}. Falling back to CPU for Diarization.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            diar_model = SortformerEncLabelModel.from_pretrained(args.diar_model_name)
        else:
            raise
    diar_model.eval()

    all_segments = []
    for window_path in args.windows:
        raw = diar_model.diarize(audio=[window_path], batch_size=1)
        all_segments.append(raw[0])

    args.output.write_text(json.dumps(all_segments), encoding="utf-8")


def _read_window_list(path):
    """One window path per line; blank lines ignored; a UTF-8 BOM (as
    PowerShell writes by default) is tolerated."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return [line.strip() for line in lines if line.strip()]


if __name__ == "__main__":
    main()
