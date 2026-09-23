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
            "Detects the spoken language of each audio window in an isolated "
            "process. Run as a throwaway subprocess by canary_server.py, "
            "before any ASR calls, for the same reason nemo_diarize_worker.py "
            "runs diarization in its own subprocess: calling a second NeMo "
            "model's inference (this LID model, or the Sortformer diarizer) "
            "in the same process as a warm ASR model's .transcribe() calls "
            "hard-crashes it (a native CUDA abort, not a catchable exception) "
            "on this torch/nemo_toolkit/Windows stack once the two models' "
            "calls alternate. This subprocess pass runs to completion (and "
            "exits) before canary_server.py loads/calls the ASR model, and "
            "the diarization subprocess still runs after that, so the three "
            "passes (LID subprocess -> ASR in-process -> diarization "
            "subprocess) never interleave with each other."
        )
    )
    ap.add_argument("--langid-model-name", required=True)
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
        help=(
            'Output JSON path: a list of {"language": <ISO 639-1 code>, '
            '"confidence": <float 0-1>} dicts, one per window, in the same '
            "order the --window flags were given"
        ),
    )
    args = ap.parse_args()
    if args.window_list:
        args.windows += _read_window_list(args.window_list)
    if not args.windows:
        ap.error("no windows given (use --window and/or --window-list)")

    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    # nvidia/langid_ambernet (VoxLingua107-based, 107 ISO 639-1-style codes,
    # loaded through the same EncDecSpeakerLabelModel class NeMo uses for
    # speaker verification/titanet — LID and speaker-ID are the same model
    # family, just trained on different label sets). Verified interactively
    # before writing this worker: `EncDecSpeakerLabelModel.from_pretrained(
    # model_name="langid_ambernet")` restores from NVIDIA's NGC model
    # registry (not the HF hub id `nvidia/langid_ambernet`, which is not how
    # this checkpoint is distributed), and its `.get_label()` /
    # `.infer_segment()` calls are the same inference API this worker uses.
    import os
    try:
        langid_model = EncDecSpeakerLabelModel.from_pretrained(model_name=args.langid_model_name)
    except Exception as e:
        if "NVIDIA driver" in str(e) or "too old" in str(e) or "CUDA" in str(e):
            print(f"[Warning] CUDA driver incompat: {e}. Falling back to CPU for LID.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            langid_model = EncDecSpeakerLabelModel.from_pretrained(model_name=args.langid_model_name)
        else:
            raise
    langid_model.eval()

    labels = list(langid_model._cfg["train_ds"].get("labels"))
    target_sr = langid_model._cfg["train_ds"].get("sample_rate", 16000)

    results = []
    for window_path in args.windows:
        audio, sr = sf.read(window_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != target_sr:
            audio = librosa.core.resample(audio, orig_sr=sr, target_sr=target_sr)

        # infer_segment (not get_label) so this worker can report a
        # confidence alongside the label: get_label only does an internal
        # argmax and majority-votes across random sub-segments, discarding
        # the softmax probability canary_server.py needs to decide whether
        # to trust a window's detected language or fall back to the
        # pipeline's configured language.
        _emb, logits = langid_model.infer_segment(audio)
        probs = torch.softmax(logits, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)
        results.append({"language": labels[int(top_idx)], "confidence": float(top_prob)})

    args.output.write_text(json.dumps(results), encoding="utf-8")


def _read_window_list(path):
    """One window path per line; blank lines ignored; a UTF-8 BOM (as
    PowerShell writes by default) is tolerated."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return [line.strip() for line in lines if line.strip()]


if __name__ == "__main__":
    main()
