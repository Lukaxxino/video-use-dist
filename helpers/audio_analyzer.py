import json
import logging
import subprocess
import numpy as np
from pathlib import Path
import librosa

logger = logging.getLogger(__name__)

def extract_audio_wav(video_path: Path, output_wav_path: Path):
    """Extracts a temporary WAV file for librosa analysis."""
    if output_wav_path.exists():
        logger.info("Temporary WAV file already exists.")
        return
        
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav_path)
    ]
    logger.info("Extracting audio to WAV for analysis...")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def analyze_audio_dynamics(
    video_path: Path,
    transcript_data: dict,
    scenes: list[dict],
    output_path: Path,
) -> list[dict]:
    """Analyze per-scene dynamics without modifying static scene data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_wav_path = output_path.parent / "temp_audio.wav"

    try:
        extract_audio_wav(video_path, temp_wav_path)
        logger.info("Loading audio with librosa (this may take a moment)...")
        y, sr = librosa.load(temp_wav_path, sr=16000)

        global_rms = float(np.mean(librosa.feature.rms(y=y)))
        if global_rms == 0:
            global_rms = 0.0001

        transcript_words = [
            word
            for word in transcript_data.get("words", [])
            if word.get("type") == "word"
        ]
        audio_dynamics_result = []

        logger.info("Analyzing audio dynamics per speaker per scene...")
        for scene in scenes:
            scene_start = scene.get("start_time", 0.0)
            scene_end = scene.get("end_time", 0.0)
            scene_words = [
                word
                for word in transcript_words
                if scene_start <= word.get("start", 0.0) < scene_end
            ]

            speakers = {}
            for word in scene_words:
                speaker_id = word.get("speaker_id", "unknown")
                speaker = speakers.setdefault(
                    speaker_id,
                    {"words": [], "speaking_time": 0.0, "audio_slices": []},
                )
                speaker["words"].append(word)

                word_start = word.get("start", 0.0)
                word_end = word.get("end", 0.0)
                speaker["speaking_time"] += max(word_end - word_start, 0.01)

                sample_start = int(word_start * sr)
                sample_end = min(int(word_end * sr), len(y))
                if sample_start < len(y):
                    speaker["audio_slices"].append(y[sample_start:sample_end])

            scene_dynamics = {}
            for speaker_id, speaker in speakers.items():
                speaking_time = speaker["speaking_time"]
                words_per_second = (
                    len(speaker["words"]) / speaking_time
                    if speaking_time > 0
                    else 0
                )
                pacing = "Normal"
                if words_per_second < 2.0:
                    pacing = "Slow"
                elif words_per_second > 4.0:
                    pacing = "Fast"

                speaker_rms = 0.0
                if speaker["audio_slices"]:
                    combined_audio = np.concatenate(speaker["audio_slices"])
                    if len(combined_audio) > 0:
                        speaker_rms = float(
                            np.mean(librosa.feature.rms(y=combined_audio))
                        )

                relative_loudness = speaker_rms / global_rms
                loudness = "Normal"
                if relative_loudness < 0.5:
                    loudness = "Quiet"
                elif relative_loudness > 1.8:
                    loudness = "Loud (Shouting)"

                scene_dynamics[speaker_id] = {
                    "loudness": loudness,
                    "pacing": pacing,
                    "words_per_second": round(words_per_second, 2),
                    "relative_loudness_ratio": round(relative_loudness, 2),
                }

            audio_dynamics_result.append(
                {
                    "scene_number": scene.get("scene_number"),
                    "dynamics": scene_dynamics,
                }
            )

        output_path.write_text(
            json.dumps(audio_dynamics_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Audio dynamics saved to %s", output_path)
        return audio_dynamics_result
    finally:
        if temp_wav_path.exists():
            temp_wav_path.unlink()
