"""Convert an EDL JSON to a dynamic FCP 7 XML (xmeml) for DaVinci Resolve.

Usage:
    python helpers/export_fcpxml.py <edl.json> -o timeline.xml
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2

def get_rate_info(fps: float):
    """Map actual fps to FCP 7 XML timebase and ntsc flag."""
    r_fps = round(fps, 3)
    if r_fps == 23.976:
        return 24, "TRUE"
    elif r_fps == 24.0:
        return 24, "FALSE"
    elif r_fps == 25.0:
        return 25, "FALSE"
    elif r_fps == 29.97:
        return 30, "TRUE"
    elif r_fps == 30.0:
        return 30, "FALSE"
    elif r_fps == 50.0:
        return 50, "FALSE"
    elif r_fps == 59.94:
        return 60, "TRUE"
    elif r_fps == 60.0:
        return 60, "FALSE"
    else:
        return int(round(fps)), "FALSE"

def frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))

def get_video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Warning: Could not open {path} with cv2. Using defaults.")
        return 25.0, 1920, 1080, 900000
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    cap.release()
    
    if fps <= 0: fps = 25.0
    if width <= 0: width = 1920
    if height <= 0: height = 1080
    
    # Calculate duration in frames based on exact fps
    if frame_count <= 0:
        frame_count = 900000 # Fallback 10 hours
        
    return fps, width, height, int(frame_count)

def add_source_file(parent, f_info):
    """Append a full <file> element (name, URL-encoded pathurl, rate, timecode, media
    with both video and audio) to `parent`. Used on the video AND audio clipitems so
    DaVinci Resolve can link the audio — a bare <file idref> leaves audio offline."""
    c_timebase, c_ntsc = get_rate_info(f_info["fps"])
    file_elem = ET.SubElement(parent, "file", id=f_info["id"])
    ET.SubElement(file_elem, "name").text = f_info["path"].name
    # URL-encode the path so spaces, '+', and non-ASCII (e.g. "Lukáš") in the folder
    # name don't break the file:// reference — NLEs fail to relink an unencoded pathurl.
    encoded_path = quote(f_info["path"].as_posix().lstrip("/"), safe="/:")
    ET.SubElement(file_elem, "pathurl").text = "file://localhost/" + encoded_path

    f_rate = ET.SubElement(file_elem, "rate")
    ET.SubElement(f_rate, "timebase").text = str(c_timebase)
    ET.SubElement(f_rate, "ntsc").text = c_ntsc
    ET.SubElement(file_elem, "duration").text = str(f_info["duration"])

    tc = ET.SubElement(file_elem, "timecode")
    tc_r = ET.SubElement(tc, "rate")
    ET.SubElement(tc_r, "timebase").text = str(c_timebase)
    ET.SubElement(tc_r, "ntsc").text = c_ntsc
    ET.SubElement(tc, "string").text = "00:00:00:00"
    ET.SubElement(tc, "frame").text = "0"
    ET.SubElement(tc, "displayformat").text = "NDF"

    f_media = ET.SubElement(file_elem, "media")
    f_vid = ET.SubElement(f_media, "video")
    f_v_sample = ET.SubElement(f_vid, "samplecharacteristics")
    f_v_rate = ET.SubElement(f_v_sample, "rate")
    ET.SubElement(f_v_rate, "timebase").text = str(c_timebase)
    ET.SubElement(f_v_rate, "ntsc").text = c_ntsc
    ET.SubElement(f_v_sample, "width").text = str(f_info["width"])
    ET.SubElement(f_v_sample, "height").text = str(f_info["height"])
    f_aud = ET.SubElement(f_media, "audio")
    ET.SubElement(f_aud, "channelcount").text = "2"
    return file_elem


def resolve_media_sources(edl: dict, edl_path: Path) -> dict[str, Path]:
    """Resolve and validate the media path for every named EDL source."""
    edit_dir = edl_path.parent
    sources = edl.get("sources", {})
    overrides = edl.get("source_overrides", {})

    unknown_overrides = set(overrides) - set(sources)
    if unknown_overrides:
        names = ", ".join(sorted(unknown_overrides))
        raise ValueError(f"Source override refers to unknown source(s): {names}")

    resolved = {}
    for src_name, src_path_str in sources.items():
        src_path = Path(src_path_str)
        if not src_path.is_absolute():
            src_path = edit_dir / src_path
        src_path = src_path.resolve()

        if src_name in overrides:
            selected_path = Path(overrides[src_name])
            if not selected_path.is_absolute():
                selected_path = edit_dir / selected_path
            selected_path = selected_path.resolve()
        else:
            legacy_debug_path = edit_dir / f"{src_path.stem}_debug.mp4"
            if legacy_debug_path.exists():
                print(
                    f"Notice: Using debug video {legacy_debug_path.name} "
                    "instead of original source."
                )
                selected_path = legacy_debug_path.resolve()
            else:
                selected_path = src_path

        if not selected_path.exists():
            raise FileNotFoundError(
                f"Media source '{src_name}' not found: {selected_path}"
            )
        resolved[src_name] = selected_path

    return resolved


def build_fcp7_xml(edl: dict, edl_path: Path) -> str:
    sources = resolve_media_sources(edl, edl_path)
    
    # We will assume all videos share the format of the first video for the sequence settings
    # or fallback to 1080p25
    seq_fps = 25.0
    seq_width = 1920
    seq_height = 1080
    
    file_map = {}
    f_idx = 1
    
    for src_name, src_path in sources.items():
        fps, width, height, duration_frames = get_video_info(src_path)
        
        # Use first valid video's properties for sequence
        if f_idx == 1:
            seq_fps = fps
            seq_width = width
            seq_height = height
            
        file_map[src_name] = {
            "id": f"file-{f_idx}",
            "path": src_path,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration_frames
        }
        f_idx += 1
        
    seq_timebase, seq_ntsc = get_rate_info(seq_fps)
    
    xmeml = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(xmeml, "sequence", id="Agent Edit")
    ET.SubElement(sequence, "name").text = "Agent Edit"
    
    seq_duration = sum(frames(float(r["end"]) - float(r["start"]), seq_fps) for r in edl.get("ranges", []))
    ET.SubElement(sequence, "duration").text = str(seq_duration)
    
    rate_elem = ET.SubElement(sequence, "rate")
    ET.SubElement(rate_elem, "timebase").text = str(seq_timebase)
    ET.SubElement(rate_elem, "ntsc").text = seq_ntsc
    
    timecode_elem = ET.SubElement(sequence, "timecode")
    tc_rate = ET.SubElement(timecode_elem, "rate")
    ET.SubElement(tc_rate, "timebase").text = str(seq_timebase)
    ET.SubElement(tc_rate, "ntsc").text = seq_ntsc
    ET.SubElement(timecode_elem, "string").text = "01:00:00:00"
    ET.SubElement(timecode_elem, "frame").text = str(seq_timebase * 60 * 60)
    
    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")
    
    v_format = ET.SubElement(video, "format")
    v_sample = ET.SubElement(v_format, "samplecharacteristics")
    v_rate = ET.SubElement(v_sample, "rate")
    ET.SubElement(v_rate, "timebase").text = str(seq_timebase)
    ET.SubElement(v_rate, "ntsc").text = seq_ntsc
    ET.SubElement(v_sample, "width").text = str(seq_width)
    ET.SubElement(v_sample, "height").text = str(seq_height)
    
    v_track = ET.SubElement(video, "track")
    
    audio = ET.SubElement(media, "audio")
    a_format = ET.SubElement(audio, "format")
    a_sample = ET.SubElement(a_format, "samplecharacteristics")
    ET.SubElement(a_sample, "depth").text = "16"
    ET.SubElement(a_sample, "samplerate").text = "48000"
    
    a_track1 = ET.SubElement(audio, "track")
    a_track2 = ET.SubElement(audio, "track")
    a_track3 = ET.SubElement(audio, "track")
    a_track4 = ET.SubElement(audio, "track")
    
    current_out_frame = 0
    clip_idx = 1
    for r in edl.get("ranges", []):
        src_name = r["source"]
        f_info = file_map.get(src_name)
        if not f_info:
            continue
            
        c_fps = f_info["fps"]
        c_timebase, c_ntsc = get_rate_info(c_fps)
        
        start_sec = float(r["start"])
        end_sec = float(r["end"])
        
        audio_in_offset = float(r.get("audio_in_offset", 0.0))
        audio_out_offset = float(r.get("audio_out_offset", 0.0))
        
        # Video timings
        dur_frames = frames(end_sec - start_sec, seq_fps)
        in_frame = frames(start_sec, c_fps)
        out_frame = in_frame + frames(end_sec - start_sec, c_fps)
        
        start_frame = current_out_frame
        end_frame = current_out_frame + dur_frames
        
        # Audio timings
        a_start_sec = start_sec + audio_in_offset
        a_end_sec = end_sec + audio_out_offset
        
        a_in_frame = frames(a_start_sec, c_fps)
        a_out_frame = a_in_frame + frames(a_end_sec - a_start_sec, c_fps)
        
        a_start_frame = current_out_frame + frames(audio_in_offset, seq_fps)
        a_end_frame = a_start_frame + frames(a_end_sec - a_start_sec, seq_fps)
        
        # Checkerboard tracks
        if clip_idx % 2 == 1:
            target_track_1 = a_track1
            target_track_2 = a_track2
            t_idx = 1
        else:
            target_track_1 = a_track3
            target_track_2 = a_track4
            t_idx = 3
        
        # --- VIDEO CLIPITEM ---
        v_clip = ET.SubElement(v_track, "clipitem", id=f"clip-v-{clip_idx}")
        ET.SubElement(v_clip, "name").text = src_name
        ET.SubElement(v_clip, "duration").text = str(dur_frames)
        
        c_rate = ET.SubElement(v_clip, "rate")
        ET.SubElement(c_rate, "timebase").text = str(c_timebase)
        ET.SubElement(c_rate, "ntsc").text = c_ntsc
        
        ET.SubElement(v_clip, "start").text = str(start_frame)
        ET.SubElement(v_clip, "end").text = str(end_frame)
        ET.SubElement(v_clip, "in").text = str(in_frame)
        ET.SubElement(v_clip, "out").text = str(out_frame)
        
        add_source_file(v_clip, f_info)

        # --- AUDIO CLIPITEM 1 ---
        a_clip1 = ET.SubElement(target_track_1, "clipitem", id=f"clip-a{t_idx}-{clip_idx}")
        ET.SubElement(a_clip1, "name").text = src_name
        a_rate1 = ET.SubElement(a_clip1, "rate")
        ET.SubElement(a_rate1, "timebase").text = str(seq_timebase)
        ET.SubElement(a_rate1, "ntsc").text = seq_ntsc
        
        ET.SubElement(a_clip1, "start").text = str(a_start_frame)
        ET.SubElement(a_clip1, "end").text = str(a_end_frame)
        ET.SubElement(a_clip1, "in").text = str(a_in_frame)
        ET.SubElement(a_clip1, "out").text = str(a_out_frame)
        add_source_file(a_clip1, f_info)

        sourcetrack1 = ET.SubElement(a_clip1, "sourcetrack")
        ET.SubElement(sourcetrack1, "mediatype").text = "audio"
        ET.SubElement(sourcetrack1, "trackindex").text = "1"
        
        # --- AUDIO CLIPITEM 2 ---
        a_clip2 = ET.SubElement(target_track_2, "clipitem", id=f"clip-a{t_idx+1}-{clip_idx}")
        ET.SubElement(a_clip2, "name").text = src_name
        a_rate2 = ET.SubElement(a_clip2, "rate")
        ET.SubElement(a_rate2, "timebase").text = str(seq_timebase)
        ET.SubElement(a_rate2, "ntsc").text = seq_ntsc
        
        ET.SubElement(a_clip2, "start").text = str(a_start_frame)
        ET.SubElement(a_clip2, "end").text = str(a_end_frame)
        ET.SubElement(a_clip2, "in").text = str(a_in_frame)
        ET.SubElement(a_clip2, "out").text = str(a_out_frame)
        add_source_file(a_clip2, f_info)

        sourcetrack2 = ET.SubElement(a_clip2, "sourcetrack")
        ET.SubElement(sourcetrack2, "mediatype").text = "audio"
        ET.SubElement(sourcetrack2, "trackindex").text = "2"
        
        current_out_frame += dur_frames
        clip_idx += 1

    rough_string = ET.tostring(xmeml, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    
    doctype = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n'
    return doctype + reparsed.toprettyxml(indent="  ").split("?>", 1)[-1].strip()

def main():
    ap = argparse.ArgumentParser(description="Export edl.json to FCP 7 XML")
    ap.add_argument("edl", type=Path, help="Path to edl.json")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output XML path")
    args = ap.parse_args()

    edl_path = args.edl.resolve()
    if not edl_path.exists():
        sys.exit(f"EDL not found: {edl_path}")

    with open(edl_path, "r", encoding="utf-8") as f:
        edl = json.load(f)

    xml_content = build_fcp7_xml(edl, edl_path)
    
    out_path = (
        args.output.resolve()
        if args.output is not None
        else edl_path.parent / "timeline.xml"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_content)
        
    print(f"Exported timeline to {out_path.name}")

if __name__ == "__main__":
    main()
