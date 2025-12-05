# scripts/preprocess_ubfc.py
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import cv2

RAW = Path("data/raw/ubfc_rppg")                # input: subject folders
OUT = Path("data/processed/ubfc_rppg"); OUT.mkdir(parents=True, exist_ok=True)
RAW_CSV_OUT = Path("data/raw/ohiot1dm"); RAW_CSV_OUT.mkdir(parents=True, exist_ok=True)
# if you prefer a different CSV target path, change RAW_CSV_OUT above

def read_ground_truth(gt_path: Path) -> np.ndarray:
    """Try several strategies to read ground_truth into a 1D numpy float array."""
    txt = gt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not txt:
        return np.array([], dtype=float)

    # Try pandas auto-detect with several separators
    for sep in [None, r"\s+", ",", ";"]:
        try:
            if sep is None:
                df = pd.read_csv(gt_path, engine="python")
            else:
                df = pd.read_csv(gt_path, sep=sep, engine="python", header=None)
            # pick the first numeric column
            for col in df.columns:
                colvals = pd.to_numeric(df[col], errors="coerce")
                if colvals.notna().sum() > 0:
                    return colvals.dropna().to_numpy(dtype=float)
        except Exception:
            continue

    # fallback: regexp find numbers
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?", txt)
    return np.array([float(x) for x in nums], dtype=float)

def align_gt_to_frames(gt_values: np.ndarray, fps: float, n_frames: int, duration: float) -> np.ndarray:
    """
    Interpolate/align ground-truth samples to per-frame values.
    Returns array length n_frames (floats).
    """
    if gt_values.size == 0:
        return np.full(n_frames, np.nan)

    if gt_values.size == n_frames:
        return gt_values.astype(float)

    frame_times = np.arange(n_frames) / float(fps if fps > 0 else 30.0)

    # common case: one sample per second (length ~ duration seconds)
    if gt_values.size == int(round(duration)) and gt_values.size > 1:
        gt_times = np.arange(gt_values.size)
    else:
        # assume samples are evenly spaced across the duration
        gt_times = np.linspace(0.0, duration if duration > 0 else 1.0, num=gt_values.size, endpoint=False)

    if gt_values.size == 1:
        return np.full(n_frames, float(gt_values[0]))

    # use numpy.interp; extrapolate by edges
    interp = np.interp(frame_times, gt_times, gt_values, left=gt_values[0], right=gt_values[-1])
    return interp

def process_subject_dir(subject_dir: Path):
    """
    Expect subject_dir to contain a video (vid.avi / vid.mp4) and ground_truth.txt (or .csv).
    Produces metadata json and per-frame CSV.
    """
    # find video
    video = None
    for ext in ("vid.avi", "vid.mp4", "*.avi", "*.mp4"):
        # first try exact names 'vid.avi' or 'vid.mp4'
        if ext in ("vid.avi", "vid.mp4"):
            cand = subject_dir / ext
            if cand.exists():
                video = cand
                break
        else:
            lst = list(subject_dir.glob(ext))
            if lst:
                video = lst[0]
                break

    # find ground truth
    gt = None
    for name in ("ground_truth.txt", "ground_truth.csv", "*.txt", "*.csv"):
        if name in ("ground_truth.txt", "ground_truth.csv"):
            cand = subject_dir / name
            if cand.exists():
                gt = cand
                break
        else:
            lst = list(subject_dir.glob(name))
            if lst:
                # prefer files that contain 'ground' or 'ppg' in their name
                chosen = None
                for f in lst:
                    if "ground" in f.stem.lower() or "ppg" in f.stem.lower():
                        chosen = f
                        break
                gt = chosen or lst[0]
                break

    if video is None:
        print(f"[skip] {subject_dir.name}: no video found")
        return

    # read video meta
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[error] cannot open video {video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    # if frame count is zero, count frames manually (slower)
    if frames == 0:
        frames = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frames += 1
        cap.release()
        cap = cv2.VideoCapture(str(video))

    duration = frames / fps if fps > 0 else 0.0

    meta = {
        "subject": subject_dir.name,
        "video": str(video).replace("\\", "/"),
        "fps": float(fps),
        "frames": int(frames),
        "width": int(width),
        "height": int(height),
        "duration_sec": float(duration)
    }

    gt_values = read_ground_truth(gt) if gt is not None else np.array([], dtype=float)
    if gt is not None:
        meta["ppg_file"] = str(gt).replace("\\", "/")
        meta["ppg_samples"] = int(gt_values.size)

    # align to frames
    aligned = align_gt_to_frames(gt_values, fps, frames, duration)

    # build dataframe
    rows = []
    for i in range(frames):
        time_sec = i / float(fps) if fps > 0 else 0.0
        val = aligned[i]
        rows.append({
            "subject": subject_dir.name,
            "frame": int(i),
            "time": float(time_sec),
            "rppg": float(val) if not np.isnan(val) else "",
            "video_path": str(video).replace("\\", "/")
        })
    df = pd.DataFrame(rows)

    # write meta json
    out_meta = OUT / f"{subject_dir.name}_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2))
    print("Wrote", out_meta)

    # write processed CSV (in processed folder)
    out_csv_proc = OUT / f"{subject_dir.name}.csv"
    df.to_csv(out_csv_proc, index=False)
    print("Wrote", out_csv_proc)

    # also write CSV to RAW_CSV_OUT so other scripts expecting CSVs in data/raw/ohiot1dm find them
    out_csv_raw = RAW_CSV_OUT / f"{subject_dir.name}.csv"
    df.to_csv(out_csv_raw, index=False)
    print("Wrote", out_csv_raw)

    cap.release()
    return df

def main():
    # discover subject dirs: any subdirectory containing a video file is treated as a subject
    subject_dirs = []
    # first prefer immediate child directories
    for p in sorted(RAW.iterdir()):
        if p.is_dir():
            # check for at least one video file inside
            if any(p.glob("*.avi")) or any(p.glob("*.mp4")):
                subject_dirs.append(p)

    # also handle videos directly in RAW root (if any)
    root_videos = list(RAW.glob("*.avi")) + list(RAW.glob("*.mp4"))
    for v in root_videos:
        # create a pseudo-subject folder name from stem
        pseudo = OUT / v.stem
        # treat v's parent as subject path by creating a temporary Path object wrapper - easier to process consistently:
        subject_dirs.append(v.parent)  # will process folder (but may contain many videos)

    if not subject_dirs:
        print("No subject directories with videos found in", RAW)
        return

    # process each subject directory individually
    processed = 0
    for sd in sorted(set(subject_dirs)):
        # for safety, if a dir contains multiple videos, process each video as separate subject entry named dirname__videoname
        vids = sorted(list(Path(sd).glob("*.avi")) + list(Path(sd).glob("*.mp4")))
        if len(vids) <= 1:
            # use folder name as subject
            process_subject_dir(Path(sd))
            processed += 1
        else:
            # multiple videos -> process each separately and create unique subject id
            for v in vids:
                # create a temp folder-like wrapper by naming subject as foldername__videoname
                temp_name = f"{Path(sd).name}__{v.stem}"
                temp_dir = Path(sd)  # re-use same dir but we will pick the particular video inside function by matching names
                # slight hack: move/rename isn't necessary; instead adjust function to prefer exact video file when multiple found:
                # call a variant of process_subject_dir that accepts explicit video path
                def process_single_video(subject_name, video_path):
                    # replicate logic but forcing video_path
                    video = Path(video_path)
                    # try to locate ground truth with same stem or containing 'ground' or 'ppg'
                    gt_candidates = [f for f in video.parent.glob("*.txt")] + [f for f in video.parent.glob("*.csv")]
                    gt = None
                    for f in gt_candidates:
                        if video.stem in f.stem or "ground" in f.stem.lower() or "ppg" in f.stem.lower():
                            gt = f
                            break
                    # fallback to first candidate
                    if gt is None and gt_candidates:
                        gt = gt_candidates[0]

                    cap = cv2.VideoCapture(str(video))
                    if not cap.isOpened():
                        print(f"[error] cannot open video {video}")
                        return
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                    if frames == 0:
                        frames = 0
                        while True:
                            ret, _ = cap.read()
                            if not ret:
                                break
                            frames += 1
                        cap.release()
                        cap = cv2.VideoCapture(str(video))
                    duration = frames / fps if fps > 0 else 0.0
                    meta = {"subject": subject_name, "video": str(video).replace("\\","/"), "fps": float(fps), "frames": int(frames), "width": int(width), "height": int(height), "duration_sec": float(duration)}
                    gt_values = read_ground_truth(gt) if gt is not None else np.array([],dtype=float)
                    if gt is not None:
                        meta["ppg_file"] = str(gt).replace("\\","/")
                        meta["ppg_samples"] = int(gt_values.size)
                    aligned = align_gt_to_frames(gt_values, fps, frames, duration)
                    rows = []
                    for i in range(frames):
                        time_sec = i / float(fps) if fps > 0 else 0.0
                        val = aligned[i]
                        rows.append({"subject": subject_name, "frame": int(i), "time": float(time_sec), "rppg": float(val) if not np.isnan(val) else "", "video_path": str(video).replace("\\","/")})
                    df = pd.DataFrame(rows)
                    out_meta = OUT / f"{subject_name}_meta.json"
                    out_meta.write_text(json.dumps(meta, indent=2))
                    print("Wrote", out_meta)
                    out_csv_proc = OUT / f"{subject_name}.csv"
                    df.to_csv(out_csv_proc, index=False)
                    print("Wrote", out_csv_proc)
                    out_csv_raw = RAW_CSV_OUT / f"{subject_name}.csv"
                    df.to_csv(out_csv_raw, index=False)
                    print("Wrote", out_csv_raw)
                    cap.release()
                    return df

                process_single_video(temp_name, v)
                processed += 1

    print(f"Processed {processed} subject(s). CSVs written to {RAW_CSV_OUT} (also in {OUT}).")

if __name__ == "__main__":
    main()
