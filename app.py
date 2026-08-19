"""
Meeting GIF Maker — local web app.

Drag in a recording of an online call, the tool samples candidate stills,
scores them on sharpness / scene-change / faces / time-spread, auto-picks a
nice set, and stitches them into a GIF. You can refine the picks and tweak
GIF params (frame count, hold time, width, fps, slideshow vs motion) per run.

Run:  ./run.sh   (or  venv/bin/python app.py)
Then open http://127.0.0.1:5001
"""

import base64
import glob
import io
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# ffmpeg gives a single fast sequential decode; OpenCV msec-seeks re-decode
# from the start on VP9/webm, which made long videos quadratic-slow.
# Resolution order: project-local bin/ (auto-fetched by run.sh on first run),
# then PATH, then the Homebrew location.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_tool(name):
    for cand in (os.path.join(_HERE, "bin", name),
                 shutil.which(name),
                 "/opt/homebrew/bin/" + name):
        if cand and os.path.exists(cand):
            return cand
    if name == "ffmpeg":
        try:  # bundled with the pip deps — guarantees ffmpeg on any machine
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return name  # last resort: rely on PATH at call time


FFMPEG = _find_tool("ffmpeg")
FFPROBE = _find_tool("ffprobe")
HAS_FFMPEG = os.path.isabs(FFMPEG) and os.path.exists(FFMPEG)
HAS_FFPROBE = os.path.isabs(FFPROBE) and os.path.exists(FFPROBE)

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
OUTPUT_DIR = os.path.join(BASE, "output")
STATIC_DIR = os.path.join(BASE, "static")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 * 1024  # 16 GB cap


@app.after_request
def add_cors(resp):
    # lets the UI work even when opened as a file:// page (preview panel)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    return ("", 204)

# job_id -> dict(state, pct, message, result, frames[], video_path)
JOBS = {}
JOBS_LOCK = threading.Lock()

# Frames are cached on disk at this max width so rebuilds are instant and
# the GIF can be sharper than the preview thumbnails.
CACHE_WIDTH = 1280
THUMB_WIDTH = 200
MAX_SAMPLES = 700          # cap on how many frames we score (bounds runtime)
MIN_SAMPLE_INTERVAL = 0.8  # seconds — don't sample finer than this

# CascadeClassifier is not thread-safe — give each scoring thread its own.
# The XML is vendored in data/ because some opencv wheels don't ship it.
_CASCADE_PATHS = [
    os.path.join(BASE, "data", "haarcascade_frontalface_default.xml"),
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    cv2.data.haarcascades + "haarcascade_frontalface_alt_tree.xml",
]
_TLS = threading.local()


def _face_cascade():
    if not hasattr(_TLS, "cascade"):
        _TLS.cascade = None
        for p in _CASCADE_PATHS:
            if os.path.exists(p):
                c = cv2.CascadeClassifier(p)
                if not c.empty():
                    _TLS.cascade = c
                    break
    return _TLS.cascade


def set_job(job_id, **kw):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(kw)


def get_job(job_id):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        return dict(j) if j else None


# ---------------------------------------------------------------------------
# Frame scoring helpers
# ---------------------------------------------------------------------------

def sharpness_score(gray):
    """Variance of the Laplacian — low for blurry / out-of-focus frames."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def hist_signature(bgr):
    """Normalised HSV histogram, used to measure layout/scene change."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def face_score(gray):
    """Rough 'people are visible' signal: count + relative size of faces.
    Degrades to 0 (other signals still work) if no cascade is available."""
    cascade = _face_cascade()
    if cascade is None:
        return 0.0, 0
    h, w = gray.shape[:2]
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=5,
        minSize=(max(24, w // 20), max(24, h // 20)),
    )
    if len(faces) == 0:
        return 0.0, 0
    area = sum(fw * fh for (_, _, fw, fh) in faces) / float(w * h)
    # presence (capped face count) + a bonus for faces filling more frame
    return min(len(faces), 4) / 4.0 * 0.7 + min(area * 4, 1.0) * 0.3, len(faces)


def normalize(values):
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo, hi = np.percentile(arr, 5), np.percentile(arr, 95)
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


# ---------------------------------------------------------------------------
# Analysis job — sample, score, cache frames, auto-select
# ---------------------------------------------------------------------------

def ffprobe_duration(path):
    if HAS_FFPROBE:
        try:
            out = subprocess.check_output(
                [FFPROBE, "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", path], timeout=30)
            return float(out.strip())
        except Exception:
            pass
    # no ffprobe (e.g. pip-bundled ffmpeg only): ffmpeg prints the duration
    # on stderr when probing the input
    try:
        p = subprocess.run([FFMPEG, "-hide_banner", "-i", path],
                           capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
        if m:
            h, mi, s = m.groups()
            return int(h) * 3600 + int(mi) * 60 + float(s)
    except Exception:
        pass
    return 0.0


def spatial_signature(gray):
    """Tiny grayscale layout map — two frames with the same colours but a
    different layout (speaker switch, screenshare) differ strongly here."""
    return (cv2.resize(gray, (32, 18)).astype(np.float32) / 255.0).ravel()


def score_frame_file(path):
    """Score one cached frame JPEG. Pure function → safe to run in parallel."""
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (320, max(1, int(h * 320.0 / w))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return (sharpness_score(gray), *face_score(gray),
            hist_signature(small), spatial_signature(gray),
            make_thumb_b64(bgr))


def append_scored(samples, src_name, video_path, ordered):
    """Sequential pass: scene-change needs frames in time order."""
    prev_hist = None
    for file_t, path, scored in ordered:
        if scored is None:
            continue
        sharp, fscore, fcount, hist, sig, thumb = scored
        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            scene = float(np.clip(1.0 - corr, 0, 1))
        else:
            scene = 1.0
        prev_hist = hist
        samples.append(dict(idx=len(samples), file_t=float(file_t),
                            src=src_name, src_path=video_path, sharp=sharp,
                            face=fscore, faces=int(fcount), scene=scene,
                            path=path, thumb=thumb, hist=hist, sig=sig))


def sample_one_video(video_path, src_name, time_offset, samples, frames_dir,
                     progress_cb):
    """Extract candidate frames with one fast sequential ffmpeg decode, then
    score them in parallel across CPU cores. Falls back to sequential OpenCV
    decoding if ffmpeg is unavailable. Returns the video's duration."""
    duration = ffprobe_duration(video_path) if HAS_FFMPEG else 0.0
    if duration <= 0:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        nf = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        duration = (nf / fps) if (fps and nf) else 0.0
    interval = max(MIN_SAMPLE_INTERVAL,
                   (duration / MAX_SAMPLES) if duration > 0 else MIN_SAMPLE_INTERVAL)

    extracted = []
    if HAS_FFMPEG:
        prefix = uuid.uuid4().hex[:6]
        pattern = os.path.join(frames_dir, f"{prefix}_%05d.jpg")
        cmd = [FFMPEG, "-hide_banner", "-nostats", "-loglevel", "error",
               "-i", video_path,
               "-vf", f"fps=1/{interval:.4f},scale='min({CACHE_WIDTH},iw)':-2",
               "-q:v", "3", "-progress", "pipe:1", "-y", pattern]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True)
            for line in proc.stdout:  # extraction ≈ 70% of the work
                if line.startswith(("out_time_us=", "out_time_ms=")):
                    try:
                        sec = int(line.split("=", 1)[1]) / 1e6
                    except ValueError:
                        continue
                    if duration > 0:
                        progress_cb(0.7 * min(1.0, sec / duration))
            proc.wait()
            if proc.returncode == 0:
                extracted = sorted(
                    glob.glob(os.path.join(frames_dir, f"{prefix}_*.jpg")))
        except OSError:
            extracted = []

    if not extracted:
        return sample_one_video_cv(video_path, src_name, samples, frames_dir,
                                   progress_cb, interval)

    # parallel scoring (cv2 releases the GIL for the heavy parts)
    results = [None] * len(extracted)
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futs = {ex.submit(score_frame_file, p): i
                for i, p in enumerate(extracted)}
        done_n = 0
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
            done_n += 1
            if done_n % 10 == 0:
                progress_cb(0.7 + 0.3 * done_n / len(extracted))

    ordered = [(i * interval, path, res)
               for i, (path, res) in enumerate(zip(extracted, results))]
    append_scored(samples, src_name, video_path, ordered)
    return duration if duration > 0 else len(extracted) * interval


def sample_one_video_cv(video_path, src_name, samples, frames_dir,
                        progress_cb, interval):
    """No-ffmpeg fallback: ONE sequential decode (grab + retrieve every Nth
    frame) — never random seeks, which are pathologically slow on VP9."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open '{src_name}'. Try MP4/MOV/WEBM/MKV, or re-export it.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nf = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    step = max(1, int(round(fps * interval)))
    ordered, i = [], 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, bgr = cap.retrieve()
            if ok and bgr is not None:
                h, w = bgr.shape[:2]
                if w > CACHE_WIDTH:
                    bgr = cv2.resize(bgr, (CACHE_WIDTH, int(h * CACHE_WIDTH / w)))
                path = os.path.join(frames_dir, f"cv_{len(ordered):05d}.jpg")
                cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                ordered.append((i / fps, path, score_frame_file(path)))
            if nf:
                progress_cb(min(0.99, i / nf))
            if len(ordered) >= MAX_SAMPLES:
                break
        i += 1
    cap.release()
    append_scored(samples, src_name, video_path, ordered)
    return (i / fps) if fps else 0.0


def analyze_job(job_id, video_paths, want_count):
    try:
        set_job(job_id, state="running", pct=2, message="Opening video(s)…")
        frames_dir = os.path.join(UPLOAD_DIR, job_id, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        samples = []
        offset = 0.0
        n_videos = len(video_paths)
        for vi, (vpath, vname) in enumerate(video_paths):
            def progress_cb(frac, vi=vi, vname=vname):
                base = 3 + int(85 * (vi + frac) / n_videos)
                tag = f" ({vi + 1}/{n_videos})" if n_videos > 1 else ""
                set_job(job_id, pct=base,
                        message=f"Scoring frames{tag}: {vname}")
            before = len(samples)
            dur = sample_one_video(vpath, vname, offset, samples, frames_dir,
                                   progress_cb)
            for s in samples[before:]:
                s["t"] = offset + s["file_t"]
            offset += dur if dur > 0 else (
                (samples[-1]["file_t"] if samples else 0) + MIN_SAMPLE_INTERVAL)

        if not samples:
            raise RuntimeError("No frames could be read from the video(s).")

        duration = offset
        set_job(job_id, pct=92, message="Selecting the nicest stills…")
        rank_samples(samples)
        selected = auto_select(samples, want_count, duration)

        for s in samples:
            s["selected"] = s["idx"] in selected

        # candidates to show: selected set + best alternates so you can swap
        show_n = min(len(samples), max(want_count * 3, want_count + 8))
        by_rank = sorted(samples, key=lambda s: s["composite"], reverse=True)
        shown_idx = set(selected) | {s["idx"] for s in by_rank[:show_n]}
        candidates = sorted([s for s in samples if s["idx"] in shown_idx],
                            key=lambda s: s["t"])

        result = dict(
            duration=duration,
            total_samples=len(samples),
            n_videos=n_videos,
            candidates=[{k: s[k] for k in
                         ("idx", "t", "file_t", "src", "sharp", "face",
                          "faces", "scene", "composite", "selected", "thumb")}
                        for s in candidates],
        )
        # per-sample info needed for motion mode (src video + in-file time)
        samp_meta = {s["idx"]: {"src_path": s["src_path"], "file_t": s["file_t"]}
                     for s in samples}
        set_job(job_id, state="done", pct=100, message="Ready",
                result=result, frames={s["idx"]: s["path"] for s in samples},
                samp_meta=samp_meta, duration=duration)
    except Exception as e:  # noqa
        traceback.print_exc()
        set_job(job_id, state="error", pct=0, message=str(e))


def make_thumb_b64(bgr):
    h, w = bgr.shape[:2]
    tw = THUMB_WIDTH
    th = int(h * tw / w)
    small = cv2.resize(bgr, (tw, th))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def rank_samples(samples):
    sharp_n = normalize([s["sharp"] for s in samples])
    face_n = normalize([s["face"] for s in samples])
    scene_n = normalize([s["scene"] for s in samples])
    for s, sh, fa, sc in zip(samples, sharp_n, face_n, scene_n):
        # weighted blend of the four signals the user asked for
        s["composite"] = 0.40 * sh + 0.30 * fa + 0.30 * sc
        s["sharp_n"], s["face_n"], s["scene_n"] = float(sh), float(fa), float(sc)


def frame_distance(a, b):
    """Visual difference between two frames (0 = identical, 1 = very different).
    Blends global colour (histogram) with spatial layout (tiny grayscale map),
    so a speaker switch or screenshare counts as different even when the
    overall colours barely change — that's what calls mostly look like."""
    corr = cv2.compareHist(a["hist"], b["hist"], cv2.HISTCMP_CORREL)
    color_d = float(np.clip(1.0 - corr, 0, 1))
    layout_d = float(np.clip(np.abs(a["sig"] - b["sig"]).mean() * 5.0, 0, 1))
    return 0.4 * color_d + 0.6 * layout_d


def auto_select(samples, want, duration):
    """Two stages, as requested:
    1. keep only *good* frames (drop the blurriest),
    2. greedily pick `want` of them optimised for VARIETY — each new pick
       maximises quality + how visually different it is from everything
       already chosen (plus a nudge toward spreading across the timeline).
    This avoids a GIF full of near-identical frames that looks like one image.
    """
    want = max(1, min(want, len(samples)))
    if want >= len(samples):
        return [s["idx"] for s in samples]
    if duration <= 0:
        duration = samples[-1]["t"] or 1.0

    # stage 1 — quality gate: drop the blurriest fifth, keep a WIDE pool so
    # variety has real options (a narrow top-quality pool is self-similar)
    sharp_vals = np.array([s["sharp"] for s in samples])
    blur_floor = np.percentile(sharp_vals, 20)
    good = [s for s in samples if s["sharp"] >= blur_floor] or list(samples)
    pool = sorted(good, key=lambda s: s["composite"], reverse=True)
    pool = pool[:max(want * 15, 150)]

    # stage 2 — diversity-first greedy (max-marginal-relevance style):
    # each pick maximises how DIFFERENT it is from everything chosen so far,
    # with quality as the tie-breaker rather than the driver.
    LAMBDA = 0.50          # quality weight (diversity gets the rest)
    min_gap = duration / (want * 4.0)   # soft no-clumping window
    start = max(pool, key=lambda s: s["composite"])
    chosen = [start]
    chosen_idx = {start["idx"]}
    while len(chosen) < want and len(chosen) < len(pool):
        best, best_score = None, -1.0
        for s in pool:
            if s["idx"] in chosen_idx:
                continue
            visual_div = min(frame_distance(s, c) for c in chosen)
            tgap = min(abs(s["t"] - c["t"]) for c in chosen)
            time_div = min(tgap / max(min_gap * 2, 1e-6), 1.0)
            score = (LAMBDA * s["composite"]
                     + (1 - LAMBDA) * (0.65 * visual_div + 0.35 * time_div))
            if tgap < min_gap:          # discourage clumping hard
                score *= 0.35
            if score > best_score:
                best, best_score = s, score
        if best is None:
            break
        chosen.append(best)
        chosen_idx.add(best["idx"])
    return list(chosen_idx)


# ---------------------------------------------------------------------------
# GIF building
# ---------------------------------------------------------------------------

def pil_from_path(path, width):
    img = Image.open(path).convert("RGB")
    if width and img.width != width:
        h = int(img.height * width / img.width)
        img = img.resize((width, h), Image.LANCZOS)
    return img


def quantize(img):
    return img.convert("RGB").quantize(
        colors=256, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)


def build_slideshow(paths, width, hold_ms):
    frames = [quantize(pil_from_path(p, width)) for p in paths]
    # normalise all frames to the first frame's size
    w0, h0 = frames[0].size
    frames = [f if f.size == (w0, h0) else f.resize((w0, h0)) for f in frames]
    return frames, hold_ms


def motion_clip_cv(vpath, t, width, fps, clip_secs):
    """OpenCV fallback for one clip (only used when ffmpeg is missing)."""
    out = []
    cap = cv2.VideoCapture(vpath)
    n = max(1, int(clip_secs * fps))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    keep = max(1, int(round(src_fps / fps)))
    grabbed = 0
    while len(out) < n:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        if grabbed % keep == 0:
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if img.width != width:
                img = img.resize((width, int(img.height * width / img.width)),
                                 Image.LANCZOS)
            out.append(quantize(img))
        grabbed += 1
    cap.release()
    return out


def build_motion(shots, width, fps, clip_secs):
    """shots: list of (video_path, in-file timestamp). Extracts a short clip
    at each via ffmpeg fast-seek (-ss before -i), stitches them in order."""
    frames = []
    n = max(1, int(clip_secs * fps))
    tmpdir = tempfile.mkdtemp(prefix="gifmotion_")
    try:
        for i, (vpath, t) in enumerate(shots):
            shot_files = []
            if HAS_FFMPEG:
                pattern = os.path.join(tmpdir, f"s{i:03d}_%03d.jpg")
                cmd = [FFMPEG, "-hide_banner", "-loglevel", "error",
                       "-ss", f"{max(0.0, t):.3f}", "-t", f"{clip_secs:.3f}",
                       "-i", vpath, "-vf", f"fps={fps},scale={width}:-2",
                       "-frames:v", str(n), "-q:v", "3", "-y", pattern]
                try:
                    subprocess.run(cmd, check=True, capture_output=True,
                                   timeout=120)
                    shot_files = sorted(
                        glob.glob(os.path.join(tmpdir, f"s{i:03d}_*.jpg")))
                except (subprocess.SubprocessError, OSError):
                    shot_files = []
            if shot_files:
                for p in shot_files:
                    frames.append(quantize(Image.open(p).convert("RGB")))
            else:
                frames.extend(motion_clip_cv(vpath, t, width, fps, clip_secs))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    if not frames:
        raise RuntimeError("Could not read motion frames from the video.")
    w0, h0 = frames[0].size
    frames = [f if f.size == (w0, h0) else f.resize((w0, h0)) for f in frames]
    return frames, int(1000.0 / fps)


def save_gif(frames, duration_ms, out_path):
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, disposal=2, optimize=True,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    files = request.files.getlist("video")
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify(error="No video uploaded"), 400
    want = int(request.form.get("count", 12))
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    video_paths = []  # (path, display_name)
    for i, f in enumerate(files):
        ext = os.path.splitext(f.filename or "video.mp4")[1] or ".mp4"
        path = os.path.join(job_dir, f"source_{i:02d}{ext}")
        f.save(path)
        video_paths.append((path, os.path.basename(f.filename)))

    set_job(job_id, state="queued", pct=0, message="Queued…")
    threading.Thread(target=analyze_job, args=(job_id, video_paths, want),
                    daemon=True).start()
    return jsonify(job_id=job_id)


@app.route("/api/progress/<job_id>")
def api_progress(job_id):
    j = get_job(job_id)
    if not j:
        return jsonify(error="unknown job"), 404
    out = {k: j.get(k) for k in ("state", "pct", "message")}
    if j.get("state") == "done":
        out["result"] = j.get("result")
    return jsonify(out)


@app.route("/api/build", methods=["POST"])
def api_build():
    data = request.get_json(force=True)
    job_id = data["job_id"]
    j = get_job(job_id)
    if not j or j.get("state") != "done":
        return jsonify(error="Analyze the video first."), 400

    idxs = data.get("selected", [])
    if not idxs:
        return jsonify(error="Pick at least one still."), 400

    width = int(data.get("width", 600))
    mode = data.get("mode", "slideshow")
    frames_map = j["frames"]

    # selected frames in timeline order
    items = sorted(idxs, key=lambda i: i)  # idx order == timeline order
    paths = [frames_map[i] for i in items if i in frames_map]
    if not paths:
        return jsonify(error="Selected stills not found — re-analyze."), 400

    out_name = f"{job_id}_{int(time.time())}_{uuid.uuid4().hex[:4]}.gif"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    try:
        if mode == "motion":
            fps = float(data.get("fps", 8))
            clip = float(data.get("clip_secs", 1.0))
            # (source video, in-file timestamp) per selected still
            meta = j["samp_meta"]
            shots = [(meta[i]["src_path"], meta[i]["file_t"])
                     for i in items if i in meta]
            frames, dur = build_motion(shots, width, fps, clip)
        else:
            hold_ms = int(float(data.get("hold", 0.5)) * 1000)
            frames, dur = build_slideshow(paths, width, hold_ms)
        save_gif(frames, dur, out_path)
    except Exception as e:  # noqa
        traceback.print_exc()
        return jsonify(error=str(e)), 500

    size = os.path.getsize(out_path)
    return jsonify(url=f"/output/{out_name}", bytes=size,
                   frames=len(frames))


@app.route("/output/<path:name>")
def output_file(name):
    return send_from_directory(OUTPUT_DIR, name)


if __name__ == "__main__":
    print("\n  Meeting GIF Maker → http://127.0.0.1:5001\n")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
