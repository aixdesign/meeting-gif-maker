# Meeting GIF Maker

Drop in a recording of an online call → it picks the nicest, most varied
stills (sharp, faces visible, layout changes, spread across the call) → builds
a looping GIF you can fine-tune and download.

**Everything runs on your own machine — recordings are never uploaded anywhere.**

---

## Quick start (for teammates)

1. **Get the folder** (download the zip, or `git clone …`) and put it anywhere,
   e.g. your home folder.
2. **Double-click `Start GIF Maker.command`.**
   - *First time only:* macOS may warn about an app from the internet —
     **right-click the file → Open → Open**. After that, plain double-click works.
   - *First run* takes a minute or two to set itself up (needs internet once).
3. Your browser opens the tool automatically. Keep the Terminal window open
   while you use it; close it when you're done.

That's it — everything needed (including ffmpeg for fast video decoding) is
installed automatically into the tool's own folder on first run. Nothing is
installed system-wide.

---

## Using it

1. **Drop in a video** — MP4 / MOV / WEBM / MKV. Big files are fine.
   Drop several at once to pool them into a single GIF.
2. Set **Number of stills**, **Pace** (Slow / Medium / Fast) and **Style**:
   - **Slideshow** — each still holds on screen, then the next.
   - **Motion** — a short live clip at each moment, stitched together.
3. Hit **Make GIF**. It scans the whole recording, auto-picks the stills,
   and shows you the looping GIF.
4. Not quite right? Open **▸ Edit chosen frames**, click frames to add or
   remove them, then **Rebuild GIF**.
5. **Download GIF.** Done. (GIFs are also saved in the `output/` folder.)

### How it picks stills

Every ~second of the recording is scored on:
- **Sharpness** — blurry frames are skipped
- **Faces** — frames with people visible score higher
- **Scene change** — screenshares starting, speakers switching, people joining
- **Variety** — picks are chosen to look different from each other and to
  spread across the whole call (50/50 balance of quality vs. variety)

---

## For the technically curious

- Python 3 + Flask backend, vanilla-JS frontend, OpenCV for scoring,
  ffmpeg for fast frame extraction (pip-bundled via `imageio-ffmpeg`; a
  system/Homebrew ffmpeg is used instead when present, with a pure-OpenCV
  fallback if neither exists).
- Start from a terminal instead: `./run.sh`
- The web app runs at http://127.0.0.1:5001 — local only, nothing exposed.
- `uploads/` holds per-job frame caches (safe to delete anytime),
  `output/` holds your built GIFs.
