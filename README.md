# GallerySweep

Finds duplicate and near-duplicate photos in a folder, shows them side by side,
and moves the extras to the Recycle Bin.

It catches copies that a normal file comparison misses:

- the same photo saved rotated 90/180/270 degrees
- screenshots with black, white or coloured bars added
- WhatsApp forwards, re-compressed JPEGs, resized copies
- lightly cropped copies (Smart mode)

**Videos are not scanned.** Only still images. Video duplicate detection needs
frame sampling and temporal matching, which is a different problem and is not
attempted here — so the MP4s in a WhatsApp folder are simply ignored.

Supported formats: `.jpg .jpeg .png .webp .bmp .tif .tiff .gif`. HEIC (iPhone)
is not supported.

## Results

Tested on 16,000 generated images: 800 photos, each put through 19 transforms.
That gives 152,000 duplicate pairs and 127.8 million non-duplicate pairs.

| Mode | precision | recall | F1 |
|---|---|---|---|
| Fast (perceptual hash) | 0.99996 | 0.771 | 0.871 |
| Smart (hash + ResNet-50) | 0.99971 | 0.805 | 0.892 |

Accuracy is not reported. 99.88% of all pairs are non-duplicates, so a program
that always answers "not a duplicate" would score 99.88% and find nothing.

Recall by transform, Fast mode:

| Transform | recall | without normalization |
|---|---|---|
| rotate 90/180/270 | 1.000 | 0.000 |
| black / white / coloured bars | 0.99 | 0.13 |
| JPEG q90/q60/q30, resize, brightness | 0.997 - 1.000 | same |
| crop 5% | 0.609 | 0.608 |
| crop 10% | 0.035 | 0.031 |
| crop 20% | 0.000 | 0.000 |

Two things this shows. Rotations go from 0.000 to 1.000 because of the
normalization step — without it, a rotated copy sits about 32 bits away from its
own original, the same distance as a completely unrelated photo. And cropping is
where the hash fails, which is basically all of the missing recall.

Smart mode helps in one specific place: crop 5% goes from 0.609 to 0.953. It is
also *worse* on heavily shrunk copies — resize 25% drops from 0.998 to 0.638 —
which is why Fast is the default.

Every number above comes from the scripts in `eval/`. Running them writes the
full tables and plots to `results/`, which is generated rather than committed:
`python -m eval.run_eval` for the hash engine, `python -m eval.run_cnn_eval`
for the three-way engine comparison.

## Setup

You need **Python 3.12 or newer**. Nothing else — no Node, no database server.

**1. Get the code**

```bash
git clone <repo-url>
```

```bash
cd Image2
```

**2. Make a virtual environment**

```bash
python -m venv .venv
```

**3. Install the packages**

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This installs FastAPI, Pillow, imagehash, NumPy, SciPy, send2trash, and PyTorch.
PyTorch is about 250 MB and is only used by Smart mode. The CPU version installs
by default on Windows, which is what this needs.

On Linux or macOS use `.venv/bin/python` instead of `.venv\Scripts\python.exe`
everywhere below.

## Running it

```bash
.venv\Scripts\python.exe -m app.main
```

That starts the server and opens your browser at <http://127.0.0.1:8000>.
`Ctrl+C` stops it. There is no launcher script — this one command is the whole
thing on every platform.

To use a different port, or to not open a browser:

```bash
python -m uvicorn app.main:app --port 9000
```

## Using it

1. Click **Browse** and pick a folder, or paste a path into the box.
   Scanning goes into subfolders too, so a photo in `Gallery` and the same photo
   in `WhatsApp/Media` end up in one group.
2. Pick **Fast** or **Smart**.
   - Fast uses the perceptual hash only. Quick, and the default.
   - Smart runs a CNN over the candidates the hash found. Slower, finds more
     lightly-cropped copies, misses more heavily-shrunk ones. It shows a
     similarity slider.
3. Click **Scan folder** and wait. A second scan of the same folder is almost
   instant, because hashes are cached.
4. Look through the groups. **The best copy is left unchecked, the rest are
   checked.** Uncheck anything you want to keep.
5. Click **Delete**. Everything checked goes to the Recycle Bin.

The Recycle Bin is the undo — there is no separate quarantine folder. On network
drives and some USB sticks there is no Recycle Bin, and the app will tell you
instead of deleting permanently.

Files it cannot read (broken downloads, a PNG named `.jpg`) are counted and
skipped. One bad file does not stop the scan.

Nothing is written into the folder you are cleaning. The cache lives in
`cache/` inside the project.

## How it works

```
folder -> normalize -> pHash x4 rotations -> Hamming search -> group -> review
             |                                     |
      EXIF orientation                       Smart mode only:
      crop uniform borders                   ResNet-50 -> cosine
```

**Normalize first.** Perceptual hashing is not rotation-invariant — pHash reads
low-frequency DCT coefficients, and rotating by 90 degrees rearranges them
completely. So the image is EXIF-corrected and any uniform border is cropped off
before hashing.

**All four rotations are stored.** Each photo's upright hash is compared against
all four rotations of every other photo, so a photo matches its own rotated copy.

**Matching** is Hamming distance between 64-bit hashes. Anything within 10 bits
is a match; anything within 16 is stored as a candidate for Smart mode.
Ten was picked by measuring, not guessing — it is the loosest value that keeps
precision above 0.999.

**Grouping** is connected components, so if A matches B and B matches C, all
three end up in one group.

**Smart mode** embeds only the images the hash already flagged, using ResNet-50's
second-to-last layer (2048 numbers per image), then compares them with cosine
similarity. Nothing is trained — the ImageNet weights are used as-is.

**The keeper** is the photo with the most pixels, then the biggest file, then the
oldest one.

## Project layout

```
app/
  config.py      thresholds and paths
  normalize.py   EXIF rotation, border crop
  hashing.py     perceptual hash
  embeddings.py  ResNet-50 (Smart mode)
  matching.py    distance search, grouping, keeper choice
  db.py          SQLite cache
  scanner.py     the background scan
  thumbs.py      thumbnails
  actions.py     deleting
  main.py        the web server
web/             index.html, app.js, styles.css
eval/            benchmark and scoring scripts
tests/           the test suite
results/         generated by eval/, not committed
```

Two SQLite tables in `cache/index.db`: one for images and their hashes, one for
matched pairs. Groups are not stored — they are rebuilt from the pairs whenever
you move the slider, which is why that is instant.

## Tests

```bash
.venv\Scripts\python.exe -m pytest
```

Nine tests. They build their own images, so no photos are committed to the repo.
Each one pins down a behaviour that would be a real bug if it broke: the
cross-rotation hash match, survival through JPEG recompression, unrelated photos
staying apart, the hex round-trip through uint64, border cropping (and *not*
cropping a dark textured edge), the full transform group, keeper selection, and
the rotation hint the CNN scores on.

## Reproducing the numbers

The benchmark needs two extra packages:

```bash
.venv\Scripts\python.exe -m pip install -r requirements-eval.txt
```

Download and build the dataset (about 1.2 GB, goes in `data/`, gitignored):

```bash
.venv\Scripts\python.exe -m eval.make_dataset
```

Measure the hash:

```bash
.venv\Scripts\python.exe -m eval.run_eval
```

Measure the CNN and the cascade. This one embeds 64,000 images and takes about
45 minutes on a CPU:

```bash
.venv\Scripts\python.exe -m eval.run_cnn_eval
```

There is also a second, messier test set shaped like a real phone backup:

```bash
.venv\Scripts\python.exe -m eval.make_testset
```

```bash
.venv\Scripts\python.exe -m eval.score_testset
```

Nothing is labelled by hand. Two files made from the same photo are duplicates
and two from different photos are not, so the labels come free and the whole
benchmark rebuilds from a seed.

## Known limits

- **Cropping.** Anything past about a 10% crop is not found. This is most of the
  missing recall.
- **The benchmark is easier than a real gallery.** It uses 800 professional
  photos that all look different from each other. A real phone gallery has burst
  shots and near-identical screenshots, so the 0.99996 precision figure is a
  best case, not a promise.
- **No video, no HEIC.**
