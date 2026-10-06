# Real-Time Sign Phrase Recognition (Custom Phrases Edition)

This is an upgraded version of the original A/B/C letter-detection tutorial,
reworked to recognize full **phrases** instead of single letters:

- hello
- how are you
- see you later
- i need help
- thank you
- yes
- no

(Edit the `actions` list in `function.py` to add, remove, or rename phrases.)

## Important: there is no pretrained model included

This is the honest part. A model that actually recognizes "hello" vs.
"thank you" vs. "I need help" has to be trained on **real video of someone
signing those exact phrases**, many times over. No such dataset is bundled
with this project (and there isn't a free, ready-made one for this specific
phrase list either). So instead of shipping a fake/random model that would
*look* like it works but wouldn't actually recognize anything, this package
gives you the full pipeline to **record your own training data in about
15-20 minutes** and train a real model on it.

What's improved vs. the original tutorial:
- Tracks **both hands + body pose** (MediaPipe Holistic) instead of just one
  hand, so the model has far more signal to tell phrases apart.
- Trains with dropout + early stopping to reduce overfitting.
- Live app now builds a running sentence and **speaks recognized phrases
  out loud** (text-to-speech, fully offline via `pyttsx3`).

## Setup (VS Code)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

You need a real, accessible webcam — this won't work in a cloud/remote
container with no camera.

## Step 1 — Record your training data

```bash
python collectdata.py
```

This walks you through recording 40 short clips (1 second / 30 frames each)
for every phrase in `actions`. Watch the on-screen prompt — it tells you
which phrase to sign and when recording starts.

Tips for better accuracy:
- Vary your position slightly between recordings — don't sit frozen in the
  same exact spot every time.
- Keep consistent lighting and stay inside the camera frame.
- Sign at a natural pace.
- 40 recordings per phrase is a reasonable starting point — more is better,
  and recording over a couple of sessions/lighting conditions helps a lot.

## Step 2 — Train the model

```bash
python trainmodel.py
```

Trains an LSTM on the keypoint sequences you just recorded, then saves
`model.json` + `model.h5`. It prints validation accuracy at the end and
warns you if it's low (usually means: record more data, with more
variation).

## Step 3 — Run real-time recognition

```bash
python app.py
```

A window opens showing your webcam feed with hand/pose landmarks drawn on
top, a confidence bar for the current prediction, and a running sentence
across the top of the screen. Recognized phrases are also spoken aloud.

Controls:
- `q` — quit
- `c` — clear the sentence on screen

## Setting realistic expectations

This setup does **isolated phrase recognition** — it recognizes each
phrase as one whole canned gesture/sentence, the way the original project
recognized whole letters. It is not the same thing as continuous sign
language translation (stringing together arbitrary words on the fly),
which is a much harder, still-unsolved research problem even for large
tech companies. With a clean, fairly large set of recordings (40-100+ per
phrase, ideally from more than one person/session) this approach can work
quite well for a fixed vocabulary like the one above — but don't expect
it to generalize to phrases or signers it has never seen.
