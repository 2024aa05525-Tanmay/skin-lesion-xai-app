# Explainable Skin Lesion Classification — Demo App

A minimal Streamlit app that serves the ResNet-50 / HAM10000 classifier from the
dissertation and explains each prediction with **Grad-CAM++**.

> Research and educational demo only. Not a diagnostic device.

---

## What's here

| File | Purpose |
|---|---|
| `inference.py` | Model, preprocessing, prediction, Grad-CAM++, overlay |
| `app.py` | Streamlit UI |
| `tests/test_smoke.py` | Runs the whole pipeline on random weights (no checkpoint needed) |
| `.github/workflows/ci.yml` | Lint + test on every push |
| `requirements.txt` | Pinned runtime deps, CPU-only PyTorch |
| `requirements-dev.txt` | Adds `pytest` and `ruff` |
| `packages.txt` | apt packages Streamlit Cloud needs for OpenCV |
| `ruff.toml` | Lint rules |

The CI is deliberately tiny: **lint, then run one smoke test**. It never downloads
the trained weights, so a full run finishes in about a minute.

---

## Step 1 — Get the checkpoint out of Colab

The notebook already writes `resnet50_ham10000_best.pt` to
`MyDrive/skin_lesion_xai/`. Download it to your laptop. It's roughly 95 MB.

## Step 2 — Create the GitHub repo

```bash
git init
git add .
git commit -m "Streamlit app + CI for HAM10000 CAM explanations"
git branch -M main
git remote add origin https://github.com/<you>/skin-lesion-xai-app.git
git push -u origin main
```

Make it **public** — GitHub Actions minutes are unlimited on public repos, and
Streamlit Community Cloud needs the repo anyway.

## Step 3 — Host the weights (not in git)

`.gitignore` excludes `*.pt` on purpose: a 95 MB binary in git history makes every
clone slow and sits right on GitHub's 100 MB file limit.

Instead, on GitHub: **Releases → Draft a new release → tag `v1.0` → attach
`resnet50_ham10000_best.pt` → Publish.** Copy the asset's download URL, which looks
like:

```
https://github.com/<you>/skin-lesion-xai-app/releases/download/v1.0/resnet50_ham10000_best.pt
```

## Step 4 — Run it locally first

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                        # should print 5 passed
export MODEL_URL="<your release URL>"
streamlit run app.py
```

Debugging locally is far quicker than debugging a failed cloud build.

## Step 5 — Deploy

Go to <https://share.streamlit.io>, connect GitHub, pick this repo, main branch,
`app.py`. Under **Advanced settings → Secrets**, add:

```toml
MODEL_URL = "https://github.com/<you>/skin-lesion-xai-app/releases/download/v1.0/resnet50_ham10000_best.pt"
```

That's the whole CD story: Streamlit Cloud rebuilds automatically on every push to
`main`, so CI checks the code and Streamlit ships it. No deploy job, no secrets in
Actions, nothing to maintain.

---

## Known constraints

- **Memory.** Streamlit Community Cloud's free tier gives roughly 1 GB. PyTorch plus
  ResNet-50 plus one image lands around 500–600 MB with the CPU-only wheel, so it
  fits, but not with much room. `torch.set_num_threads(1)` in `inference.py` is
  there for this reason. If you hit an OOM, try Hugging Face Spaces (16 GB) instead.
- **CPU wheels.** The `--extra-index-url` line in `requirements.txt` matters. Without
  it pip pulls the CUDA build of PyTorch (~2.5 GB) and the build times out. If the
  build fails on the pinned `+cpu` resolution, drop the version pins to
  `torch` / `torchvision` and keep the index line.
- **Cold start.** The app sleeps after about 12 hours of inactivity and takes 30–60 s
  to wake, including the one-time checkpoint download.
- **Grad-CAM++, not Score-CAM.** Score-CAM needs 30–60 s per image on CPU. Grad-CAM++
  measures about 0.5 s, which is what makes the app usable. This matches the
  dissertation's own recommendation for interactive use.
