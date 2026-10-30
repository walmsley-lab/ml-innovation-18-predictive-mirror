# Demo-first local video training (no cloud required)

The original recurrent synthetic-world checkpoint cannot be expected to predict webcam video. This optional video-specific recurrent **residual model starts at exact persistence** and trains on *observed* frame changes, not simulator truth. It is a deliberately simple baseline: success must beat persistence on held-out frames.

Install optional camera dependency on your own computer, then record locally (default 32px grayscale at 10 Hz for 120 seconds; video is never uploaded):

```bash
python -m pip install opencv-python
python -m video_train capture --seconds 120 --size 32 --out recordings/webcam.npz
python -m video_train fit --input recordings/webcam.npz --steps 500 --out runs/webcam
cat runs/webcam/metrics.json
python -m mirror --architecture video_residual --parameters runs/webcam/parameters.npz --size 32 --fps 10
```

Move hands/objects smoothly in front of a static camera while recording; try similar *unseen* movements during the live mirror. The fit command holds out the last 20% of the recording, with a 10% chronological embargo between train and validation, and prints both trained-model and last-frame persistence MSE. If `beats_persistence` is false, be transparent about the negative result and adjust the dataset/objective before claiming predictive superiority. Validation from one recording is not independent confirmation. For a stronger test, use a completely separate recording later.

Recordings/ is gitignored; do not commit personal camera footage. The existing live mirror shows observed/predicted/error and internal activity, with next-frame prediction committed before the next application-level webcam read. Camera exposure timing and actual FPS remain unverified. This PR adds no GPU dependency, provider requirement or remote recording.
