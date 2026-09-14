## Setting up the LIBERO Environment Server

This guide walks through setting up the LIBERO environment server and verifying it with a simple terminal test. The server runs LIBERO in Python 3.8 and exposes it to OpenRLHF-Agent over HTTP.

### 1. Create the LIBERO Conda Environment

Run the following commands from the repository root:

```bash
conda create -n libero python=3.8 -y
conda activate libero
python -m pip install \
  libero==0.1.1 \
  'huggingface-hub>=0.23' \
  'imageio[ffmpeg]>=2.34' \
  'Pillow>=10' \
  fastapi==0.115.6 \
  uvicorn==0.30.6
```

### 2. Configure LIBERO Assets

Import LIBERO once to create its config file:

```bash
python -c "import libero.libero"
```

LIBERO stores its paths in `$LIBERO_CONFIG_PATH/config.yaml`, or `~/.libero/config.yaml` when the environment variable is unset.

Download the assets:

```bash
huggingface-cli download jadechoghari/libero-assets \
  --local-dir "$(python -c 'from libero.libero import get_libero_path; print(get_libero_path("assets"))')"
```

### 3. Start the Environment Server

```bash
python -m examples.robot.libero.env_server \
  --max-sessions 1 \
  --video-dir data/robot/videos
```

The server listens on port `8010`. `--max-sessions` limits the number of active simulators, and additional sessions wait for a free slot. Omit `--video-dir` to disable recording.

For concurrent training rollouts, increase the session limit according to the available CPU, GPU, and memory:

```bash
python -m examples.robot.libero.env_server --max-sessions 8
```

### 4. Test the Environment Server

Check that the server is running:

```bash
curl http://127.0.0.1:8010/health
```

The expected response is:

```json
{"status":"ok"}
```

Create and close one simulator session:

```bash
SESSION_ID=smoke

curl -X POST "http://127.0.0.1:8010/sessions/$SESSION_ID/new" \
  -H "Content-Type: application/json" \
  -d '{"suite":"libero_spatial","task_id":0,"init_state_id":0}'

curl -X POST "http://127.0.0.1:8010/sessions/$SESSION_ID/close" \
  -H "Content-Type: application/json" \
  -d '{}'
```
