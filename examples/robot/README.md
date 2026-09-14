# LIBERO Robot Example

This example runs a vision-language agent that controls a Franka Panda robot in LIBERO.

## Setup

Follow [`libero/README.md`](libero/README.md) to install LIBERO and its assets in a Python 3.8 environment.

Install OpenRLHF-Agent in a separate Python 3.10 or newer environment:

```bash
conda create -n openrlhf-agent python=3.10 -y
conda activate openrlhf-agent
python -m pip install -e .
```

## Run

Start the environment server from the LIBERO environment:

```bash
conda activate libero
python -m examples.robot.libero.env_server --video-dir examples/robot/data/episodes/videos
```

Start a compatible vLLM server. Then run an episode from the OpenRLHF-Agent environment:

```bash
conda activate openrlhf-agent
python -m examples.robot.runtime_demo \
  --env-url http://127.0.0.1:8010 \
  --base-url http://127.0.0.1:8009
```

Each `act` call requests a relative movement of at most `0.3 m` and a rotation of at most `1 rad`. The environment server executes it over one second at `20 Hz`.
