# Mac service

The Mac runs two plain services. All decisions (state, prompts, conversation) happen on
the robot; see `../MASTER-PLAN.md`.

| Service | Port | What it is |
|---|---|---|
| LLM | 8081 | `llama-server`: a standard OpenAI-compatible endpoint |
| Face service | 8100 | `face_service.py`: stores face embeddings per person, matches new faces |

Both must listen on the network (`0.0.0.0`) so the VM / robot can reach them.
From the UTM VM, the Mac is `192.168.64.1`.

## 1. LLM

```bash
brew install llama.cpp          # once
llama-server -hf LiquidAI/LFM3-1.7B-GGUF:Q4_0 --alias lfm3-1.7b -c 4096 --jinja --host 0.0.0.0 --port 8081
```

- `--jinja` uses the model's own chat template (thinking is switched off per request).
- `-c 4096`: a full visit (talk, fetch, hand over) with state lines fits comfortably.
- Any other OpenAI-compatible server works too (e.g. LM Studio on port 1234). The robot
  sends a llama.cpp `grammar` to force the reply format; servers that ignore it still work,
  with less guarantee on format.

## 2. Face service

Standard library only, no install needed:

```bash
python3 face_service.py                   # 0.0.0.0:8100
python3 face_service.py --verbose         # print every register / identify / verify
```

Embeddings work now (the simulator sends synthetic ones). Real camera cutouts
(`image_jpeg_b64`) return HTTP 501 until a face model is chosen and tested.
Everything is in memory; nothing is written to disk.

## Check from the VM

```bash
curl http://192.168.64.1:8081/v1/models
curl http://192.168.64.1:8100/health
```

If those hang, check the macOS firewall prompt (allow incoming connections for
`llama-server` and Python), and that VS Code isn't auto-forwarding these ports
(`remote.autoForwardPorts: false`).
