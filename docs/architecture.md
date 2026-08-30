# Architecture

Inference Server Lab has three deliberately separate kinds of code:

- `src/inference_lab/` becomes the final, polished engine.
- `course/` manages lessons and verification; it is not part of the serving runtime.
- `lessons/` stores self-contained starter, solution, test, and benchmark snapshots.

The target runtime flow is:

```text
HTTP/API ingress
      ↓
request lifecycle and streaming
      ↓
engine queue → scheduler → model runner
                    ↕             ↕
              KV manager     model adapter
                    ↓             ↓
             memory blocks   PyTorch model/device
```

Ingress validates requests and creates cancellable request state. The engine owns the asynchronous
loop. The scheduler decides which prefill and decode tokens fit the next iteration's budget. The
model runner prepares tensors through a narrow model-adapter interface. The KV manager owns cache
capacity and releases it on every terminal path. Streaming carries incremental output back without
letting slow clients block the engine.

The curriculum introduces those boundaries before optimizing them. A full-sequence reference path
remains available to check cached and batched paths. CPU correctness is mandatory; device-specific
acceleration is optional. See the
[full project design](superpowers/specs/2026-08-29-inference-server-learning-project-design.md)
for invariants, model scope, benchmark methodology, and lesson design.
