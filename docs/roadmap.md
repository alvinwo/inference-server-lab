# Roadmap

Status describes repository delivery, not a promised date.

## Milestone 0 — Project skeleton (`complete`)

Repository metadata, package layout, course CLI, tiny fixtures, CI, documentation foundation, and
checkpoint verification.

Milestone 1 begins by publishing Lesson 1. It must preserve the established learner workspace and
checkpoint contracts unless an architecture decision record explains and tests a change.

## Milestone 1 — Foundations (`in progress`)

Lesson 1 is published. Lessons 2–3 and the tested tiny-model generation path remain planned.

## Milestone 2 — First real server (`planned`)

Publish Lessons 4–7: a real-model adapter, naive and asynchronous servers, streaming, and a
repeatable benchmark harness.

## Milestone 3 — Inference engine (`planned`)

Publish Lessons 8–11: KV cache, padded batching, continuous batching, scheduling, and block-based
memory management.

## Milestone 4 — Production shape (`planned`)

Publish Lessons 12–14: compatibility API, observability, resilience, final engine, and graduation
report.

## Milestone 5 — Advanced track (`planned`)

After Version 1 is stable, add prefix caching, chunked prefill, structured output, speculative
decoding, and quantization one lesson at a time with design and benchmark evidence.
