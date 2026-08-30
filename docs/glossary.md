# Beginner glossary

**Token:** A numeric unit the tokenizer uses to represent part of text. It may be a word, part of a
word, punctuation, or whitespace.

**Logit:** An unnormalized score the model assigns to a possible next token. Sampling transforms
logits into probabilities.

**Sampling:** Selecting the next token from model scores, either deterministically (such as greedy
selection) or probabilistically with controls such as temperature and top-k.

**Prefill:** Processing all prompt tokens to produce the first next-token scores and populate the KV
cache. Its work grows with prompt length.

**Decode:** Generating later tokens one step at a time while reusing cached prompt and output state.

**KV cache:** Stored attention keys and values from earlier tokens. It avoids recomputing them during
decode, while consuming memory for every live sequence and layer.

**Batch:** Multiple sequences processed in one model invocation. Their lengths and current phases
may differ, so the engine needs masks and metadata.

**Scheduler:** The component that chooses which requests and tokens run in the next engine
iteration, subject to fairness, memory, and token budgets.

**TTFT (time to first token):** Time from submitting a request until its first output token becomes
available. Queueing and prefill usually dominate it.

**TPOT (time per output token):** Average time between generated output tokens after the first one.

**ITL (inter-token latency):** The distribution of delays between consecutive streamed tokens;
unlike an average, it reveals stalls and tail latency.

**Throughput:** Work completed per unit of time, reported as requests or tokens per second with the
workload and hardware stated.
