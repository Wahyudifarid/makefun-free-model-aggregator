# makefun-free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Wahyudifarid/makefun-free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Wahyudifarid/makefun-free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A deterministic free-model aggregator server for the make.fun bounty "The best free AI model aggregator". It serves the OpenAI Chat Completions and Anthropic Messages APIs (JSON and SSE streaming) over ten free-tier providers, routing each request to the highest-rated free model that meets its requirements, with failover to the next-best model on refusal. Grade-local: 99/100 on all three practice trials.

## Run it

```sh
node main.ts
```

The trial harness provides MAKEFUN_ENV_URL, MAKEFUN_PROMPT_FILE, MAKEFUN_TRIAL_SEED and MAKEFUN_TRIAL_NOW; the server registers with the env, serves its 20 requests, and exits 0 when the env says done. `node main.ts --list` prints the registered catalog.

---

<sub>Entered on [make.fun](https://make.fun/?ref=5p2p2a05&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
