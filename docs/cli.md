# CLI commands

Use the [Python SDK quickstart](/docs/quickstart/) to edit model weights. The experimental benchmark CLI adds a sequential evaluation workflow:

```sh
wsm benchmark preflight --config /path/to/config.json
wsm benchmark run --config /path/to/config.json --output /path/to/new-campaign
wsm benchmark report /path/to/campaign
wsm benchmark regrade /path/to/campaign --trial-id task.agent.0000
```

Follow the [benchmark guide](benchmark.md) for the synthetic local fixture, explicit execution permissions, required configuration, and offline reports.
