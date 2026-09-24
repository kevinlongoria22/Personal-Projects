# Investing Tool Design

A lightweight market-setup dashboard inspired by the AMD example discussed in the shared ChatGPT thread.

## Live site

**https://kevinlongoria22.github.io/Personal-Projects/**

`cache.json` auto-refreshes on weekdays via a scheduled GitHub Action.

## Run locally

From this folder:

```bash
python -m http.server 8000
```

Then open:

http://localhost:8000

## What it does

- Compares investment setups across symbols
- Scores catalyst quality, trend strength, earnings power, and risk discipline
- Shows a simple scenario chart for a bull/base case
- Helps evaluate whether a setup is meaningful enough to warrant attention
