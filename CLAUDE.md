# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Automated crypto trading bot using LLM (via OpenRouter) for trading decisions on Hyperliquid perpetual futures. Features a NiceGUI web dashboard for monitoring and control.

## Tech Stack
- **Backend:** Python 3.12, NiceGUI (web framework)
- **Exchange:** Hyperliquid Testnet API
- **LLM:** OpenRouter API (default: `google/gemini-3-flash-preview`)
- **Indicators:** TAAPI.io (free tier, 1 req/15s)
- **Deploy:** Railway (autodeploy from GitHub)

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (starts web server on port 8080)
python main.py

# Run tests
pytest
pytest -v src/backend/  # specific directory

# View production logs (Railway CLI)
railway logs --service crypto-alpha-arena --tail 100
```

## Project Structure
```
src/
├── backend/
│   ├── agent/
│   │   └── decision_maker.py    # LLM decision logic, fallback models
│   ├── indicators/
│   │   ├── taapi_client.py      # Technical indicators
│   │   ├── taapi_cache.py       # Indicator caching (TTL 60s)
│   │   ├── sentiment_client.py  # Fear & Greed Index
│   │   ├── whale_alert_client.py
│   │   └── enhanced_context.py  # Aggregated market context
│   ├── trading/
│   │   └── hyperliquid_api.py   # Exchange API wrapper
│   ├── bot_engine.py            # Main trading loop
│   └── config_loader.py         # ENV config
├── gui/
│   ├── app.py                   # NiceGUI main app, autostart logic
│   ├── pages/                   # Dashboard, positions, history, etc.
│   └── services/
│       ├── bot_service.py
│       └── state_manager.py
└── main.py                      # Entry point
```

## Key Files

### `src/backend/bot_engine.py`
- Main trading loop (`_main_loop`)
- Anti-churn logic with cooldowns:
  - `MIN_DIRECTION_CHANGE_COOLDOWN = 1800` (30 min)
  - `MIN_HOLD_TIME = 900` (15 min)
  - `MIN_TRADE_INTERVAL = 300` (5 min)
- Position tracking: `_is_trade_allowed()`, `_record_trade()`, `_clear_position_tracking()`
- Uses `self.state.positions` for current positions

### `src/backend/agent/decision_maker.py`
- LLM prompt construction (system + user context)
- Fallback model list for 402/429/404 errors:
  ```python
  FALLBACK_MODELS = [
      "google/gemini-2.0-flash-exp:free",
      "google/gemma-3n-e2b-it:free",
      "qwen/qwen3-coder:free",
      "mistralai/devstral-2512:free",
      "nvidia/nemotron-nano-9b-v2:free",
  ]
  ```
- `_switch_to_fallback()` - cycles through free models
- `_supports_tools` flag - disabled for free models

### `src/gui/app.py`
- Autostart logic with retry (5 attempts, exponential backoff)
- Triggered by `AUTOSTART_BOT=true` env var

## Environment Variables (Railway)
```
OPENROUTER_API_KEY=...
HYPERLIQUID_PRIVATE_KEY=...
TAAPI_API_KEY=...
LLM_MODEL=google/gemini-3-flash-preview
ASSETS=BTC ETH
AUTOSTART_BOT=true
TRADING_MODE=AUTO
```

## Recent Fixes (Dec 2025)

1. **NameError fix** - `positions` -> `self.state.positions` in bot_engine.py:1433
2. **Anti-churn system** - Added cooldowns to prevent rapid direction changes
3. **Fallback models** - Auto-switch to free models on 402/429/404
4. **Autostart** - Bot starts automatically after Railway deploy
5. **Tool use disabled** - For free models that don't support it

## Known Issues / TODOs

### Potential Optimizations (not yet implemented)
1. Reduce LLM prompt size (~1500 tokens, could be ~400)
2. Remove 15s waits between assets (use bulk TAAPI)
3. Compact JSON (remove `indent=2`)
4. Limit indicator series to last 3-5 values
5. Make anti-churn params configurable via ENV
6. Reduce leverage policy (currently suggests 3-10x)

## Monitoring

### Check logs
```bash
railway logs --service crypto-alpha-arena --tail 100
```

### Key log patterns
- `INFO:root:Sending request to OpenRouter` - LLM call
- `INFO:src.backend.bot_engine:BTC: HOLD` - Decision made
- `WARNING:root:FALLBACK: Switched from` - Model fallback triggered
- `ERROR:src.backend.bot_engine:Error in main loop` - Check for bugs

## Git Config
```bash
git config user.email "198627965+fede-crypto-lab@users.noreply.github.com"
git config user.name "fede-crypto-lab"
```

## Files to Ignore
- `error.log` - Runtime errors (gitignored)
- `trade_history.csv` - Trade log (gitignored)
- `llm_requests.log` - LLM debug log
