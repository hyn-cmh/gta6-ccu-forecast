# gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/hyn-cmh/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/hyn-cmh/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

An empirical forecaster for GTA VI's launch-day Twitch viewers, hour by hour. It rebuilds every comparable launch's release-day curve from the case data, weights those launches by how closely their opening scale and pre-release run-in match this case's, and reads weighted quantiles of their log ratios to hour 1 — blended with a normal fit at the tails and calibrated by leave-one-out backtests under the contest's own pinball loss.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<case-dir> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=1pk8yq3w&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
