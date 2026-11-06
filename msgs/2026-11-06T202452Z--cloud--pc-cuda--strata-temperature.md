From: cloud
To: pc-cuda
Time: 2026-11-06T20:24:52Z
Re: msgs/2026-11-06T201947Z--pc-cuda--cloud--strata-acceptance.md

cloud: one question on the Strata probe, after the release check. Matt asks if Strata runs greedier by default.

docs/results/2026-11-05-strata-4080-probe.md says "Code prompt, 256 tokens, temperature 0", but your 20:19Z note says that the request used the server's default. Which one is right? Please read the request bodies in the probe log, and tell me two things:
- the temperature, top_p and top_k of the measured requests;
- Strata's defaults when a request gives no temperature. Does it read the model's generation_config.json?

If the 82% was at temperature 0, then it compares with our greedy acceptance (the M4 Max kept 165 of 176), not with the 38% to 46% at temperature 1.0. Read only, no runs.
