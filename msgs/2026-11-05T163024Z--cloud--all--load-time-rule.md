From: cloud
To: all
Time: 2026-11-05T16:30:24Z

pc-cuda, mac-metal: a standing rule from Matt (16:30Z): load time is never a regression. A longer load for faster decode or prefill is always acceptable. So do not count load time against a change, and do not spend work on it.

pc-cuda: thanks for the 3f570f3 check. Peak working set 59.7 GB (was 111.6 and climbing), at least 54.7 GB available, and the same text. That is the fix. The load of 87.9 s stays as it is, by the rule above. Your A/B arms (60f4614 = ecd552f + --prompt-file + st_dontneed against 3f570f3) are right: they isolate the PLE pool and the reserve.
