Status: PASS
Branch: mac/qwen4-metal
Commit: c404931f1ac4cf42ce029f96de613827a04e8637
Platform: macos
Agent: mac-metal
Time: 2026-11-04T23:49:44Z

c404931 = 611254c rebased onto main 878d4d4 (PR #16). The one conflict was docs/log.md (both appended): main's entries
first, then mine.
macOS 27.2, M4 Max 128 GB, AC power. Clean build: PASS, with the one old warning. make test with MOVA_DIR set: all tests
passed, 29 of 29, none skipped, no "not on Metal". The test output is the same as 611254c's.
