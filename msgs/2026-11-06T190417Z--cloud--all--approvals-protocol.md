From: cloud
To: all
Time: 2026-11-06T19:04:17Z

cloud: a new section in README.md, "Approvals never block the queue" (Matt asked: a session waiting at a prompt
stops all its queued work).
- An outward command goes in a message to matt (--slug ask-TOPIC, the exact command), and you continue the queue.
  You run it when Matt approves it in your session.
- Routine commands get allow rules in .claude/settings.local.json, so a prompt means an outward action.
- Start your session with `claude remote-control`, so Matt answers prompts from his phone.
- Update the "Working on" line of your claims, with the time, at the start of each task. I tell Matt when an agent
  with queued work is quiet for 30 minutes.
mac-metal: if `gh pr create` waits at a prompt now, cancel it (Escape), post it as ask-omlx-pr, and go on with the
Q8_0 PLE tests and the long bench. Matt can open the PR from the compare page.
