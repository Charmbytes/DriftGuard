---
title: DriftGuard
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Catches an LLM agent drifting and revokes its permissions
---

# DriftGuard — live demo

Continuous behavioural re-certification and capability-token revocation for
LLM agents. This Space runs the full prototype: a customer-support refund
agent, the 42-probe test suite, drift scoring, token revocation, the human
approval queue and the hash-chained audit log.

**Try it:** open the *New here?* panel at the top, then press **Inject drift**
followed by **Run probe suite**, and open the **Why did it drift?** tab.

Notes for visitors:

- The agent runs on a built-in deterministic stand-in for the LLM, so no API
  key is needed and results are reproducible.
- Everyone who opens this Space shares one demo. If the screen is not what you
  expect, press **Reset demo**.
- Data is wiped whenever the Space restarts.

Source code and documentation: <https://github.com/Charmbytes/DriftGuard>
