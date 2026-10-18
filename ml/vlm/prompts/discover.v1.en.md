---
id: discover
version: v1
lang: en
placeholders: [n_frames, frame_times, checklist, type_catalog]
---
You see {{n_frames}} consecutive frames from a ceiling camera above a bank cash desk; frame times (s): {{frame_times}}.

Check whether anything from this checklist is visible (key — description):
{{checklist}}

Violation type reference (in Russian):
{{type_catalog}}

Rules:
- Mark an item ONLY when there is a concrete visible fact (an object in a hand, an open safe door, a person at the cashier's desk, etc.).
- Normal cashier work, serving a client across the desk/window, counting money on the desk in view of the camera is NOT a violation.
- Never claim what cannot be seen: amounts, authenticity, identity, intentions, authority (clearance), contents of bags or the safe. Do not cite regulations or clauses.
- If in doubt, do not mark. An empty list is the normal answer for most fragments.
- Key "other" is only for a concrete visible action that breaks cash-handling discipline and fits no other item; describe it in observed_facts.
- observed_facts: IN RUSSIAN, 1–2 short sentences about visible facts.

Answer with ONE JSON object and nothing else:
{"findings": [{"item": "<checklist key>", "observed_facts": ["..."], "confidence": number 0..1, "frame": frame number 1..{{n_frames}}}], "uncertain_because": "..."}
If nothing is found: {"findings": [], "uncertain_because": ""}
