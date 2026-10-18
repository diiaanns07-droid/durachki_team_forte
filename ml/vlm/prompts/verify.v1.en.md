---
id: verify
version: v1
lang: en
placeholders: [n_frames, frame_times, boxes_note, candidate_type, candidate_name, candidate_facts, type_catalog]
---
You see {{n_frames}} frames of one fragment from a ceiling camera above a bank cash desk, in order; frame times (s): {{frame_times}}.
{{boxes_note}}

The automatic system suspects a violation:
- type: {{candidate_type}} "{{candidate_name}}"
- cues found by the system: {{candidate_facts}}

Type reference (in Russian: what is visible on camera and what must NOT be claimed):
{{type_catalog}}

Task: decide from the frames whether the described violation is VISIBLE.
Rules:
1. Describe only what is visible: people, where they are, what is in their hands, objects (phone, cash, safe, bag, door, food, drink).
2. Never claim what cannot be seen: amounts, accounting, banknote authenticity, identity, intentions or authority (clearance), contents of closed bags or the safe. Do not cite regulations, clauses or rules — only what is visible.
3. Infer roles from position and objects: "cashier" sits at the cash desk; "cash collector" carries a green collection bag; otherwise "visitor" or "person".
4. Normal service of a client across the desk/window is not a violation.
5. If the frames do not allow a confident decision, answer visible=false or low confidence and explain in uncertain_because.
6. If a different violation from the reference is visible, give its type_id.
7. observed_facts: 1–3 short sentences IN RUSSIAN about visible facts only (e.g. «Кассир держит у уха предмет, похожий на мобильный телефон»).

Answer with ONE JSON object and nothing else:
{"visible": true or false, "type_id": "V0x", "observed_facts": ["..."], "confidence": number 0..1, "best_frame": frame number 1..{{n_frames}}, "visible_frames": [frame numbers where the cue is visible], "uncertain_because": "..."}
