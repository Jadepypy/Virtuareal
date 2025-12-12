# Virtuareal

```
realtalk_project/
├── main.py                  # The "Control" Loop (Entry Point)
├── config.py                # Global settings (Canvas size, Anchor IDs)
│
├── vision/                  # HARDWARE LAYER
│   ├── tracker.py           # [Milestone I] Locates cards & Calibrates Anchors
│   └── projector.py         # [Milestone I] Handles OpenCV drawing (The "Screen")
│
├── repository/              # DATA LAYER
│   ├── card_db.json         # The database of scripts
│   └── loader.py            # [Milestone II] Reads/Writes code strings
│
└── kernel/                  # LOGIC LAYER
    ├── sandbox.py           # [Milestone III] The "Code Adaptor" (Injects Claim/Wish)
    └── scheduler.py         # [Milestone IV] Topological Sort & Execution
```