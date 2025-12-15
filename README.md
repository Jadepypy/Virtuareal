# Virtuareal

# Projector-Camera Card System Documentation

# 1 Objective

Virtuareal is an interactive system that allows users to use boards on a table surface to customize picture processing pipelines and observe the process and results simltaneously. Users can edit the pipeline by merely moving the boards representing code blocks, and observe real time feedbacks of their manipulations.

# 2 Description

Picture Coming Soon!

# 3 Deliverables

## 3.1 Project plan

The BIG IDEA is to visualize the process of 2D-convolution into an interactive desktop. With simultaneous manipulations users can directly understand how different programs affect pictures and customize their own pipelines to process pictures through convolutions and all kinds of programmable code blocks!

## 3.2 Functioning project

The functioning device is an interactive desktop that detects the cards on the table, runs corresponding programs and demonstrates the real-time results.

Picture coming soon!

## 3.3 Documentation of design process

## 3.3.1 Function Definition

The system should contain 3 parts: observation, calculation and demonstration. 

The observation part should scan the whole desktop and return a dictionary of card information, which is wrapped up in a self-built Card class, as the keys and the cards' corresponding positions as values.

The calculation part should use the dictionary of card information-position as an input, and process these data in a certain logic to simulate the process of using code block cards to "process" the picture. This part should return the processed pictures to be projected.

The demonstration part should arrange the all the information in a canvas and project the canvas through a projector to the desktop.

## 3.3.2 Roadmap Confirmation

(a) The observation part

    We initially planned to only detect the relative positions to document the 


## 1.1 Code Structure

```
.
├── main.py                  # Entry Point & Controller
│   ├── Init: OpenCV camera, Window, and Aruco Detector.
│   ├── Loop: Captures frames, manages Homography calibration.
│   └── Task: Coordinate CardSystem updates and calls the render loop.
│
├── card_engine.py           # Logic Core & State Management
│   ├── CardSystem: Static database storing card state (Position, ID, TTL).
│   ├── BaseCard: Parent class handling spatial inputs/outputs and dependency resolution.
│   ├── Concrete Classes: ImageCard, KernelCard, KernelAdditionCard.
│   └── Factory: Instantiates physical cards based on JSON config.
│
├── common.py                # Shared Resources
│   ├── Config: Constants (Resolution, Anchor IDs, TTLs).
│   ├── Utils: Math (Homography), Image loading, Projections.
│   └── Scripts: Raw string Python scripts injected into cards at runtime.
│
└── cards.json               # (Required External File)
    └── Maps Marker IDs to Card Types and initialization scripts.
```


## 2. Execution Flow & Architecture

**Prerequisite:** Create `cards.json` defining marker behaviors.
Example: `{"10": {"init_script": "card = ImageCard(np.zeros((100,100,3)))"}}`

### Step-by-Step Workflow

1. **Input Acquisition (main.py)**
   - Captures raw frame from Webcam.
   - Detects ArUco markers.
   - **Calibration**: Uses markers defined in `ANCHOR_IDS` (corners) to compute the Homography Matrix (M) mapping Camera Space -> Projector Space.

2. **Physical Synchronization (main.py -> card_engine.py)**
   - Transforms detected marker coordinates using M.
   - **Register**: Calls `create_physical_card_instance`. If new, Factory spawns object.
   - **Update**: Updates coordinates in `CardSystem`.
   - **Decay**: Decrements TTL for unseen cards; unregisters dead cards.

3. **Simulation Loop (main.py -> card_engine.py)**
   - Runs for `MAX_CHAIN_DEPTH` iterations to propagate logic chains.
   - **Resolve**: Each card probes `LEFT`/`RIGHT` for neighbors (spatial hit-testing).
   - **Execute**: Runs injected `run_logic()` (from `common.py` scripts).
   - **Spawn**: If logic generates output (e.g., Filter result), a Virtual Card is registered `DOWN` relative to parent.

4. **Rendering (card_engine.py)**
   - Draws visual content (images/matrices) onto the global `projector_canvas`.
   - Draws connection lines (green/red) between dependent cards.

5. **Output (main.py)**
   - `cv2.imshow` displays the final `projector_canvas` full screen.

### System Flowchart

```
[ Web Camera ]
      |
      v
[ main.py: Aruco Detection ] ----> [ Anchor History ]
      |                                  |
      | (Raw Corners)                    v
      |                          [ Compute Homography (M) ]
      v                                  |
[ Transform Coords ] <-------------------|
      |
      v
[ card_engine: Sync Physical ]
   - Spawn new cards (Factory)
   - Update positions (Smoothing)
   - Kill old cards (TTL)
      |
      v
[ Simulation Loop (Depth 0..N) ] <===========|
      |                                      |
      |-- 1. Resolve Inputs (Spatial Query)  |
      |-- 2. Run Logic Scripts (common.py)   |
      |-- 3. Spawn Virtual Cards (Output) ---|
      |
      v
[ card_engine: Render ]
      |
      v
[ Projector / Screen ]
```
