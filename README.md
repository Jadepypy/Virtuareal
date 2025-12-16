# Virtuareal

# Projector-Camera Card System Documentation

# 1 Objective

Virtuareal is an interactive system that allows users to use boards on a table surface to customize picture processing pipelines and observe the process and results simltaneously. Users can edit the pipeline by merely moving the boards representing code blocks, and observe real time feedbacks of their manipulations.

# 2 Description

![e833ce5220a802381a1b688425ee8e7c](https://github.com/user-attachments/assets/521248e6-e89c-4912-a959-b248b6c45adc)

# 3 Deliverables

## 3.1 Project plan

## 3.1.1 BIG IDEA

The BIG IDEA is to visualize the process of 2D-convolution into an interactive desktop. With simultaneous manipulations users can directly understand how different programs affect pictures and customize their own pipelines to process pictures through convolutions and all kinds of programmable code blocks!

## 3.1.2 Timeline

Nov 15: Set up the camera → Raspberry Pi → projector feedback loop

Nov 22: Implement reliable paper card detection and basic filter triggering

Dec 1: Build real-time, stackable filter pipeline and test responsiveness

Dec 5: Refine physical layout and run pilot tests with 1–2 users to assess interaction clarity

Dec 14: Final demo + Documentation

## 3.1.3 Parts Description
![setup](https://github.com/user-attachments/assets/bdb72b38-7fcf-469e-ab51-342bb1bbf4c6)

- 1x Projector
  
- 1x Whiteboard
  
- 1x Rhaspberry Pi
  
- 1x HDMI wire
  
- 10x Card board with QR code

- 10x magnet

## 3.2 Functioning project

The functioning device is an interactive desktop that detects the cards on the table, runs corresponding programs and demonstrates the real-time results.

![4aec6ae93e265b7a93cc17a9c8e999bd](https://github.com/user-attachments/assets/71bea9ad-e560-4932-8324-bd0e673bec9e)

## 3.3 Documentation of design process

## 3.3.1 Function Definition

The system should contain 3 parts: observation, calculation and demonstration. 

The observation part should scan the whole desktop and return a dictionary of card information, which is wrapped up in a self-built Card class, as the keys and the cards' corresponding positions as values.

The calculation part should use the dictionary of card information-position as an input, and process these data in a certain logic to simulate the process of using code block cards to "process" the picture. This part should return the processed pictures to be projected.

The demonstration part should arrange the all the information in a canvas and project the canvas through a projector to the desktop.

## 3.3.2 Roadmap Confirmation

(a) The observation part

We initially planned to only detect the relative positions to document the collaboration relationship between cards. However, it turned out that the there should be 4 anchors to sign the boarder of the desktop and return the absolute positions of cards.

(b) The calculation part

We initially planned to process the picture with the program blocks at a sequence of from left to right and then return a final result. However, we finally confirmed that, the layout allowing the code vlock cards to look for import with "tentacles" and demonstrating the step result immediately, has a better demonstration effect.

(c) The demonstration part

We initially planned to read the picture put in a certain area from camera and demonstrate the picture to another certain area. However, since the camera has a rather low resolution, we decided to build the picture in "picture" Cards. Besides, since the layout of calculation cards is decided to be demonstrating results by step, the idea of sector demonstration is eliminated.

In conclusion, the overview of layout of is demonstrated below:


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
*Example:* `{"10": {"init_script": "card = ImageCard(np.zeros((100,100,3)))"}}`

### Step-by-Step Workflow

#### 1. Input Acquisition (`main.py`)
* **Hardware Initialization**: Initializes `cv2.VideoCapture(0)` at a resolution of 1280x720 to capture the raw physical workspace.
* **Marker Detection**: Uses `cv2.aruco.ArucoDetector` configured with `common.ARUCO_DICT_TYPE` to locate the corners of ArUco markers in the frame.
* **Calibration (The "M" Matrix)**:
    * **Anchor Logic**: The system filters for specific marker IDs defined in `common.ANCHOR_IDS`. These serve as fixed calibration points (usually the 4 corners of the table). Their positions are averaged over time in `anchor_history` to stabilize signal jitter.
    * **Homography Calculation**: Uses `common.get_homography_from_history` to compute **`M`**.
    * **What is M?**: `M` is a **3x3 Homography Matrix**. It mathematically defines the perspective transformation required to map a 2D point from **Camera Space** (the distorted, angled view of the webcam) to **Projector Space** (the flat, rectilinear 1920x1080 coordinate system of the screen).
    * **Inverse (`M_inv`)**: Computed as `np.linalg.inv(M)`. This allows the system to map virtual boundaries *back* onto the raw camera feed for debugging alignment.

#### 2. Physical Synchronization (`main.py` -> `card_engine.py`)
* **Coordinate Transformation**: The system iterates through non-anchor markers. It takes the **bottom-left corner** of the physical marker and applies `common.transform_point(point, M)` to calculate its precise `(cx, cy)` location on the digital canvas.
* **Registration (Factory Pattern)**:
    * If a marker ID is detected for the first time, `card_engine.create_physical_card_instance(mid)` is called. This factory function looks up the ID in `cards.json`, executes the `init_script`, and returns a Python object (e.g., `ImageCard`).
    * The object is registered in `CardSystem` with `is_virtual=False`.
* **State Update**: Existing cards have their positions updated to the new `(cx, cy)` every frame.
* **Decay (Persistence)**:
    * **TTL Hysteresis**: To prevent cards from "flickering" due to temporary occlusion (e.g., a hand passing over a marker), the system uses a Time-To-Live (TTL) counter.
    * **Garbage Collection**: `decrease_ttl(card)` is called on unseen cards. Only when TTL reaches 0 are they permanently removed via `unregister`.

#### 3. Simulation Loop (`main.py` -> `card_engine.py`)
* **Context Isolation**: The system temporarily swaps the global drawing surface to a `dummy_canvas` (an empty array). This separates **Logic** from **Rendering**, ensuring that calculations (which may spawn new virtual cards) occur before the final frame is drawn.
* **Propagation Chain**:
    * The loop runs `MAX_CHAIN_DEPTH` times to resolve multi-hop dependencies (e.g., Card A feeds Card B, which feeds Card C).
    * **Resolve**: Each card queries the `CardSystem` to find neighbors (e.g., "Is there a card to my LEFT?") using spatial hit-testing.
    * **Execute**: The system runs the injected `run_logic()` scripts (loaded from `common.py`).
    * **Spawn**: If logic generates an output (e.g., a "Filter" card producing a filtered image), a **Virtual Card** is spawned and immediately added to `active_cards` for processing in the same frame.

#### 4. Rendering (`card_engine.py`)
* **Canvas Switch**: `CardSystem.canvas` is switched back to the actual `projector_canvas`.
* **Final Pass**: The system iterates through all active cards one last time to generate the visual output.
* **Draw**:
    * Visual content (images, matrices, text) is blitted onto the canvas.
    * **Connection Lines**: Green (valid connection) or Red (invalid connection) lines are drawn between dependent cards to visualize the logic flow.

#### 5. Output (`main.py`)
* **Fullscreen Display**: `cv2.imshow` displays the final `projector_canvas`. The window is forced to fullscreen using `cv2.WINDOW_FULLSCREEN`.
* **Debug Overlay**:
    * If `DEBUG_MODE` is on, the system uses `M_inv` to project the digital boundaries of cards back onto the raw webcam feed (`cv2.perspectiveTransform`).
    * This verifies that the digital projection aligns perfectly with the physical paper cards.

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

## 3. User Interaction
[Play With Cards](https://drive.google.com/file/d/1MT6r4tiNtQnSdND5HIT2RgMhW1Fx84QT/view?usp=sharing)

#### Kernel Addition
![kernel_addition](https://github.com/user-attachments/assets/5380c199-ba7f-415a-a7a9-48a701e31323)
![kernel_addition_demo](https://github.com/user-attachments/assets/be5b1f4f-d4f4-4f55-b5c4-ded8aa58dc14)
Applying Horizontal and Vertical Grad Separately
![horizontal_and_vertical](https://github.com/user-attachments/assets/0c4227f6-e87e-40b2-89c5-bb07a696a46f)

#### Blur Kernel
![blur_kernel](https://github.com/user-attachments/assets/ea52deba-12d8-4df7-85d7-e6408ff19172)


## 4. Collaboration
This project was a highly collaborative effort between Junxiong Chen and Chiahsuan Chang. Chiahsuan proposed the idea of building an application using camera detection and projection. Junxiong introduced the core image convolution concept and developed the card calculation prototype. The final system integration and documentation were completed jointly.
