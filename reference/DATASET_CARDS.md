# Surveillance & Action Video Benchmark Datasets: Comprehensive Data Cards

This repository and data directory contains four core video benchmark datasets used in computer vision research for surveillance analytics, action recognition, anomaly detection, and violence recognition.

---

## 1. Quick Reference & Comparative Matrix

| Dataset Folder | Benchmark Identity | Citation / Source | Media Format | File Count (on disk) | Disk Size | Target Tasks | State on Disk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Anomaly-Detection-Dataset`** | **UCF-Crime** | Sultani, Chen, Shah (CVPR 2018) | MP4 (H.264), TXT, ZIP | 12 archive/split files + 487 extracted clips | 98.82 GB total (~26 GB extracted) | Weakly supervised anomaly detection, anomaly classification, temporal localization | Partially extracted (Part 1 fully, Parts 2–4 & Normal partial; zip archives present) |
| **`cross camera data`** | **UCF-ARG** (Aerial, Rooftop, Ground) | UCF CRCV / Chen et al. | AVI | 1,410 video clips | 7.11 GB | Multi-view action recognition, cross-camera domain adaptation, drone surveillance | Fully extracted (nested subfolders: `aerial_clips`, `ground_clips`, `rooftop_clips`) |
| **`real life voilence data`** | **RLVS** (Real Life Violence Situations) | Soliman et al. (ICICIS 2019 / Kaggle) | MP4 (3,902), AVI (98) | 4,000 files (2,000 unique videos duplicated) | 3.63 GB | Binary violence detection, fight recognition, CCTV safety monitoring | Fully extracted; contains 100% duplicate subfolder |
| **`tinyvirat`** | **TinyVIRAT** (v1 & v2) | Vaswani et al. (ICPR 2020 / CVPRW 2021) | MP4 (H.264), JSON, TXT | 39,184 video clips + 11 metadata files | 2.04 GB | Low-resolution action recognition, multi-label activity classification | Fully extracted (both v1 and v2 splits present) |
| *Root Archives* | **UBnormal.zip** | Acsintoae et al. (CVPR 2022) | ZIP | 1 zip archive | 14.94 GB | Open-set synthetic video anomaly detection | Archived zip file (unextracted) |
| *Root Documentation* | **VIRAT 2.0 Intro** | DARPA / Kitware / UMD / UCF | PDF | 1 PDF document | 501.2 KB | Benchmark specifications for full-frame VIRAT 2.0 | Reference document |

---

## 2. Dataset Card: UCF-Crime Anomaly Detection Dataset

### 2.1 Overview & Academic Citation
* **Full Benchmark Name**: UCF-Crime: Real-world Anomaly Detection in Surveillance Videos
* **Authors & Institution**: Waqas Sultani, Chen Chen, Mubarak Shah (Center for Research in Computer Vision, University of Central Florida)
* **Publication**: IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018
* **Dataset Directory**: `d:\capstone dataset\Anomaly-Detection-Dataset`
* **Total Volume on Disk**: 98.82 GB (consisting of 9 zip archives, 3 split/annotation text files, and 5 extracted subdirectories).

### 2.2 Domain & Problem Formulation
UCF-Crime is a large-scale real-world surveillance video benchmark designed to bypass the unrealistic nature of prior synthetic or staged anomaly datasets (such as UCSD Ped1/Ped2 or Avenue). The videos are captured from real-world, unconstrained CCTV security cameras in diverse illumination, weather conditions, viewing angles, and image degradation.
* **Core Paradigm**: Weakly Supervised Video Anomaly Detection via Multiple Instance Learning (MIL). Video-level labels (anomaly present vs normal) are available for training, while test videos feature precise temporal frame boundaries.

### 2.3 Directory Structure & Extraction Status
The folder currently contains the full official distribution archives alongside partially extracted parts:
```text
Anomaly-Detection-Dataset/
├── Anomaly-Videos-Part-1.zip           (5.84 GB, 200 videos: Abuse, Arrest, Arson, Assault)
├── Anomaly-Videos-Part-2.zip           (6.23 GB, 204 videos: Burglary, Explosion, Fighting)
├── Anomaly-Videos-Part-3.zip           (5.23 GB, 354 videos: Robbery, RoadAccidents, Shooting)
├── Anomaly-Videos-Part-4.zip           (6.11 GB, 204 videos: Shoplifting, Stealing, Vandalism)
├── Normal_Videos_for_Event_Recognition.zip (1.00 GB, 50 normal event videos)
├── Testing_Normal_Videos.zip           (4.34 GB, normal test evaluation videos)
├── Training-Normal-Videos-Part-1.zip   (35.46 GB, long normal surveillance training videos)
├── Training-Normal-Videos-Part-2.zip   (31.68 GB, long normal surveillance training videos)
├── UCF_Crimes-Train-Test-Split.zip     (12 KB, split indices)
├── Anomaly_Train.txt                   (1,610 training video paths & class labels)
├── Temporal_Anomaly_Annotation_for_Testing_Videos.txt (290 test anomaly temporal annotations)
├── ReadMe-Anomaly-Detection.txt        (Official dataset readme)
│
├── [Extracted Folders on Disk]
│   ├── Anomaly-Videos-Part-1/          (200 videos: Abuse [50], Arrest [50], Arson [50], Assault [50])
│   ├── Anomaly-Videos-Part-2/          (74 videos: Burglary [74])
│   ├── Anomaly-Videos-Part-3/          (93 videos: Robbery [93])
│   ├── Anomaly-Videos-Part-4/          (70 videos: Stealing [20], Vandalism [50])
│   └── Normal_Videos_for_Event_Recognition/ (50 normal video clips)
```

### 2.4 Class Breakdown & Dataset Statistics
UCF-Crime covers **13 anomalous event classes** plus **Normal** surveillance activities:

| Anomaly Category | Training Count (`Anomaly_Train.txt`) | Extracted on Disk | Archive Source |
| :--- | :--- | :--- | :--- |
| **Abuse** | 48 | 50 | `Anomaly-Videos-Part-1.zip` |
| **Arrest** | 45 | 50 | `Anomaly-Videos-Part-1.zip` |
| **Arson** | 41 | 50 | `Anomaly-Videos-Part-1.zip` |
| **Assault** | 47 | 50 | `Anomaly-Videos-Part-1.zip` |
| **Burglary** | 87 | 74 | `Anomaly-Videos-Part-2.zip` |
| **Explosion** | 29 | Pending extraction | `Anomaly-Videos-Part-2.zip` |
| **Fighting** | 45 | Pending extraction | `Anomaly-Videos-Part-2.zip` |
| **RoadAccidents** | 127 | Pending extraction | `Anomaly-Videos-Part-3.zip` |
| **Robbery** | 145 | 93 | `Anomaly-Videos-Part-3.zip` |
| **Shooting** | 27 | Pending extraction | `Anomaly-Videos-Part-3.zip` |
| **Shoplifting** | 29 | Pending extraction | `Anomaly-Videos-Part-4.zip` |
| **Stealing** | 95 | 20 | `Anomaly-Videos-Part-4.zip` |
| **Vandalism** | 45 | 50 | `Anomaly-Videos-Part-4.zip` |
| **Training Normal Videos** | 800 | Archived in zips | `Training-Normal-Videos-Part-1/2.zip` |
| **Normal Event Recognition** | N/A | 50 | `Normal_Videos_for_Event_Recognition.zip` |
| **Testing Anomaly Videos** | 290 | In respective parts | Parts 1–4 |
| **Testing Normal Videos** | 150 | Archived in zip | `Testing_Normal_Videos.zip` |

### 2.5 Annotations & Ground Truth Format
* **Temporal Test Ground Truth (`Temporal_Anomaly_Annotation_for_Testing_Videos.txt`)**:
  Contains 290 anomalous test video annotations:
  ```text
  Video_Name           Anomaly_Class  Start_Frame_1  End_Frame_1  Start_Frame_2  End_Frame_2
  Abuse028_x264.mp4    Abuse          165            240          -1             -1
  Arrest001_x264.mp4   Arrest         1185           1485         -1             -1
  ```
  - Standard frame rate is fixed at **30 FPS**.
  - Negative values (`-1 -1`) denote that no second anomalous instance occurs in that video.
* **Evaluation Metrics**: Frame-level ROC curve and Area Under Curve (AUC).

---

## 3. Dataset Card: UCF-ARG Multi-View Action Dataset

### 3.1 Overview & Academic Citation
* **Full Benchmark Name**: UCF Aerial, Rooftop, Ground (UCF-ARG) Dataset
* **Authors & Institution**: Center for Research in Computer Vision (CRCV), University of Central Florida
* **Dataset Directory**: `d:\capstone dataset\cross camera data`
* **Total Volume on Disk**: 7.11 GB
* **Total Video Clips**: 1,410 video clips (all `.avi` format)

### 3.2 Domain & Sensor Perspective
UCF-ARG is a unique multi-camera synchronous action recognition dataset recorded from three drastically different elevation angles:
1. **Aerial View (`aerial_clips`)**: Recorded from a high-altitude camera mounted on an airborne balloon or UAV platform, characterized by strong camera motion, significant viewpoint distortion, and smaller human figure scales.
2. **Rooftop View (`rooftop_clips`)**: Recorded from a fixed camera situated on the rooftop of a multi-story building overlooking an open field.
3. **Ground View (`ground_clips`)**: Recorded from ground-level surveillance tripod cameras (eye-level perspective).

### 3.3 Class Breakdown & Clip Distribution

All 1,410 clips represent 10 distinct individual human actions performed by multiple human subjects:

| Action Class | Aerial Clips | Ground Clips | Rooftop Clips | Total Video Clips |
| :--- | :---: | :---: | :---: | :---: |
| **boxing** | 48 | 48 | 48 | **144** |
| **carrying** | 48 | 48 | 48 | **144** |
| **clapping** | 48 | 48 | 48 | **144** |
| **digging** | 48 | 48 | 48 | **144** |
| **jogging** | 48 | 48 | 48 | **144** |
| **openclosetrunk** | 39 | 39 | 36 | **114** |
| **running** | 48 | 48 | 48 | **144** |
| **throwing** | 48 | 48 | 48 | **144** |
| **walking** | 48 | 48 | 48 | **144** |
| **waving** | 48 | 48 | 48 | **144** |
| **TOTAL** | **471** | **471** | **468** | **1,410** |

### 3.4 File Naming Convention & Directory Structure
* Directory nesting on disk:
  - `cross camera data/aerial_clips/aerial_clips/<class>/<file_name>.avi`
  - `cross camera data/ground_clips/ground_clips/<class>/<file_name>.avi`
  - `cross camera data/rooftop_clips/rooftop_clips/<class>/<file_name>.avi`
* Filename naming pattern:
  `person<ActorID>_<RunID>_<Viewpoint>_<Action>.avi`
  - Example: `person01_01_aerial_boxing.avi`, `person01_01_ground_boxing.avi`, `person01_01_rooftop_boxing.avi`
* Typical duration: ~6 to 10 seconds per clip (~200 to 300 frames at ~30 fps).

---

## 4. Dataset Card: Real Life Violence Situations Dataset (RLVS)

### 4.1 Overview & Academic Citation
* **Full Benchmark Name**: Real Life Violence Situations Dataset (RLVS)
* **Authors & Publication**: M. Soliman, M. Kamal, M. Nashed, Y. Mostafa, B. Chawky, D. Khattab, *"Violence Recognition from Videos using Deep Learning Techniques"*, Proceedings of the 9th International Conference on Intelligent Computing and Information Systems (ICICIS), 2019.
* **Dataset Directory**: `d:\capstone dataset\real life voilence data`
* **Total Volume on Disk**: 3.63 GB (4,000 files total, 2,000 unique videos)

### 4.2 Storage Redundancy & Directory Structure Notice
The dataset directory contains an identical duplicated extraction inside:
* Folder 1: `d:\capstone dataset\real life voilence data\Real Life Violence Dataset` (2,000 videos, 1.81 GB)
* Folder 2: `d:\capstone dataset\real life voilence data\real life violence situations\Real Life Violence Dataset` (2,000 videos, 1.81 GB)
* **Finding**: Both directories contain identical files with 0 hash/name difference. One copy can be safely archived or deleted to reclaim **1.81 GB** of disk space.

### 4.3 Class Breakdown & Video Distribution

| Class | Video Count | File Extension Breakdown | Description |
| :--- | :---: | :--- | :--- |
| **Violence** | 1,000 | 1,000 `.mp4` (`V_1.mp4` to `V_1000.mp4`) | Real-world physical street fights, altercations, mobs, and aggressive assaults collected from YouTube and CCTV streams. |
| **NonViolence** | 1,000 | **951 `.mp4` + 49 `.avi`** (`NV_1` to `NV_1000`) | Normal day-to-day human interactions: walking, sports, casual talking, jogging, dancing, and benign public assembly. |
| **TOTAL** | **2,000** | **3,902 MP4 / 98 AVI across both copies** | Balanced binary classification dataset. |

### 4.4 Critical Data Loader Gotcha
> [!WARNING]
> In the `NonViolence` class, exactly **49 videos are saved in `.avi` container format** (specifically `NV_602.avi` and `NV_863.avi` through `NV_911.avi`), whereas the remaining 951 are `.mp4`.
> If your PyTorch / OpenCV Dataset script uses `glob('*.mp4')`, it will silently drop 49 negative samples, causing training imbalance and evaluation data leakage. Ensure your file pattern matches both `*.mp4` and `*.avi`.

---

## 5. Dataset Card: TinyVIRAT & TinyVIRAT-v2

### 5.1 Overview & Academic Citation
* **Full Benchmark Name**: TinyVIRAT: Low-resolution Video Action Recognition & CVPR 2021 ActivityNet TinyAction Challenge
* **Authors & Institutions**: Gaurav Vaswani, Min-hsuan Sung, Rohit Babbar, Mubarak Shah et al. (CRCV UCF, Kitware)
* **Publications**:
  - ICPR 2020: *"TinyVIRAT: Low-resolution Video Action Recognition"*
  - CVPR 2021 ActivityNet Workshop: *"TinyAction Challenge: Recognizing Real-world Low-resolution Activities in Videos"*
* **Dataset Directory**: `d:\capstone dataset\tinyvirat`
* **Total Volume on Disk**: 2.04 GB
* **Total Video Clips**: 39,184 video clips in `.mp4` format + 11 metadata JSON/TXT files

### 5.2 Domain & Problem Formulation
In real-world wide-area surveillance footage (e.g., parking lots, facilities, school grounds), humans and vehicles occupy a minute fraction of the visual field (often fewer than 70×70 pixels). Standard benchmarks (Kinetics, UCF101, Sports-1M) feature dominant, centered, high-resolution subjects. TinyVIRAT was created to evaluate deep models specifically on **naturally occurring low-resolution, small-scale action tubes** cropped from the VIRAT 2.0 surveillance video repository.

### 5.3 Version Comparison (v1 vs v2)

The directory contains both versions of the benchmark:

```text
tinyvirat/
├── TinyVIRAT/                 (TinyVIRAT Version 1: 12,829 videos)
│   └── TinyVIRAT/
│       ├── videos/train/      (7,663 clips across VIRAT parent video folders)
│       ├── videos/test/       (5,166 clips across VIRAT parent video folders)
│       ├── tiny_train.json    (7,663 training tubes with label, path, dimensions)
│       └── tiny_test.json     (5,166 testing tubes)
│
└── TinyVIRAT-v2/              (TinyVIRAT Version 2 - CVPR TinyAction Challenge: 26,355 videos)
    └── TinyVIRAT_V2/
        ├── videos/train/      (16,950 clips across parent surveillance sequences)
        ├── videos/val/        (3,308 clips across parent surveillance sequences)
        ├── videos/test/       (6,097 clips directly stored as 00000.mp4 - 06096.mp4)
        ├── class_map.json     (26 action class index mappings)
        ├── tiny_train_v2.json (16,950 multi-label action tube entries)
        ├── tiny_val_v2.json   (3,308 multi-label action tube entries)
        ├── tiny_test_v2_public.json (6,097 test benchmark clip records)
        └── changelog.txt      (Official split rebalancing notes)
```

### 5.4 Class Taxonomy (TinyVIRAT-v2: 26 Classes)
TinyVIRAT-v2 is framed as a **multi-label action recognition** problem where multiple labels can co-occur in the same action tube (e.g. `['activity_carrying', 'activity_walking']`):

```json
{
  "Opening": 0,
  "Interacts": 1,
  "Pull": 2,
  "activity_carrying": 3,
  "Entering": 4,
  "vehicle_moving": 5,
  "Exiting": 6,
  "Loading": 7,
  "Talking": 8,
  "activity_running": 9,
  "vehicle_turning_left": 10,
  "vehicle_stopping": 11,
  "Riding": 12,
  "Closing": 13,
  "activity_walking": 14,
  "Push": 15,
  "specialized_using_tool": 16,
  "vehicle_starting": 17,
  "specialized_miscellaneous": 18,
  "activity_standing": 19,
  "Transport_HeavyCarry": 20,
  "activity_gesturing": 21,
  "vehicle_turning_right": 22,
  "specialized_talking_phone": 23,
  "specialized_texting_phone": 24,
  "Misc": 25
}
```

### 5.5 Annotation Schema
Sample entry from `tiny_train_v2.json`:
```json
{
  "id": "000000",
  "video_id": "VIRAT_S_000204_04_000738_000977",
  "path": "VIRAT_S_000204_04_000738_000977/000000.mp4",
  "dim": [82, 72, 72],
  "label": ["activity_carrying", "activity_walking"]
}
```
* `dim`: Array `[num_frames, frame_height, frame_width]`. Notice the native cropped spatial dimension is ~72×72 pixels across 82 frames.
* `label`: String list of concurrent activities.

---

## 6. Root Level Auxiliary Benchmarks & Documents

### 6.1 UBnormal (`UBnormal.zip`)
* **File Size**: 14.94 GB (16,037,804,331 bytes)
* **Citation**: Andra Acsintoae, Andrei Florescu, Mariana-Iuliana Georgescu, Tudor Mare, Paul Sumedrea, Radu Tudor Ionescu, Fahad Shahbaz Khan, Mubarak Shah, *"UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection"*, CVPR 2022.
* **Nature**: Synthetic open-set surveillance dataset generated in Cinema4D featuring photo-realistic 3D human avatars and 22 anomaly classes across 29 virtual scenes. Provides pixel-level annotations and frame-level ground truth.

### 6.2 VIRAT 2.0 Introduction (`VIRAT_Video_Dataset_Release2.0_Introduction_v1.0.pdf`)
* **File Size**: 501.2 KB
* **Significance**: Official release manual describing the full-frame high-definition VIRAT 2.0 Ground Dataset collected under DARPA VIRAT. Explains camera calibration, scene geometry, bounding box formats, and action event trigger specifications that underpin TinyVIRAT.

---

## 7. Recommended Workflow & Data Hygiene Checklist

1. **UCF-Crime Extraction**:
   - Parts 2, 3, and 4 currently have remaining categories in their `.zip` archives (`Explosion`, `Fighting`, `RoadAccidents`, `Shooting`, `Shoplifting`). Unzip them into their respective folders before launching full 13-class multi-event training.
2. **RLVS Deduplication**:
   - Delete `real life voilence data/real life violence situations` to recover **1.81 GB** without any loss of training data.
3. **RLVS Data Loading**:
   - Ensure the video loader accepts `['*.mp4', '*.avi']` to prevent dropping the 49 AVI files in the `NonViolence` class.
4. **TinyVIRAT Model Input**:
   - Due to the small spatial resolutions (70×70 to 90×90), avoid standard ImageNet heavy spatial downsampling. Use spatial super-resolution branches or spatio-temporal attention architectures (e.g. Video Swin Transformer, TimeSformer, or 3D CNNs with low pooling strides).
