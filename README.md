## Project Workflow

This project generates **virtual IMU data** for Human Activity Recognition (HAR) from natural-language activity prompts.

### 1. Activity & Prompt Input
- **Activity Selection:** Choose an activity (e.g., *Jumping*).
- **Prompt Description:** Provide a short description (≤15 words, 1 person only, no environment details).

### 2. Text Generation with ChatGPT
- ChatGPT converts the activity and prompt into a descriptive sentence.  
  *Example:* “A man jumps over a small gap in the road, safely landing.”

### 3. Motion Synthesis (T2M-GPT)
- **CLIP Encoder:** Embeds the text description.
- **Transformer Layers:** Map the embedding to a **Codebook** of motion tokens (`e1 … eK`).
- **Dequantization + Decoder:** Reconstruct continuous 3D human motion sequences.

### 4. Kinematic Conversion
- **Inverse Kinematics:** Convert generated motion to joint positions.
- **Joint Rotation & Translation:** Extract 3D joint angles and translations.

### 5. IMU Simulation
- **IMUSim:** Generate virtual inertial sensor signals from joint motions.
- **Calibration:** Adjust the virtual IMU data to match real sensor characteristics.

### USAGE: Model Training & Deployment
- **Classifier Training:** Train a human-activity recognition model using the virtual IMU data.
- **HAR Deployment:** Deploy the trained model on real wearable devices.

---

## How to Run the Project

This repository implements a **three–stage pipeline** to generate virtual IMU sensor data from natural-language activity prompts.
1. **Text → 3D Motion** (`B1_text_to_motion.py`)
2. **3D Motion → Quaternion Representation** (`B2_quat_generation.py`)
3. **Quaternion → IMU Simulation** (`B3_imu_simulate.py`)

---

## Environment Setup
- **Python:** 3.8+
- **Recommended GPU:** CUDA-enabled NVIDIA GPU
- Create and activate a conda environment from the provided YAML file:
```bash
conda env create -f environment_new.yml
conda activate wearable_har
```

## Download the pretrained models

```bash
bash prepare/download_model.sh
bash prepare/download_extractor.sh
```
