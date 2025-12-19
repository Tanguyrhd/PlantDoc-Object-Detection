# Detailed Project Analysis - PlantDoc Object Detection

## 1. Project Overview

### The Agricultural Problem

Plant diseases cause significant crop losses worldwide:

- In India, **35% of annual crop yield** is lost due to plant diseases
- Early detection often requires **laboratory infrastructure** and **plant pathology expertise** that isn't widely available
- Manual diagnosis is **slow and expensive**

### Main Objective

Develop a complete solution for **automatic detection and classification of plant diseases** using computer vision and deep learning. The system can identify not only whether a plant is diseased, but also the plant species and the specific disease it's suffering from.

### Proposed Solution

A **3-level classification architecture** based on YOLO from ultralytics:

1. **Species Identification**: Recognition of 13 plant species
2. **Binary Classification**: Quick detection (healthy vs diseased)
3. **Disease Diagnosis**: Precise classification among 9 diseases

All deployed via an **asynchronous REST API** containerized with Docker and hosted on Google Cloud Run.

### Technologies Used

- **Deep Learning**: YOLOv8 (Ultralytics) for object detection and classification
- **Backend**: Python 3.10, FastAPI, asyncio
- **Image Processing**: OpenCV, Pillow
- **Deployment**: Docker, Google Cloud Run
- **Data Engineering**: Pandas, NumPy for preprocessing

---

project overview
Go directly to the final architecture solution (link to streamlit)
PResent the result
Explained three concept of the solutions
  Abstract class and 3 pipelines
  threadpool executor in streamlit and asyncio
  lazy vs global loading
Explained three chalenges and solutions :
  corrupted data - fixed with function
  disease :
    model not precise enough for disease but still good for some classes
    get better data, finetuned more
    too much bad disease (too global, same but right different)
  Docker image too big :
    slim python
    CPU only pytorch
    .gitignore
Improvments short medium long term
Conclusion
