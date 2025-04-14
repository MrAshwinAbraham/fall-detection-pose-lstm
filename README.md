# Fall Detection using Pose Estimation and LSTM  
Real-time fall detection system using MediaPipe pose tracking and deep learning models (LSTM, GRU, BiLSTM) optimized for edge deployment and privacy.

---

## 📌 Overview  
This project implements a lightweight and privacy-preserving fall detection system leveraging pose keypoints extracted from videos via MediaPipe and classifying them using sequential deep learning models. The primary application is real-time surveillance in homes, elderly care environments, or industrial settings.

---

## 🎯 Key Features  
- Uses 33 keypoints from MediaPipe Pose per frame (x, y, z, visibility)  
- Sequence length of 60 frames per sample  
- Supports real-time inference from webcam or video files  
- Compares performance of **LSTM**, **GRU**, and **BiLSTM** models  
- Provides fast inference with < 100ms latency  
- Privacy-preserving: no RGB images or full video storage

---

## 🧠 Model Comparison Summary

| Model   | Accuracy     | F1 Score    | Inference Time |
|---------|--------------|-------------|----------------|
| LSTM    | ✅ Best       | ✅ Best      | ⚡ Fast         |
| GRU     | Good         | Moderate     | Fastest        |
| BiLSTM  | Highest Recall | Moderate   | ❗ Slowest     |

LSTM achieved the best balance of accuracy, precision, recall, and speed for deployment in real-time scenarios.

---

## 🗂️ Project Structure

```
fall-detection-pose-lstm/
├── models/               # Trained models (LSTM, GRU, BiLSTM .h5 files)
├── videos/               # Sample videos for testing/inference (has to be added)
├── venv/                 # Virtual environment
├── main.py               # Main script for real-time/video-based fall detection
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Installation  

1. **Clone the repo:**
```bash
git clone https://github.com/MrAshwinAbraham/fall-detection-pose-lstm.git
cd fall-detection-pose-lstm
```
2. **Create the vitrual Environment:**
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
```
3. **Install requirements:**
```bash
pip install -r requirements.txt
```
## ▶️ Running the Project  
```bash
python main.py
```
Press "w" to run on webcam, "v" to test on sample videos giving video paths.

