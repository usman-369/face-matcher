# 🧔‍♂️ Face Matcher for ID Card Verification

The **face_matcher** is a lightweight Python package powered by [DeepFace](https://github.com/serengil/deepface). It compares a person's ID card photo with a selfie to verify if they belong to the same individual. This is useful in identity verification flows like user onboarding or KYC (Know Your Customer) processes.

---

## 🚀 Features

- 🔐 **Offline & Secure** – No external API calls or internet required.
- 🧬 **DeepFace Integration** – Supports multiple face recognition models.
- 📷 **Flexible Input** – Works with image files or in-memory images.
- 📊 **Matching Logs** – Logs confidence score and result (optional).
- 🛠️ **Easy to Integrate** – Perfect for Python backends (like Django or Flask).

---

## ⚙️ How It Works

- Compares two images: one from an ID card, and one selfie.
- Uses OpenCV for image handling and processing.
- Detects and compares faces using deep learning.
- Returns a match status (True or False) and confidence score.
- Logs comparison results (optional).

---

## 📦 Installation

Install required packages:

```bash
pip install deepface opencv-python
```

**Note:** `numpy` will be installed automatically.  
If you use TensorFlow-based models (e.g., VGG-Face, Facenet), `tensorflow` will also be installed by DeepFace as needed.

---

## 🗂️ Project Structure

```bash
.
├── face_matcher
│   ├── constants.py
│   ├── core.py
│   ├── __init__.py
│   ├── logger.py
│   ├── test_face_matcher.py
│   └── utils.py
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🏎️🏁 Quick Start

```python
from face_matcher import FaceMatcher

matcher = FaceMatcher(id_file, selfie_file)
match = matcher.match_faces()
print("Matched!" if match else "No match")
```

---

## 🖼️ Supported Models

You can choose any of the following models:

- ArcFace
- Facenet512
- DeepFace
- Facenet
- VGG-Face
- Dlib
- DeepID
- OpenFace

**To use a specific model:**

```python
matcher = FaceMatcher(id_file, selfie_file, model_name="ArcFace")
```

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Authors

- **Name:** Usman Ghani, Imran Nawaz
- **GitHub:** [usman-369](https://github.com/usman-369), [codewithimran-786](https://github.com/codewithimran-786)

![AI Assisted](https://img.shields.io/badge/Built_with-ChatGPT-8A2BE2?logo=openai&logoColor=white&style=flat-square)