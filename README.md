

> **Deep Learning-Based U²-Net ID Card Background Remover**

---

# 🆔 ID Card Background Remover using U²-Net

This project implements an **AI-based ID card background removal system** using **deep learning (U²-Net)** and computer vision techniques. It automatically extracts the ID card from any image while removing complex backgrounds, making it suitable for **eKYC, digital onboarding, and document verification workflows**.

---

## ✨ **Features**

* Uses **U²-Net deep learning architecture** for accurate foreground-background segmentation.
* Removes the background from ID card images while preserving **key details** such as photo, text, and barcode.
* Supports **image upload and webcam capture** through an integrated web interface.
* Deployed via a **Flask API**, enabling real-time processing.

---

## 🔧 **Technologies Used**

* **Python**
* **PyTorch** – model implementation and training
* **OpenCV & Pillow** – image processing
* **Flask** – API development
* **HTML, CSS, JavaScript** – frontend integration
* **Label Studio** – dataset annotation and mask creation

---

## 🚀 **Project Workflow**

1. **Problem Definition:** Automate ID card background removal for real-time verification systems.
2. **Approach Exploration:**

   * Edge Detection (Canny)
   * Face Detection with Ratio Cropping
   * **Deep Learning-based Segmentation (U²-Net)**
3. **Dataset Preparation:**

   * Used **Iranian National ID Card Dataset** from Kaggle.
   * Annotated images using **Label Studio**.
   * Performed **data augmentation** to expand the dataset to **1500+ image-mask pairs**.
4. **Model Training:**

   * Trained U²-Net for **25 epochs** with **Binary Cross Entropy loss**.
   * Achieved **high accuracy and generalisation**.
5. **Evaluation:**

   * **F1 Score:** \~0.99
   * **IoU:** \~0.98
6. **Deployment:**

   * Built a **Flask API** to serve the trained model.
   * Developed a **web interface** for users to upload or capture images and download the extracted ID card.

---

## 📊 **Evaluation Results**

| **Metric** | **Value** |
| ---------- | --------- |
| Precision  | 0.9920    |
| Recall     | 0.9927    |
| F1 Score   | 0.9923    |
| IoU        | 0.9848    |

---

## 💡 **How to Run the Project**

1. **Clone this repository:**

   ```bash
   git clone https://github.com/yourusername/id-card-background-remover.git
   cd id-card-background-remover
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app:**

   ```bash
   python app.py
   ```

4. **Access the web interface:**

   * Visit **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.
   * Upload or capture an ID card image to get the **background-removed output**.

---
