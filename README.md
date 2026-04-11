# 🍽️ CalorieTracker — AI-Powered Food Detection & Nutrition Analysis

> Upload a food photo. Get instant calorie counts, macronutrient breakdowns, volume estimates, and personalized health advice — all powered by YOLOv8 and depth estimation.

---

## 📸 Overview

**CalorieTracker** is an end-to-end AI system that detects food items in an image, estimates their physical volume using depth maps, calculates calorie and macronutrient content, and provides personalized health advice based on your BMI and activity level.

Built with [Streamlit](https://streamlit.io/), [Ultralytics YOLOv8](https://docs.ultralytics.com/), and [MiDaS](https://github.com/isl-org/MiDaS) depth estimation.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Food Detection** | Custom-trained YOLOv8 model detects food items with bounding boxes and segmentation masks |
| 📐 **Volume Estimation** | MiDaS depth model estimates the 3D volume of each detected food item |
| 🔥 **Calorie Calculation** | Volume × density → weight → calories, protein, carbs, fat using a curated food CSV database |
| 🥗 **Macronutrient Breakdown** | Interactive pie charts and tables for per-item and total nutrition |
| 💪 **Health Advice** | BMI calculator + Harris-Benedict equation for personalized daily calorie needs |
| 📊 **Calorie Comparison** | Visual bar chart comparing your meal vs. recommended intake vs. daily needs |

---

## 🛠️ Tech Stack

- **Backend / ML:** Python, PyTorch, Ultralytics YOLOv8, MiDaS, OpenCV
- **Frontend / UI:** Streamlit
- **Data Processing:** Pandas, NumPy, Scikit-image, Matplotlib
- **Computer Vision:** OpenCV, Pillow

---

## 📁 Project Structure

```
CalorieTracker/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── finalcalories.csv           # Food nutrition & density database
│
├── models/
│   ├── image_detection.py      # YOLOv8 food detection (ViewDetector class)
│   ├── volume_estimation.py    # MiDaS depth-based volume estimation
│   ├── calorie_calculation.py  # Nutrition calculator + BMI + health metrics
│   └── health_advice.py        # Personalized health advice engine
│
└── Finaltyolo-master/          # Legacy React client (Nutritionix API version)
    ├── client/                 # React frontend
    └── models/
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- A trained YOLOv8 model file: `final_yolov8.pt`

### 1. Clone the Repository

```bash
git clone https://github.com/PratikHarkare06/CalorieTracker.git
cd CalorieTracker
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the Model File

Download or provide your trained YOLOv8 model and place it in the `models/` directory:

```
models/final_yolov8.pt
```

> 📌 The app will also search in the project root and `~/Downloads/` as fallback locations.

### 5. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧠 How It Works

```
Upload Image
     │
     ▼
YOLOv8 Detection
  ├─ Bounding boxes drawn on detected food items
  └─ Segmentation masks (if model supports it)
     │
     ▼
MiDaS Depth Estimation
  ├─ Generates a depth map of the image
  └─ Depth map used to estimate 3D volume of each food region
     │
     ▼
Volume → Weight → Nutrition
  ├─ Volume (cm³) × Density (g/cm³) = Weight (g)
  └─ Weight × per-gram nutrients = Calories, Protein, Carbs, Fat
     │
     ▼
Health Advice
  ├─ BMI calculation
  ├─ Harris-Benedict equation → Daily calorie needs
  └─ Personalized meal recommendations
```

---

## 📊 Calorie Calculation Method

Nutritional values are derived in three steps:

1. **Volume Estimation** — Each detected food's pixel area is multiplied by its depth value (in cm) to get a volume in cm³.
2. **Weight Calculation** — `Weight (g) = Volume (cm³) × Density (g/cm³)` using values from `finalcalories.csv`.
3. **Nutrition Lookup** — `Calories = Weight × calories_per_g`, and similarly for protein, carbs, fat, fiber, sugars, and sodium.

### Macronutrient Energy Values
| Macronutrient | Calories per gram |
|---|---|
| Protein | 4 kcal/g |
| Carbohydrates | 4 kcal/g |
| Fat | 9 kcal/g |

---

## 🧬 Health Metrics

### BMI Categories

| BMI Range | Category |
|---|---|
| < 18.5 | Underweight |
| 18.5 – 24.9 | Normal weight |
| 25 – 29.9 | Overweight |
| ≥ 30 | Obese |

### Activity Levels (Harris-Benedict)

| Level | Description | Multiplier |
|---|---|---|
| Sedentary | Little or no exercise | 1.2× |
| Light | Exercise 1–3 days/week | 1.375× |
| Moderate | Exercise 3–5 days/week | 1.55× |
| Active | Exercise 6–7 days/week | 1.725× |
| Very Active | Hard exercise + physical job | 1.9× |

---

## 📦 Requirements

```
numpy>=1.18.5
opencv-python>=4.1.2
torch>=1.7.0
torchvision>=0.8.1
Pillow>=7.1.2
PyYAML>=5.3.1
requests>=2.23.0
scipy>=1.4.1
tqdm>=4.41.0
ultralytics>=8.0.0
streamlit>=1.0.0
matplotlib>=3.5.0
scikit-image>=0.18.0
```

---

## ⚠️ Known Limitations

- Volume estimation accuracy depends on the **pixel-to-cm ratio** and **depth scale factor** sliders — calibrate them based on your camera distance.
- The nutrition database (`finalcalories.csv`) covers a limited set of food classes; unrecognized items will use a default density of `1.0 g/cm³`.
- The YOLOv8 model (`final_yolov8.pt`) must be obtained separately and placed in the correct directory.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is for educational and research purposes.

---

<p align="center">Made with ❤️ using YOLOv8 + MiDaS + Streamlit</p>
