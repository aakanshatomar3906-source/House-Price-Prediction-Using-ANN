[project_readme.md](https://github.com/user-attachments/files/35445537/project_readme.md)

# 🏡 NestWorth AI — California House Price Predictor (ANN)

An end-to-end Deep Learning web application that predicts median house values across California districts using an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras** and deployed via **Streamlit**.

---

## ✨ Features

- **🎯 Single Property Valuation:** Interactive sliders and numeric inputs for location, building age, room counts, neighborhood demographics, and median income.
- **⚡ Quick-Fill Presets:** One-click test scenarios (*Bay Area Luxury*, *LA Suburb Family*, and *Inland Budget*) to explore predictions instantly.
- **🗺️ Interactive Location Pinpoint:** Live map preview plotting the selected latitude and longitude coordinates.
- **💱 Dual Currency Display:** Instant valuation in both **USD ($)** and **INR (₹ Lakhs)** with customizable exchange rates.
- **📂 Batch CSV Prediction:** Upload any housing dataset (or test directly with the included `Dataset/housing.csv`) to generate and download bulk price predictions.

---

## 🏗️ Project Structure

```text
House prediction using ANN/
├── Dataset/
│   └── housing.csv                      # California Housing dataset (20,640 records)
├── models/
│   ├── model_ann.h5                     # Trained 4-layer Keras ANN weights
│   └── min_max_scaler.pkl               # Fitted Scikit-Learn MinMaxScaler
├── notebooks/
│   └── House_price_pred_using_ann.ipynb # Data preprocessing, training & evaluation notebook
├── app.py                               # Streamlit web application
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # Project documentation
```

---

## 🧠 Model Architecture & Workflow

1. **Data Cleaning & Preprocessing:**
   - Removed missing values (`total_bedrooms`), resulting in 20,433 clean records.
   - Encoded categorical `ocean_proximity` features (`<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`).
   - Normalized numerical features using `MinMaxScaler`.
2. **Neural Network Design:**
   - **Input Layer:** 9 features
   - **Hidden Layer 1:** 64 neurons (`ReLU`)
   - **Hidden Layer 2:** 32 neurons (`ReLU`)
   - **Hidden Layer 3:** 16 neurons (`ReLU`)
   - **Output Layer:** 1 neuron (`Linear`)
   - **Optimizer & Loss:** `Adam (lr=0.001)` with Mean Squared Error (`MSE`) loss, Mean Absolute Error (`MAE`) metric, and `EarlyStopping`.

---

## 🚀 Getting Started (Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/California-House-Price-ANN.git
cd California-House-Price-ANN
```

### 2. Create & Activate a Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack

- **Frontend / UI:** Streamlit, Custom HTML/CSS
- **Deep Learning:** TensorFlow, Keras
- **Data Processing:** Pandas, NumPy, Scikit-Learn
