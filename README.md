# 🍽️ Restaurant Recommendation System

A **content-based restaurant recommendation web application** built using **Python, Flask, and Machine Learning**.
The system recommends restaurants based on **similar reviews and cuisines** using **TF-IDF vectorization and Cosine Similarity**.

---

## 🚀 How It Works

1. Restaurant reviews and cuisines are combined as text features.
2. Text is converted into numerical vectors using **TF-IDF**.
3. **Cosine Similarity** compares restaurant vectors.
4. The system returns **top 10 similar restaurants** based on the selected restaurant.

---

## 🛠️ Technologies Used

* Python
* Flask
* Pandas
* NumPy
* Scikit-Learn
* HTML / CSS / Jinja2

---

## ⚙️ Installation

Install dependencies using the requirements file:

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
python app.py
```

---

## 🌐 Open in Browser

```
http://127.0.0.1:5000
```

---

## 📂 Project Structure

```
restaurant-recommendation-system
│
├── app.py
├── requirements.txt
├── restaurant1.csv
│
├── templates
│   ├── index.html
│   ├── layout.html
│   ├── extractor.html
│   └── keywords.html
│
└── static
    ├── css
    │   └── style.css
    └── images
        └── food_plate.png
```

---

## 👩‍💻 Author

Developed by **Shrawani Vinod Lavate**
