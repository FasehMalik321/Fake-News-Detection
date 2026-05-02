# 📰 Fake News Detector

A machine learning project that classifies news articles as **real** or **fake** using Natural Language Processing (NLP) and two classification models — Logistic Regression and Random Forest.

---

## 📌 Overview

This project trains a binary text classifier on a labeled dataset of real and fake news articles. It includes exploratory data analysis, custom text vectorization, model training, accuracy comparison, and a prediction function for classifying new articles.

---

## 📁 Dataset

The project expects two CSV files in the working directory:

| File       | Description                          |
|------------|--------------------------------------|
| `Fake.csv` | Articles labeled as fake news (1)    |
| `True.csv` | Articles labeled as real news (0)    |

Both files must contain a `text` column with the article body.

> **Suggested dataset:** [ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) on Kaggle.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas** — data loading and manipulation
- **numpy** — numerical operations
- **matplotlib** — data visualization
- **scikit-learn** — model training and evaluation (`LogisticRegression`, `RandomForestClassifier`)
- **collections.Counter** — vocabulary building

---

## ⚙️ How It Works

### 1. Data Loading & Labeling
Fake and real news CSVs are loaded, labeled (`1` = Fake, `0` = Real), and merged into a single DataFrame.

### 2. Exploratory Data Analysis (EDA)
Three visualizations are generated:
- **Bar chart** — count of real vs. fake articles
- **Histogram** — text length distribution by label
- **Bar chart** — average article length by label

### 3. Text Preprocessing
A `clean_text()` function lowercases all text and strips punctuation.

### 4. Feature Engineering (Bag of Words)
A custom vocabulary of the **top 3,000 most frequent words** is built. Each article is converted into a word-count vector using `text_to_vector()`.

### 5. Model Training
The dataset is split 80/20 (train/test). Two models are trained:
- **Logistic Regression** (`max_iter=1000`)
- **Random Forest Classifier**

### 6. Evaluation
Accuracy scores are printed and compared via a bar chart.

### 7. Prediction
A `predict_news()` function accepts raw text and returns a label with a confidence score:

```python
predict_news("Breaking: Government announces new policy today")
# → Real News (Confidence: 92.15%)

predict_news("Shocking! Celebrity caught in fake scandal")
# → Fake News (Confidence: 87.43%)
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 3. Add the dataset

Place `Fake.csv` and `True.csv` in the project root directory.

### 4. Run the notebook

```bash
jupyter notebook FakeNewsDetector.ipynb
```

Run all cells top to bottom to train the models and test predictions.

---

## 📊 Results

| Model               | Accuracy     |
|---------------------|--------------|
| Logistic Regression | ~98%         |
| Random Forest       | ~99%         |

> Actual values may vary slightly depending on the dataset used.

---

## 📂 Project Structure

```
fake-news-detector/
│
├── FakeNewsDetector.ipynb   # Main notebook
├── Fake.csv                 # Fake news dataset (not included)
├── True.csv                 # Real news dataset (not included)
└── README.md
```

---

## 🔮 Future Improvements

- Add TF-IDF vectorization for better feature representation
- Experiment with deep learning models (LSTM, BERT)
- Build a simple web interface for live predictions
- Add support for article URL scraping

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
