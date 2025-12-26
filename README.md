# Building My Own Text Classifier  
### An end-to-end NLP pipeline with model comparison and evaluation

## Overview

This project demonstrates a complete **classical NLP workflow** for text classification, from raw text data to model evaluation.  
Instead of focusing on high accuracy, the goal of this project is to understand **how different machine learning models behave on the same features**, and why model performance is often limited by **data quality and feature engineering** rather than the algorithm itself.

The project compares three commonly used linear models for text classification:

- Logistic Regression  
- Multinomial Naive Bayes  
- Linear Support Vector Machine (Linear SVM via SGD)

This approach reflects real-world machine learning workflows, where experimentation, evaluation, and iteration are more important than a single metric.

---

## Dataset

The dataset is a small, manually created collection of short sentences labeled as **positive** or **negative** sentiment.

- 20 total samples  
- 10 positive  
- 10 negative  

The dataset is intentionally small to:
- Highlight the impact of limited data
- Emphasize evaluation and analysis over raw accuracy
- Demonstrate the full pipeline without relying on large external datasets

---

## Text Processing

The text data is converted into numerical features using a **Bag-of-Words** representation.

- `CountVectorizer` from scikit-learn
- Each sentence is transformed into a vector of word counts
- Vocabulary is built from the entire dataset

This representation is simple and interpretable, making it ideal for understanding classical NLP models.

---

## Train / Test Split

The dataset is split into training and testing sets:

- 70% training data  
- 30% testing data  
- Fixed random seed for reproducibility  

This ensures consistent results across multiple runs and models.

---

## Models

### 1. Logistic Regression

Logistic Regression learns a linear decision boundary by optimizing class probabilities.

- Fast to train
- Easy to interpret
- Common baseline model for text classification

Evaluation includes:
- Accuracy
- Precision, recall, and F1-score using `classification_report`

---

### 2. Multinomial Naive Bayes

Multinomial Naive Bayes is a probabilistic model commonly used for text data.

- Uses word frequency statistics
- Assumes independence between words
- Performs well on small datasets with sparse features

This model provides a strong baseline and is frequently used in spam detection and sentiment analysis.

---

### 3. Linear Support Vector Machine (Linear SVM)

A Linear SVM attempts to find the decision boundary that **maximizes the margin** between classes.

- Trained using `SGDClassifier`
- Well-suited for high-dimensional sparse data such as text
- Focuses on support vectors near the decision boundary

Linear SVMs are a strong choice for classical NLP tasks and often outperform other linear models when paired with good feature engineering.

---

## Results

| Model | Accuracy |
|-----|----------|
| Logistic Regression | 0.50 |
| Naive Bayes | 0.33 |
| Linear SVM | 0.50 |

Due to the small dataset size, performance is limited and varies between models.  
This outcome is expected and highlights an important lesson in machine learning:

> **Model performance is often constrained by data quality and feature representation, not the algorithm itself.**

---

## Key Learnings

- Different algorithms can produce similar performance when features are weak
- Low accuracy is not always a failure — it is often a signal to revisit data and preprocessing
- Feature engineering has a larger impact on performance than switching models
- Machine learning is an iterative process, not a one-shot solution

---

## Possible Improvements

Future iterations of this project could include:

- Using **TF-IDF** instead of Bag-of-Words
- Adding **n-grams** to capture word order
- Expanding the dataset with more samples
- Handling negation explicitly (e.g., "not good")
- Hyperparameter tuning for each model

---

## Technologies Used

- Python  
- pandas  
- scikit-learn  
- CountVectorizer  
- Logistic Regression  
- Multinomial Naive Bayes  
- Linear SVM (SGDClassifier)  

---

## Conclusion

This project serves as a practical introduction to classical NLP techniques and model evaluation.  
It emphasizes understanding **why models behave the way they do**, rather than simply chasing higher accuracy.

The same workflow can be extended to larger datasets and more advanced techniques such as TF-IDF, word embeddings, and deep learning models.

---
