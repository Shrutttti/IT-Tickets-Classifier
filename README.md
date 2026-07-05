# IT Ticket Classifier

A machine learning pipeline that predicts the **assignment group** for IT support tickets. The data is extracted from ITSM records.

---

## Project Overview

This project takes historical IT ticket data, anonymises sensitive information, performs feature engineering, data cleaning and and trains a classification model to automatically route incoming tickets to the correct assignment group. The goal is to reduce manual triage time and improve ticket resolution speed.

---

## Project Structure

```
├── src/
│   ├── anonymization.py            # Anonymises and cleans raw ticket data
│   ├── feature_engineering.py      # Feature engineering and data cleaning
|   ├── preprocess.py               # pipeline to prepare data for ML models
|   ├── train.py                    # train ML model
|   ├── evaluate.py                 # model evaluation
|   ├── predict.py                  # prediction on new data
├── notebooks/
│   ├── IT Tickets_EDA.ipynb                        # Exploratory data analysis
│   ├── IT Tickets_FeatureEngineering.ipynb          # Feature engineering exploration
│   ├── IT Ticktes_ML_Classification Pipeline.ipynb  # Classification experiments
│   ├── IT Ticktes_ML_Clustering Pipeline.ipynb      # Clustering experiments
├──  requirements.txt      
├──  README.md
└── .gitignore

```

---

## Pipeline

```
Raw .xlsx files
      ↓
Data_Preprocessing.py     → tickets_processed.pkl
      ↓
feature_engineering.py    → ticketsdata.csv
      ↓
train_classifier.py       → best_model.pkl
      ↓
evaluate_classifier.py    → evaluation_report.txt, confusion_matrix.png
```

---

## Models

The training pipeline fits and tunes three models via `RandomizedSearchCV`:

- **Logistic Regression** — baseline classifier
- **Random Forest** — ensemble tree model
- **SVM classifier** —  Linear model
- **LightGBM** — gradient boosting model
- **Voting classfier** — combines the best model from previous training

The best hyperparameters from each are combined into a soft Voting Ensemble.

---

## Features Used

- Ticket priority, urgency, impact (ordinal encoded)
- Business service, service offering, incident type (one-hot encoded)
- Short description and description (TF-IDF, bigrams, 2000–5000 features)
- Date features: opened day, month, weekday

---

## How to Run

**1. Preprocess raw data:**
```bash
python src/Data_Preprocessing.py --input_dir <path_to_xlsx_files> --output_dir <output_path_to_save_tickets_processed.pkl >
```

**2. Feature engineering:**
```bash
python src/feature_engineering.py --input_pkl <path_to_tickets_processed.pkl> --output_dir <output_path_for_ticketsdata.csv>
```

**3. Train model:**
```bash
python src/train_classifier.py --input_csv <path_to_ticketsdata.csv> --output_dir <output_path>
```

**4. Evaluate model:**
```bash
python src/evaluate_classifier.py --model_dir <path_to_model> --output_dir <output_path>
```

---

## Tech Stack

- **Python 3.8**
- **scikit-learn** — preprocessing, modelling, evaluation
- **LightGBM** — gradient boosting classifier
- **spaCy** — NER-based text anonymisation
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualisation
- **Azure Machine Learning** — compute and data storage

---

## Data Privacy

All raw ticket data is anonymised before processing:
- Named entities (people, organisations, locations) masked using spaCy NER
- Emails, URLs, IP addresses, phone numbers masked using regex
- Categorical columns (business service, assignment group, service offering) replaced with anonymous labels
