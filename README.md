# Will This Visitor Buy? — Purchase Intent Prediction

## Project scope

This project predicts whether an e-commerce browsing session will result in a purchase (`Revenue`).

**Training:** the original UCI dataset is used to train and evaluate the model. The notebook creates a stratified train/test split, builds preprocessing inside a scikit-learn Pipeline, compares a baseline and three algorithms, performs cross-validation and grid search on training data only, evaluates the selected model on the untouched test set, and saves the final pipeline.

**Application:** the Streamlit app loads the saved pipeline and predicts on a CSV uploaded by a store owner. Uploading a CSV does **not** retrain the model.

This version analyzes ready-made CSV session data. It is not connected to a live store and does not collect browsing events automatically.

## Dataset

Online Shoppers Purchasing Intention Dataset, UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

Download the dataset and put the CSV file beside the notebook, naming it:

`online_shoppers_intention.csv`

The target column is `Revenue`. It is used for training/evaluation and must not be included as a prediction input. The app can accept a CSV with or without `Revenue`; if present, it is excluded from prediction and used only for retrospective comparison.

## Project files

- `purchase_intent_training.ipynb` — complete ML workflow.
- `app.py` — Streamlit CSV-upload application.
- `requirements.txt` — Python dependencies.
- `artifacts/purchase_intent_pipeline.joblib` — created after running the notebook.
- `artifacts/sample_sessions_for_app.csv` — held-out example rows created by the notebook.

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the UCI CSV and place it as `online_shoppers_intention.csv`.
4. Open Jupyter and run `purchase_intent_training.ipynb` from top to bottom.
5. Confirm that `artifacts/purchase_intent_pipeline.joblib` has been created.
6. Start the app:

   ```bash
   streamlit run app.py
   ```

7. Upload `artifacts/sample_sessions_for_app.csv` to test the application.

## Input CSV requirements

The CSV must contain the same feature columns and feature meanings as the UCI training dataset, except `Revenue` is not required. The app checks for missing columns and ignores extra columns. Use the generated sample CSV as a template.

## Deployment

Deploy on Streamlit Community Cloud:
1. Push `app.py`, `requirements.txt`, and the generated model artifact to a GitHub repository.
2. In Streamlit Community Cloud, choose the repository and `app.py` as the entry point.
3. Make sure the artifact path remains `artifacts/purchase_intent_pipeline.joblib`.
4. Test the public app using a private/incognito browser window.

For a public repository, check the UCI dataset terms and repository size. The project brief says not to upload large datasets; link the dataset source instead. Do not upload the original CSV if it is not necessary.

## Important reporting rules

- Run the notebook yourself and report only metrics printed by your code.
- Do not claim the app predicts live visitors automatically; it predicts rows from an uploaded CSV.
- The action rules in `app.py` are illustrative business rules, not outputs learned by the classifier.
- Explain limitations: historical public data may not represent a particular store, and predictions are not guarantees.
