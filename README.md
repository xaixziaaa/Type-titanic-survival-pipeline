# Titanic Survival Classification Pipeline

**Author:** [@xaixziaaa](https://github.com/xaixziaaa)

Most introductory Titanic notebooks follow the same pattern: fill missing ages with the global median, one-hot encode the whole dataframe, and then run `train_test_split`. While that works for a quick script, it leaks validation statistics into the training set and makes it painful to run predictions on new, uncleaned data later.

I refactored my original exploration into a single `scikit-learn` pipeline. Raw passenger rows go in, and the pipeline handles feature extraction, missing value imputation, scaling, and classification inside each cross-validation fold without leaking data.

![Results Summary](results_summary.png)

## What's Inside the Pipeline

### 1. Exploratory Data Findings
* **Missing Data:** `Cabin` is missing for 77.10% (687) of passengers, `Age` is missing for 19.87% (177), and `Embarked` is missing for 0.22% (2).
* **Baseline Survival:** Overall training survival rate is 38.4%.
* **Key Demographic Splits:** Female passengers survived at 74.2% compared to 18.9% for male passengers, and 1st-class passengers survived at more than double the rate of 3rd-class passengers (63.0% vs. 24.2%).

### 2. Custom Feature Engineering (`FeatureBuilder`)
Instead of mutating the dataframe manually, feature creation sits inside a custom `BaseEstimator` and `TransformerMixin` class:
* **Title Extraction:** Pulls social titles (`Mr`, `Mrs`, `Miss`, `Master`) out of the `Name` column and groups rare nobility/military titles into `Rare`. This gives the model a useful proxy for age and social status when `Age` is null.
* **Family Group Size:** Combines `SibSp + Parch + 1` into `FamilySize` and flags solo travelers (`IsAlone`). Solo men in 3rd class and very large families had significantly lower survival rates than small families of 2 to 4.
* **Cabin Indicator:** Because 77.1% of `Cabin` values are missing, dropping the column loses useful signal. Converting it to a binary `HasCabin` flag captures whether a passenger had a recorded cabin number (which correlates strongly with 1st-class upper decks).

### 3. Preprocessing (`ColumnTransformer`)
* **Numeric columns (`Age`, `Fare`, `FamilySize`):** Imputed with the fold median and scaled using `RobustScaler` so extreme `Fare` outliers (up to 512.33) don't skew linear weights.
* **Categorical columns (`Pclass`, `Sex`, `Embarked`, `Title`, `IsAlone`, `HasCabin`):** Imputed with the most frequent category and one-hot encoded with `handle_unknown="ignore"`.

## Model Comparison (5-Fold Stratified CV)

All three models were evaluated across the same 5-fold `StratifiedKFold` split on the training set (712 rows) before checking the final holdout test set (179 rows):

| Model | CV Accuracy | CV F1-Score | CV ROC-AUC |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.827 ($\pm$0.025) | 0.771 | 0.871 |
| Random Forest | 0.826 ($\pm$0.021) | 0.757 | 0.881 |
| **HistGradientBoosting** | **0.830 ($\pm$0.026)** | **0.768** | **0.888** |

### Holdout Test Set Performance
The final Random Forest pipeline (`n_estimators=250, max_depth=6`) was evaluated on the unseen 20% holdout split:
* **Holdout Test Accuracy:** `82.1%` (`0.821`)
* **Holdout Test ROC-AUC:** `0.851`
* **Top Predictive Features (Permutation Importance):** `Name` (via extracted `Title`), `Sex`, `Pclass`, and `Age` caused the largest accuracy drops when shuffled on the test set.

## Conclusion & Key Takeaways

1. **Preventing Data Leakage Matters:** Wrapping feature extraction and imputation inside a `Pipeline` ensures validation folds never see test medians or category encodings prematurely.
2. **Domain Features Outperform Raw Columns:** Permutation importance on the holdout set showed that `Name` (which feeds the engineered `Title` feature) and `Cabin` (which feeds `HasCabin`) carry strong predictive signal that is lost if those columns are simply dropped.
3. **Tree Ensembles vs. Linear Models:** While `HistGradientBoosting` achieved the highest cross-validation ROC-AUC (`0.888`), `RandomForest` had the lowest fold-to-fold variance ($\pm$0.021), making it a reliable choice for small tabular datasets (~891 rows).

## Repo Structure

```text
.
├── Titanic_Survival_Pipeline_xaixziaaa.ipynb   # Main EDA & modeling notebook
├── titanic_rf_pipeline.joblib                  # Saved end-to-end model pipeline
├── results_summary.png                         # Confusion matrix, ROC & permutation importance
├── requirements.txt                            # Python dependencies
└── README.md
```

## Quickstart

Clone the repo and install the dependencies:

```bash
git clone https://github.com/xaixziaaa/titanic-survival-pipeline.git
cd titanic-survival-pipeline
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Using the Saved Pipeline on Raw Data

Because feature engineering is baked into the pipeline, you can pass raw passenger dictionaries directly into `titanic_rf_pipeline.joblib` without manual preprocessing:

```python
import joblib
import pandas as pd

pipe = joblib.load("titanic_rf_pipeline.joblib")

raw_passenger = pd.DataFrame([{
    "PassengerId": 999,
    "Pclass": 1,
    "Name": "Graham, Miss. Margaret Edith",
    "Sex": "female",
    "Age": 19.0,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": "112053",
    "Fare": 30.0,
    "Cabin": "B42",
    "Embarked": "S"
}])

pred = pipe.predict(raw_passenger)[0]
prob = pipe.predict_proba(raw_passenger)[0, 1]

print(f"Predicted Survival: {pred} (Probability: {prob:.1%})")
```

## Dataset & Credits

* **Dataset Source:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) (mirror loaded via [Data Science Dojo GitHub](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) for standalone Colab reproducibility).
* **Author:** [@xaixziaaa](https://github.com/xaixziaaa)
* **Libraries:** Built with Python, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, and `seaborn`.