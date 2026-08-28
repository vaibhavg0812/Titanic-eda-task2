# Task 2: Exploratory Data Analysis (EDA) — Titanic Dataset

**Internship:** AI & ML Internship, Elevate Labs
**Objective:** Understand the Titanic dataset using statistics and visualizations.
**Tools used:** Python, Pandas, Matplotlib, Seaborn, Plotly

## Dataset
The Titanic dataset (891 passengers, 15 columns), loaded via `seaborn.load_dataset("titanic")`
and saved locally as `titanic.csv`. Target column: `survived` (0 = did not survive, 1 = survived).

## What this script does (`eda_titanic.py`)
1. **Summary statistics** — mean, median, std, min/max for numeric columns (`age`, `fare`, `sibsp`, `parch`, etc.) via `df.describe()`.
2. **Missing values check** — `deck` (77% missing), `age` (~20% missing), `embarked`/`embark_town` (2 missing).
3. **Histograms** — distribution shape of `age`, `fare`, `sibsp`, `parch`.
4. **Boxplots** — outlier detection for `age` and `fare`, and `fare` split by `pclass`.
5. **Correlation heatmap** — relationships between numeric features and `survived`.
6. **Pairplot** — pairwise feature relationships, colored by survival.
7. **Bar charts** — survival rate by `sex` and by `pclass`.
8. **Interactive Plotly chart** — `age` vs `fare`, colored by survival, with hover details (saved as an HTML file).
9. **Skewness check** — quantifies how skewed each numeric feature is.

All static plots are saved to `plots/`; the interactive chart is `plots/age_vs_fare_interactive.html`.

## Key Findings / Inferences
- **Survival was strongly tied to sex**: women survived at a much higher rate than men (~74% vs ~19%).
- **Passenger class mattered**: 1st class passengers had the highest survival rate, 3rd class the lowest — `pclass` has a negative correlation (-0.34) with `survived`.
- **Fare and class are linked**: `fare` correlates negatively with `pclass` (-0.55) — higher fares bought better (numerically lower) class tickets, and higher fares correlate positively with survival (+0.26).
- **Age has weak direct correlation with survival** (-0.08), but very young children show a somewhat higher survival rate visible in the histogram/pairplot.
- **Fare is heavily right-skewed** (skewness ≈ 4.8) — a small number of passengers paid very high fares, visible as outliers in the boxplot.
- **`sibsp` and `parch` are moderately correlated (0.41)** — passengers traveling with siblings/spouses often also traveled with parents/children (family groups).
- **`deck` has too many missing values (77%)** to be reliably used without heavy imputation or dropping.

## How to run
```bash
pip install pandas numpy matplotlib seaborn plotly
python eda_titanic.py
```

## Interview Q&A Prep (from the task guide)

**1. What is the purpose of EDA?**
To understand a dataset's structure, quality, and patterns — distributions, missing values, outliers, and relationships between features — before modeling, so you can make informed decisions on cleaning, feature engineering, and model choice.

**2. How do boxplots help in understanding a dataset?**
They show the median, interquartile range (IQR), and whiskers, making it easy to spot outliers and compare spread across categories (e.g., fare across passenger classes).

**3. What is correlation and why is it useful?**
Correlation measures the strength and direction of a linear relationship between two numeric variables (from -1 to +1). It's useful for spotting which features move together, flagging redundant (multicollinear) features, and finding features likely predictive of the target.

**4. How do you detect skewness in data?**
Visually via histograms (asymmetric tail) or boxplots (asymmetric whiskers/outliers), and numerically via `df.skew()` — values far from 0 indicate skew (positive = right tail, negative = left tail).

**5. What is multicollinearity?**
When two or more independent (predictor) variables are highly correlated with each other, making it hard to isolate each one's individual effect on the target — this can destabilize coefficients in linear models like linear/logistic regression.

**6. What tools do you use for EDA?**
Pandas (stats, grouping), Matplotlib/Seaborn (static plots — histograms, boxplots, heatmaps, pairplots), and Plotly (interactive plots).

**7. Can you explain a time when EDA helped you find a problem?**
Example from this task: the correlation heatmap and boxplot revealed `fare` was heavily right-skewed with extreme outliers (up to 512 vs a median of ~14) — something a model trained on raw fare values could be distorted by, pointing to a need for a log transform or outlier capping before modeling.

**8. What is the role of visualization in ML?**
It makes patterns, outliers, and relationships in data intuitively visible — things easy to miss in raw numbers — and helps validate assumptions, guide feature engineering, and communicate findings to others.
