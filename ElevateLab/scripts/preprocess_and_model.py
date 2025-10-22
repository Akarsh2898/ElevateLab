"""Preprocess Titanic data, save processed CSV, train logistic regression with stratified CV,
and save metrics + top coefficients. Also generate Sex_by_Survived and Embarked_by_Survived plots.

Usage:
    python scripts/preprocess_and_model.py
"""
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / 'Titanic-Dataset.csv'
PLOTS_DIR = ROOT / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)
OUT_PROCESSED = ROOT / 'Titanic-Dataset-Processed-ml.csv'
COEF_CSV = ROOT / 'model_coeffs.csv'

# --- helpers

def extract_title(name):
    if pd.isna(name):
        return 'Unknown'
    m = re.search(r",\s*([^.,]*)\.", name)
    if m:
        title = m.group(1).strip()
        return title
    # fallback: try splitting
    parts = name.split(',')
    if len(parts) > 1:
        s = parts[1].split()[0]
        return s.strip().replace('.', '')
    return 'Unknown'


# --- load
print('Loading', DATA_CSV)
df = pd.read_csv(DATA_CSV)

# --- feature engineering
# Title
print('Extracting title')
df['Title'] = df['Name'].apply(extract_title)
# Simplify common titles
common_titles = {'Mr','Mrs','Miss','Master'}
df['Title'] = df['Title'].apply(lambda t: t if t in common_titles else 'Rare')

# Deck from Cabin
print('Extracting deck')
df['Cabin'] = df['Cabin'].fillna('')
df['Deck'] = df['Cabin'].astype(str).str[0].replace('', 'U')
# Normalize unknown deck to 'U'
df['Deck'] = df['Deck'].fillna('U')

# Family size
print('Creating family features')
df['family_size'] = df['SibSp'].fillna(0).astype(int) + df['Parch'].fillna(0).astype(int) + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)

# log fare
print('Transforming Fare')
df['Fare'] = df['Fare'].fillna(0.0)
df['log_fare'] = np.log1p(df['Fare'])

# Age imputation: grouped median by Pclass + Sex + Title
print('Imputing Age')
def grouped_age_impute(df):
    df_age = df.copy()
    # compute grouped medians
    grp = df_age.groupby(['Pclass','Sex','Title'])['Age'].median()
    # apply
    def impute_row(r):
        if pd.notna(r['Age']):
            return r['Age']
        key = (r['Pclass'], r['Sex'], r['Title'])
        if key in grp.index:
            val = grp.loc[key]
            if pd.notna(val):
                return val
        # fallback to median by Pclass+Sex
        val2 = df_age.groupby(['Pclass','Sex'])['Age'].median().loc[(r['Pclass'], r['Sex'])]
        if pd.notna(val2):
            return val2
        # final fallback
        return df_age['Age'].median()
    return df_age.apply(impute_row, axis=1)

try:
    df['Age'] = grouped_age_impute(df)
except Exception:
    # fallback: simple median
    df['Age'] = df['Age'].fillna(df['Age'].median())

# Select features and encoding
print('Encoding categorical features')
cat_cols = ['Sex','Embarked','Pclass','Title','Deck']
# ensure Pclass string
df['Pclass'] = df['Pclass'].astype(str)
df['Sex'] = df['Sex'].astype(str)
df['Embarked'] = df['Embarked'].fillna('Missing').astype(str)

# One-hot
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=False)

# Features to use
features = ['Age','log_fare','family_size','is_alone']
# add all dummies
dummies = [c for c in df_enc.columns if any(prefix in c for prefix in ['Sex_','Embarked_','Pclass_','Title_','Deck_'])]
features += dummies

X = df_enc[features].fillna(0)
y = df_enc['Survived'].astype(int)

# save processed CSV
print('Saving processed CSV to', OUT_PROCESSED)
proc_df = pd.concat([df[['PassengerId','Survived','Name']], X], axis=1)
proc_df.to_csv(OUT_PROCESSED, index=False)

# --- Modeling: logistic regression with stratified CV
print('Training baseline logistic regression (5-fold stratified CV)')
clf = LogisticRegression(solver='liblinear', max_iter=1000)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# get predicted probabilities and classes via cross_val_predict
y_pred = cross_val_predict(clf, X, y, cv=skf)
y_proba = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:,1]

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_proba)

print('CV metrics:')
print(f'  Accuracy: {acc:.4f}')
print(f'  Precision: {prec:.4f}')
print(f'  Recall: {rec:.4f}')
print(f'  F1: {f1:.4f}')
print(f'  ROC AUC: {auc:.4f}')

# Fit on full data to get coefficients
print('Fitting on full data to extract coefficients')
clf.fit(X, y)
coefs = pd.Series(clf.coef_[0], index=X.columns).sort_values(ascending=False)
coefs.to_csv(COEF_CSV, header=['coef'])
print('Top positive coefficients:')
print(coefs.head(10))
print('Top negative coefficients:')
print(coefs.tail(10))

# save metrics to a small file
metrics = {
    'accuracy': acc,
    'precision': prec,
    'recall': rec,
    'f1': f1,
    'roc_auc': auc
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(ROOT / 'model_metrics.csv', index=False)

# --- plots: Sex_by_Survived and Embarked_by_Survived
print('Generating Sex_by_Survived and Embarked_by_Survived plots')
fig, ax = plt.subplots(figsize=(6,4))
sex_stack = df.groupby(['Sex','Survived']).size().unstack(fill_value=0)
sex_stack.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Sex counts stacked by Survived')
ax.set_ylabel('Count')
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'Sex_by_Survived.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6,4))
emb_stack = df.groupby(['Embarked','Survived']).size().unstack(fill_value=0)
emb_stack.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Embarked counts stacked by Survived')
ax.set_ylabel('Count')
fig.tight_layout()
fig.savefig(PLOTS_DIR / 'Embarked_by_Survived.png', dpi=150)
plt.close(fig)

print('All done. Outputs:')
print(' - Processed CSV:', OUT_PROCESSED)
print(' - Model metrics: model_metrics.csv')
print(' - Coefficients: ', COEF_CSV)
print(' - Plots: Sex_by_Survived.png, Embarked_by_Survived.png')

if __name__ == '__main__':
    pass
