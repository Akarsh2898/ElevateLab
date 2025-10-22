"""Generate feature importance bar plot and an HTML snippet with model metrics and top coefficients."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / 'plots'
PLOTS.mkdir(exist_ok=True)
METRICS_CSV = ROOT / 'model_metrics.csv'
COEF_CSV = ROOT / 'model_coeffs.csv'
OUT_IMG = PLOTS / 'feature_importance.png'
OUT_HTML = ROOT / 'model_summary_fragment.html'

# load
metrics = pd.read_csv(METRICS_CSV)
coefs = pd.read_csv(COEF_CSV, index_col=0).squeeze()
coefs = coefs.sort_values(key=lambda s: s.abs(), ascending=False)

# plot top 15 by abs value
top = coefs.head(15)
fig, ax = plt.subplots(figsize=(8,6))
colors = ['#d62728' if v<0 else '#1f77b4' for v in top.values]
ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
ax.set_xlabel('Coefficient (logistic regression)')
ax.set_title('Top 15 feature coefficients (abs sorted)')
fig.tight_layout()
fig.savefig(OUT_IMG, dpi=150)
plt.close(fig)

# create HTML fragment
metrics_row = metrics.iloc[0].to_dict()
html = []
html.append('<section id="model_summary"><h2>Model summary</h2>')
html.append('<h3>Cross-validated metrics</h3>')
html.append('<table border="1" style="width:40%">')
html.append('<tr><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC AUC</th></tr>')
html.append(f"<tr><td>{metrics_row['accuracy']:.3f}</td><td>{metrics_row['precision']:.3f}</td><td>{metrics_row['recall']:.3f}</td><td>{metrics_row['f1']:.3f}</td><td>{metrics_row['roc_auc']:.3f}</td></tr>")
html.append('</table>')
html.append('<h3>Top coefficients</h3>')
html.append('<table border="1" style="width:60%"><tr><th>Feature</th><th>Coef</th></tr>')
for feat, val in coefs.head(10).items():
    html.append(f'<tr><td>{feat}</td><td>{val:.4f}</td></tr>')
html.append('</table>')
html.append('<h3>Feature importance</h3>')
html.append(f'<img src="plots/feature_importance.png" alt="feature importance">')
html.append('</section>')

OUT_HTML.write_text('\n'.join(html), encoding='utf-8')
print('Wrote', OUT_IMG, 'and', OUT_HTML)
