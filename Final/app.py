"""Adult Census Income Prediction — Production API."""

import numpy as np
import pandas as pd
import joblib
from collections import deque
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

app = FastAPI(title="Income Prediction API")
model = joblib.load("models/final_model_pipeline.joblib")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_HISTORY = 20

WORKCLASS = ['Federal-gov','Local-gov','Private','Self-emp-inc','Self-emp-not-inc','State-gov','Without-pay']
EDUCATION = ['10th','11th','12th','1st-4th','5th-6th','7th-8th','9th','Assoc-acdm','Assoc-voc','Bachelors','Doctorate','HS-grad','Masters','Preschool','Prof-school','Some-college']
MARITAL = ['Divorced','Married-AF-spouse','Married-civ-spouse','Married-spouse-absent','Never-married','Separated','Widowed']
OCCUPATION = ['Adm-clerical','Armed-Forces','Craft-repair','Exec-managerial','Farming-fishing','Handlers-cleaners','Machine-op-inspct','Other-service','Priv-house-serv','Prof-specialty','Protective-serv','Sales','Tech-support','Transport-moving']
RELATIONSHIP = ['Husband','Not-in-family','Other-relative','Own-child','Unmarried','Wife']
RACE = ['Amer-Indian-Eskimo','Asian-Pac-Islander','Black','Other','White']
SEX = ['Female','Male']
COUNTRY = ['Cambodia','Canada','China','Columbia','Cuba','Dominican-Republic','Ecuador','El-Salvador','England','France','Germany','Greece','Guatemala','Haiti','Holand-Netherlands','Honduras','Hong','India','Iran','Ireland','Italy','Jamaica','Japan','Laos','Mexico','Nicaragua','Outlying-US(Guam-USVI-etc)','Peru','Philippines','Poland','Portugal','Puerto-Rico','Scotland','South','Taiwan','Thailand','Trinadad&Tobago','United-States','Vietnam','Yugoslavia']

EDU_MAP = {'Preschool':'Basic','1st-4th':'Basic','5th-6th':'Basic','7th-8th':'Basic','9th':'Basic','10th':'High School','11th':'High School','12th':'High School','HS-grad':'High School','Some-college':'College Prep','Assoc-voc':'Associate','Assoc-acdm':'Associate','Bachelors':'Bachelors','Masters':'Masters','Prof-school':'Doctorate','Doctorate':'Doctorate'}
MARITAL_MAP = {'Married-civ-spouse':'Married','Married-spouse-absent':'Married','Married-AF-spouse':'Married','Never-married':'Not-Married','Divorced':'Not-Married','Separated':'Not-Married','Widowed':'Not-Married'}

REGION_MAP = {
    'US/Canada': ['United-States','Canada'],
    'Central America': ['Mexico','Puerto-Rico','Cuba','Jamaica','Haiti','Dominican-Republic','El-Salvador','Guatemala','Nicaragua','Honduras','Trinadad&Tobago'],
    'South America': ['Columbia','Ecuador','Peru'],
    'Asia': ['China','India','Japan','Taiwan','Philippines','Vietnam','Thailand','Cambodia','Laos'],
    'Europe': ['England','Germany','France','Italy','Greece','Poland','Portugal','Ireland','Scotland','Hungary','Yugoslavia','Holand-Netherlands'],
}
COUNTRY_TO_REGION = {c: r for r, cs in REGION_MAP.items() for c in cs}

# In-memory prediction history
history: deque = deque(maxlen=MAX_HISTORY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _opts(options, selected=None):
    return "\n".join(f'<option value="{o}"{" selected" if o == selected else ""}>{o}</option>' for o in options)


def _build_input(form: dict) -> pd.DataFrame:
    age, edu_num = int(form['age']), int(form['edu_num'])
    cg, cl = int(form['capital_gain']), int(form['capital_loss'])
    return pd.DataFrame([{
        'age': age, 'workclass': form['workclass'], 'fnlwgt': 100000,
        'education': form['education'], 'education.num': edu_num,
        'marital.status': form['marital'], 'occupation': form['occupation'],
        'relationship': form['relationship'], 'race': form['race'],
        'sex': form['sex'], 'capital.gain': cg, 'capital.loss': cl,
        'hours.per.week': int(form['hours']), 'native.country': form['country'],
        'capital.gain_log': np.log1p(cg), 'capital.loss_log': np.log1p(cl),
        'age_group': pd.cut([age], bins=[0,25,35,45,55,65,100],
            labels=['<25','25-34','35-44','45-54','55-64','65+'], right=False)[0],
        'hours_group': pd.cut([int(form['hours'])], bins=[0,20,30,40,50,60,100],
            labels=['<20','20-29','30-39','40-49','50-59','60+'], right=False)[0],
        'capital_total': cg + cl, 'has_capital': int(cg + cl > 0),
        'education_group': EDU_MAP.get(form['education'], 'Other'),
        'marital_grouped': MARITAL_MAP.get(form['marital'], 'Other'),
        'edu_age_ratio': edu_num / (age + 1),
        'region': COUNTRY_TO_REGION.get(form['country'], 'Other'),
    }])


def _history_rows() -> str:
    if not history:
        return '<p style="color:#86868b;font-size:.85em;text-align:center;padding:20px 0">No predictions yet</p>'
    rows = []
    for h in reversed(history):
        color = '#059669' if h['pred'] == 1 else '#6b7280'
        bg = '#ecfdf5' if h['pred'] == 1 else '#f9fafb'
        label = '> $50K' if h['pred'] == 1 else '\u2264 $50K'
        rows.append(
            f'<div style="padding:10px 12px;border-radius:8px;background:{bg};margin-bottom:6px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<span style="font-weight:600;font-size:.88em;color:{color}">{label}</span>'
            f'<span style="font-size:.75em;color:#86868b">{h["prob"]:.0%}</span>'
            f'</div>'
            f'<div style="font-size:.75em;color:#6e6e73;margin-top:4px">{h["age"]}y \u00b7 {h["education"]} \u00b7 {h["occupation"]}</div>'
            f'</div>'
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------
def _render(form: dict | None = None, pred: int | None = None, prob: float | None = None) -> str:
    f = form or {}
    result_html = ""
    if pred is not None:
        label = "> $50K" if pred == 1 else "\u2264 $50K"
        pct = round(prob * 100)
        if pred == 1:
            result_html = f'<div class="result hi"><span class="result-label">Predicted Income: {label}</span><div class="bar"><div class="bar-fill hi" style="width:{pct}%"></div><span class="bar-text">{pct}%</span></div></div>'
        else:
            result_html = f'<div class="result lo"><span class="result-label">Predicted Income: {label}</span><div class="bar"><div class="bar-fill lo" style="width:{pct}%"></div><span class="bar-text">{pct}%</span></div></div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Income Predictor</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif;background:#f5f5f7;color:#1d1d1f;display:flex;min-height:100vh}}
/* --- Sidebar --- */
.side{{width:280px;background:#fff;border-right:1px solid #e5e7eb;display:flex;flex-direction:column;flex-shrink:0}}
.side-head{{padding:20px 20px 16px;border-bottom:1px solid #e5e7eb}}
.side-head h2{{font-size:1em;font-weight:600;color:#1d1d1f;margin-bottom:2px}}
.side-head p{{font-size:.78em;color:#86868b}}
.side-body{{flex:1;overflow-y:auto;padding:12px 16px}}
/* --- Main --- */
.main{{flex:1;display:flex;flex-direction:column;min-width:0}}
.topbar{{height:56px;background:#fff;border-bottom:1px solid #e5e7eb;display:flex;align-items:center;padding:0 32px;gap:12px}}
.topbar h1{{font-size:1.15em;font-weight:600;letter-spacing:-.01em}}
.topbar .tag{{font-size:.7em;background:#e0e7ff;color:#3730a3;padding:3px 8px;border-radius:980px;font-weight:500}}
.content{{flex:1;padding:32px;max-width:720px}}
/* --- Form --- */
.section{{margin-bottom:24px}}
.sec-title{{font-size:.72em;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#86868b;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #e5e7eb}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px 14px}}
.fg{{display:flex;flex-direction:column}}
label{{font-size:.78em;font-weight:500;color:#6e6e73;margin-bottom:4px}}
select,input[type=number]{{height:38px;padding:0 10px;border-radius:8px;border:1px solid #d1d5db;background:#fff;color:#1d1d1f;font-size:.85em;font-family:inherit;outline:none;transition:border .15s}}
select{{background:#fff url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='10'%3E%3Cpath fill='%2386868b' d='M5 7L0 2h10z'/%3E%3C/svg%3E") no-repeat right 10px center;padding-right:28px;cursor:pointer}}
select:focus,input[type=number]:focus{{border-color:#0071e3;box-shadow:0 0 0 3px rgba(0,113,227,.1)}}
.btn{{height:42px;border:none;border-radius:980px;background:#0071e3;color:#fff;font-size:.9em;font-weight:500;font-family:inherit;cursor:pointer;transition:background .15s;margin-top:6px}}
.btn:hover{{background:#0077ed}}
/* --- Result --- */
.result{{margin-top:24px;padding:18px;border-radius:12px;border:1px solid #d1d5db;text-align:center}}
.result.hi{{border-color:#bfdbfe;background:#eff6ff}}
.result.lo{{border-color:#e5e7eb;background:#f9fafb}}
.result-label{{font-size:.92em;font-weight:500}}
.result.hi .result-label{{color:#1d4ed8}}
.result.lo .result-label{{color:#374151}}
.bar{{margin-top:10px;background:#f5f5f7;border-radius:6px;height:22px;position:relative;overflow:hidden}}
.bar-fill{{height:100%;border-radius:6px;transition:width .4s ease}}
.bar-fill.hi{{background:#3b82f6}}.bar-fill.lo{{background:#9ca3af}}
.bar-text{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:.75em;font-weight:600;color:#1d1d1f}}
/* --- History --- */
.h-item{{padding:10px 12px;border-radius:8px;margin-bottom:6px}}
.h-item.hi{{background:#ecfdf5}}.h-item.lo{{background:#f9fafb}}
.h-head{{display:flex;justify-content:space-between;align-items:center}}
.h-label{{font-weight:600;font-size:.85em}}.h-item.hi .h-label{{color:#059669}}.h-item.lo .h-label{{color:#6b7280}}
.h-prob{{font-size:.73em;color:#86868b}}
.h-meta{{font-size:.73em;color:#6e6e73;margin-top:3px}}
</style>
</head>
<body>

<aside class="side">
  <div class="side-head">
    <h2>Prediction History</h2>
    <p>Last {MAX_HISTORY} predictions</p>
  </div>
  <div class="side-body">
    {_history_rows()}
  </div>
</aside>

<div class="main">
  <div class="topbar">
    <h1>Income Predictor</h1>
    <span class="tag">ML Powered</span>
  </div>
  <div class="content">
    <form method="post" action="/predict">
      <div class="section">
        <div class="sec-title">Personal</div>
        <div class="grid">
          <div class="fg"><label>Age</label><input type="number" name="age" value="{f.get('age',30)}" min="17" max="90" required></div>
          <div class="fg"><label>Sex</label><select name="sex">{_opts(SEX, f.get('sex','Male'))}</select></div>
          <div class="fg"><label>Race</label><select name="race">{_opts(RACE, f.get('race','White'))}</select></div>
          <div class="fg"><label>Country</label><select name="country">{_opts(COUNTRY, f.get('country','United-States'))}</select></div>
        </div>
      </div>
      <div class="section">
        <div class="sec-title">Education</div>
        <div class="grid">
          <div class="fg"><label>Level</label><select name="education">{_opts(EDUCATION, f.get('education','Bachelors'))}</select></div>
          <div class="fg"><label>Years (1\u201316)</label><input type="number" name="edu_num" value="{f.get('edu_num',13)}" min="1" max="16" required></div>
        </div>
      </div>
      <div class="section">
        <div class="sec-title">Employment</div>
        <div class="grid">
          <div class="fg"><label>Workclass</label><select name="workclass">{_opts(WORKCLASS, f.get('workclass','Private'))}</select></div>
          <div class="fg"><label>Occupation</label><select name="occupation">{_opts(OCCUPATION, f.get('occupation','Prof-specialty'))}</select></div>
          <div class="fg"><label>Hours / Week</label><input type="number" name="hours" value="{f.get('hours',40)}" min="1" max="99" required></div>
          <div class="fg"><label>Marital Status</label><select name="marital">{_opts(MARITAL, f.get('marital','Never-married'))}</select></div>
          <div class="fg"><label>Relationship</label><select name="relationship">{_opts(RELATIONSHIP, f.get('relationship','Not-in-family'))}</select></div>
        </div>
      </div>
      <div class="section">
        <div class="sec-title">Financial</div>
        <div class="grid">
          <div class="fg"><label>Capital Gain ($)</label><input type="number" name="capital_gain" value="{f.get('capital_gain',0)}" min="0" required></div>
          <div class="fg"><label>Capital Loss ($)</label><input type="number" name="capital_loss" value="{f.get('capital_loss',0)}" min="0" required></div>
        </div>
      </div>
      <button type="submit" class="btn">Predict</button>
    </form>
    {result_html}
  </div>
</div>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(_render())


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    age: int = Form(...), hours: int = Form(...), education: str = Form(...),
    edu_num: int = Form(...), workclass: str = Form(...), occupation: str = Form(...),
    marital: str = Form(...), relationship: str = Form(...), race: str = Form(...),
    sex: str = Form(...), capital_gain: int = Form(...), capital_loss: int = Form(...),
    country: str = Form(...),
):
    form = dict(age=age, hours=hours, education=education, edu_num=edu_num,
                workclass=workclass, occupation=occupation, marital=marital,
                relationship=relationship, race=race, sex=sex,
                capital_gain=capital_gain, capital_loss=capital_loss, country=country)

    X = _build_input(form)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    history.append(dict(pred=pred, prob=prob, age=age, education=education, occupation=occupation))

    return HTMLResponse(_render(form, pred, prob))


@app.get("/history", response_class=HTMLResponse)
async def get_history():
    """Return history as JSON-like HTML for AJAX or debugging."""
    import json
    return HTMLResponse(f"<pre>{json.dumps(list(history), indent=2)}</pre>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
