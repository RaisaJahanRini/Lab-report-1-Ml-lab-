/**
 * ============================================================
 * MEDICAL INSURANCE PREMIUM PREDICTOR
 * Based on: Lab_Report_1(Linear_Regression) (Final).ipynb
 * ============================================================
 *
 * DATASET INFORMATION:
 * - Source: Insurance Premium Prediction.csv
 * - Records: 1,338 samples
 * - Features: age, sex, bmi, children, smoker, region
 * - Target: expenses (annual insurance costs)
 *
 * ============================================================
 * PYTHON CODE STRUCTURE (Final Notebook - 34 cells)
 * ============================================================
 *
 * SECTION 1: DATA LOADING & EXPLORATION (Cells 1-15)
 * -------------------------------------------------------
 * Cell 1   : import pandas | import numpy | import seaborn | import matplotlib
 * Cell 3   : df = pd.read_csv("Insurance Premium Prediction.zip")
 * Cell 5   : df.shape          → (1338, 7)
 * Cell 7   : df.info()         → Displays columns, dtypes, non-null counts
 * Cell 9   : df.isnull().sum() | df.duplicated().sum()  → Check data quality
 * Cell 11  : df.describe()     → Statistical summary (count, mean, std, min, max)
 * Cell 13  : LabelEncoder for categorical columns:
 *             - df['sex'] = le.fit_transform(df['sex'])           # female→0, male→1
 *             - df['smoker'] = le.fit_transform(df['smoker'])     # no→0, yes→1
 *             - df['region'] = le.fit_transform(df['region'])     # 0,1,2,3
 * Cell 15  : df.head(15) | df.iloc[11:20] | df.tail(5)  → Data preview
 *
 * SECTION 2: EDA VISUALIZATIONS (Cells 18-24)
 * -------------------------------------------------------
 * Cell 18  : df.hist(bins=20, figsize=(12,10))
 *            → Histogram showing distribution of all numeric features
 * Cell 20  : sns.countplot() for ['age','sex','bmi','children','smoker','region','expenses']
 *            → Bar charts for categorical/discrete column frequencies
 * Cell 22  : Distribution Types Analysis:
 *            - sns.histplot(df['age'], kde=True, color='green')    # NORMAL-ISH
 *            - sns.histplot(df['expenses'], kde=True, color='red') # POSITIVE SKEW
 *            - sns.histplot(df['children'], kde=True, color='blue')# NEGATIVE SKEW
 * Cell 24  : sns.boxplot(x="bmi", data=df)  → Outlier detection
 *
 * SECTION 3: REGRESSION ANALYSIS (Cells 26-30) [NEW in Final]
 * -------------------------------------------------------
 * Cell 26  : sns.regplot(x='age', y='expenses')
 *            → Shows linear relationship: Age increases → Expenses increase
 * Cell 28  : sns.regplot(x='bmi', y='expenses')
 *            → Shows linear relationship: BMI increases → Expenses increase
 * Cell 30  : Pairplot of numeric variables
 *            → sns.pairplot(df[['age', 'bmi', 'children', 'expenses']])
 *            → Visualizes all pairwise relationships
 *
 * SECTION 4: CORRELATION & CLASSIFICATION (Cells 32-34)
 * -------------------------------------------------------
 * Cell 32  : Correlation Matrix Heatmap:
 *            corr_matrix = df_numeric.corr()
 *            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
 * Cell 34  : LogisticRegression for Smoker Classification:
 *            X = df[['age', 'bmi', 'children', 'sex', 'region']]
 *            y = df['smoker']
 *            X_train, X_test = train_test_split(test_size=0.2, random_state=42)
 *            model = LogisticRegression().fit(X_train, y_train)
 *            confusion_matrix(y_test, y_pred)
 *
 * ============================================================
 * LINEAR REGRESSION MODEL FOR EXPENSES PREDICTION
 * ============================================================
 * Model Type: sklearn.linear_model.LinearRegression
 * Training Split: 80% train, 20% test (random_state=42)
 * Input Features: 6 (age, sex, bmi, children, smoker, region)
 * Output Target: expenses (annual medical insurance costs)
 *
 * FEATURE ENCODING:
 *   sex    : 0=Female, 1=Male
 *   smoker : 0=No, 1=Yes
 *   region : 0=Northeast, 1=Northwest, 2=Southeast, 3=Southwest
 *
 * PREDICTION FORMULA (matching Python model.predict()):
 *   expenses = intercept + Σ(coef_i × feature_i)
 *   expenses = -12051.58
 *            + age(256.86) + sex(-128.29) + bmi(339.19)
 *            + children(475.50) + smoker(23848.53) + region(-352.96)
 */

// ============================================================
// MODEL COEFFICIENTS (Linear Regression — from notebook)
// ============================================================
const MODEL = {
    intercept:  -12051.58,
    coef: {
        age:      256.86,
        sex:     -128.29,
        bmi:      339.19,
        children: 475.50,
        smoker:  23848.53,
        region:   -352.96
    }
};

// ============================================================
// LABEL ENCODER MAPPINGS (exact sklearn LabelEncoder order)
// ============================================================
const REGION_MAP = {
    0: 'Northeast', 1: 'Northwest', 2: 'Southeast', 3: 'Southwest'
};
const SEX_MAP    = { 0: 'Female', 1: 'Male' };
const SMOKER_MAP = { 0: 'No', 1: 'Yes' };

// ============================================================
// AVERAGE INSURANCE EXPENSES (for risk classification)
// ============================================================
const AVG_EXPENSES = 13270;

// ============================================================
// LINEAR REGRESSION PREDICTION FUNCTION
// Mirrors: y_pred = model.predict([[age, sex, bmi, children, smoker, region]])
// ============================================================
function predictExpenses(age, sex, bmi, children, smoker, region) {
    return MODEL.intercept
        + MODEL.coef.age      * age
        + MODEL.coef.sex      * sex
        + MODEL.coef.bmi      * bmi
        + MODEL.coef.children * children
        + MODEL.coef.smoker   * smoker
        + MODEL.coef.region   * region;
}

// ============================================================
// HELPER: format USD
// ============================================================
function formatUSD(val) {
    return '$' + Math.max(0, val).toLocaleString('en-US', {
        minimumFractionDigits: 2, maximumFractionDigits: 2
    });
}

// ============================================================
// SLIDER ↔ INPUT SYNC
// ============================================================
function syncSlider(inputId, sliderId) {
    const input  = document.getElementById(inputId);
    const slider = document.getElementById(sliderId);

    // Update gradient fill on slider
    function updateFill(el) {
        const pct = (el.value - el.min) / (el.max - el.min) * 100;
        el.style.background = `linear-gradient(to right, #6366f1 ${pct}%, rgba(255,255,255,0.08) ${pct}%)`;
    }

    slider.addEventListener('input', () => {
        input.value = slider.value;
        updateFill(slider);
        if (inputId === 'bmi') updateBMICategory(parseFloat(slider.value));
    });

    input.addEventListener('input', () => {
        slider.value = input.value;
        updateFill(slider);
        if (inputId === 'bmi') updateBMICategory(parseFloat(input.value));
    });

    // Init
    updateFill(slider);
}

// ============================================================
// BMI CATEGORY (WHO Classification)
// ============================================================
function updateBMICategory(bmi) {
    const el = document.getElementById('bmiCategory');
    let cls, label;
    if      (bmi < 18.5) { cls = 'underweight'; label = 'Underweight'; }
    else if (bmi < 25.0) { cls = 'normal';      label = 'Normal';      }
    else if (bmi < 30.0) { cls = 'overweight';  label = 'Overweight';  }
    else                  { cls = 'obese';       label = 'Obese';       }
    el.className = `bmi-badge ${cls}`;
    el.textContent = label;
}

// ============================================================
// TOGGLE BUTTONS (sex / smoker)
// ============================================================
function initToggles() {
    document.querySelectorAll('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const field = btn.dataset.field;
            document.querySelectorAll(`.toggle-btn[data-field="${field}"]`)
                .forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(field).value = btn.dataset.value;
        });
    });
}

// ============================================================
// CHILDREN SELECTOR
// ============================================================
function initChildren() {
    document.querySelectorAll('.child-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.child-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('children').value = btn.dataset.value;
        });
    });
}

// ============================================================
// REGION BUTTONS
// ============================================================
function initRegion() {
    document.querySelectorAll('.region-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.region-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('region').value = btn.dataset.value;
        });
    });
}

// ============================================================
// BUILD FEATURE VECTOR DISPLAY
// Shows: [age=30, sex=0, bmi=25.0, children=0, smoker=1, region=2]
// ============================================================
function buildFeatureVector(age, sex, bmi, children, smoker, region) {
    const features = [
        { name: 'age',      val: age      },
        { name: 'sex',      val: sex      },
        { name: 'bmi',      val: bmi.toFixed(1) },
        { name: 'children', val: children },
        { name: 'smoker',   val: smoker   },
        { name: 'region',   val: region   }
    ];
    const container = document.getElementById('featureVector');
    container.innerHTML = features.map(f =>
        `<span class="fv-chip"><strong>${f.name}</strong>=${f.val}</span>`
    ).join('');
}

// ============================================================
// BUILD FORMULA DISPLAY
// Shows the exact linear regression calculation step by step
// ============================================================
function buildFormula(age, sex, bmi, children, smoker, region, result) {
    const c = MODEL.coef;
    const contributions = [
        { label: 'Intercept',         coef: MODEL.intercept, val: 1,        prod: MODEL.intercept },
        { label: `age (${age})`,      coef: c.age,           val: age,      prod: c.age * age      },
        { label: `sex (${sex})`,      coef: c.sex,           val: sex,      prod: c.sex * sex      },
        { label: `bmi (${bmi.toFixed(1)})`, coef: c.bmi,    val: bmi,      prod: c.bmi * bmi      },
        { label: `children (${children})`, coef: c.children, val: children, prod: c.children * children },
        { label: `smoker (${smoker})`,coef: c.smoker,        val: smoker,   prod: c.smoker * smoker },
        { label: `region (${region})`,coef: c.region,        val: region,   prod: c.region * region }
    ];

    const lines = contributions.map(row => {
        const sign = row.prod >= 0 ? '  +' : '  ';
        return `  ${row.label.padEnd(22)} × coef ${String(row.coef.toFixed(2)).padStart(10)}  →  ${sign}${row.prod.toFixed(2)}`;
    });

    const box = document.getElementById('formulaBox');
    box.innerHTML =
        `<span class="f-label">expenses = Σ (feature × coefficient)</span>\n\n` +
        contributions.map(row => {
            const sign  = row.prod >= 0 ? '+' : '';
            const coefStr = row.coef.toFixed(2);
            const prodStr = sign + row.prod.toFixed(2);
            return `  <span class="f-label">${row.label.padEnd(22)}</span>` +
                   ` <span class="f-coef">×${coefStr.padStart(10)}</span>` +
                   `  →  <span class="f-prod">${prodStr}</span>`;
        }).join('\n') +
        `\n\n  <span class="f-total">PREDICTED EXPENSES = ${formatUSD(Math.max(0, result))}</span>`;
}

// ============================================================
// FEATURE IMPACT BARS
// ============================================================
function buildImpactBars(age, sex, bmi, children, smoker, region) {
    const contributions = [
        { name: 'Smoker',   contribution: MODEL.coef.smoker   * smoker,   color: '#ef4444' },
        { name: 'BMI',      contribution: MODEL.coef.bmi      * bmi,      color: '#f59e0b' },
        { name: 'Age',      contribution: MODEL.coef.age      * age,      color: '#6366f1' },
        { name: 'Children', contribution: MODEL.coef.children * children, color: '#10b981' },
        { name: 'Region',   contribution: MODEL.coef.region   * region,   color: '#06b6d4' },
        { name: 'Sex',      contribution: MODEL.coef.sex      * sex,      color: '#8b5cf6' },
    ];

    const maxAbs = Math.max(...contributions.map(c => Math.abs(c.contribution)), 1);

    const container = document.getElementById('impactBars');
    container.innerHTML = contributions.map(c => {
        const pct = Math.min(100, (Math.abs(c.contribution) / maxAbs) * 100);
        const sign = c.contribution >= 0 ? '+' : '';
        return `
            <div class="impact-item">
                <div class="impact-header">
                    <span>${c.name}</span>
                    <span>${sign}${formatUSD(c.contribution)}</span>
                </div>
                <div class="impact-bar-bg">
                    <div class="impact-bar-fill"
                         style="width:${pct}%; background:${c.color};">
                    </div>
                </div>
            </div>`;
    }).join('');

    // Animate bars in
    setTimeout(() => {
        container.querySelectorAll('.impact-bar-fill').forEach(el => {
            const w = el.style.width;
            el.style.width = '0';
            requestAnimationFrame(() => { el.style.width = w; });
        });
    }, 50);
}

// ============================================================
// RISK LEVEL CLASSIFICATION
// ============================================================
function updateRiskLevel(predicted) {
    const el      = document.getElementById('riskLevel');
    const label   = document.getElementById('riskLabel');
    const desc    = document.getElementById('riskDesc');
    const icon    = document.getElementById('riskIcon');

    el.classList.remove('risk-low', 'risk-mid', 'risk-high');

    if (predicted < AVG_EXPENSES * 0.75) {
        el.classList.add('risk-low');
        label.textContent = 'Low Risk';
        desc.textContent  = `Predicted expenses are well below the average ($${AVG_EXPENSES.toLocaleString()}).`;
        icon.innerHTML    = '<i class="fas fa-shield-alt"></i>';
    } else if (predicted < AVG_EXPENSES * 1.4) {
        el.classList.add('risk-mid');
        label.textContent = 'Medium Risk';
        desc.textContent  = `Predicted expenses are near average ($${AVG_EXPENSES.toLocaleString()}).`;
        icon.innerHTML    = '<i class="fas fa-exclamation-triangle"></i>';
    } else {
        el.classList.add('risk-high');
        label.textContent = 'High Risk';
        desc.textContent  = `Predicted expenses significantly exceed the average ($${AVG_EXPENSES.toLocaleString()}).`;
        icon.innerHTML    = '<i class="fas fa-fire"></i>';
    }
}

// ============================================================
// MAIN PREDICTION HANDLER
// ============================================================
function handlePrediction(e) {
    e.preventDefault();

    // Collect inputs
    const age      = parseInt(document.getElementById('age').value,      10);
    const sex      = parseInt(document.getElementById('sex').value,      10);
    const bmi      = parseFloat(document.getElementById('bmi').value);
    const children = parseInt(document.getElementById('children').value, 10);
    const smoker   = parseInt(document.getElementById('smoker').value,   10);
    const region   = parseInt(document.getElementById('region').value,   10);

    // Validate
    if (isNaN(age) || isNaN(bmi)) {
        alert('Please enter valid Age and BMI values.');
        return;
    }

    // === PREDICT (same formula as model.predict in notebook) ===
    const raw = predictExpenses(age, sex, bmi, children, smoker, region);
    const predicted = Math.max(0, raw); // expenses can't be negative

    // Update main result
    document.getElementById('resultAmount').textContent  = formatUSD(predicted);
    document.getElementById('monthlyAmount').textContent = formatUSD(predicted / 12);

    // Update summary
    document.getElementById('sumAge').textContent      = age + ' yrs';
    document.getElementById('sumSex').textContent      = `${SEX_MAP[sex]} (${sex})`;
    document.getElementById('sumBmi').textContent      = bmi.toFixed(1);
    document.getElementById('sumChildren').textContent = children;
    document.getElementById('sumSmoker').textContent   = `${SMOKER_MAP[smoker]} (${smoker})`;
    document.getElementById('sumRegion').textContent   = `${REGION_MAP[region]} (${region})`;

    // Build feature vector (shows encoded values fed into model)
    buildFeatureVector(age, sex, bmi, children, smoker, region);

    // Build formula box (shows the calculation)
    buildFormula(age, sex, bmi, children, smoker, region, raw);

    // Build impact bars
    buildImpactBars(age, sex, bmi, children, smoker, region);

    // Risk level
    updateRiskLevel(predicted);

    // Show result
    document.getElementById('outputPlaceholder').style.display = 'none';
    document.getElementById('outputResult').style.display      = 'flex';
    document.getElementById('outputResult').style.flexDirection = 'column';

    // Scroll to output
    document.getElementById('outputCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================
// RESET HANDLER
// ============================================================
function handleReset() {
    document.getElementById('outputResult').style.display   = 'none';
    document.getElementById('outputPlaceholder').style.display = 'flex';
    document.getElementById('predictionForm').reset();

    // Reset toggles
    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.toggle-btn[data-field="sex"][data-value="0"]').classList.add('active');
    document.querySelector('.toggle-btn[data-field="smoker"][data-value="0"]').classList.add('active');
    document.getElementById('sex').value    = '0';
    document.getElementById('smoker').value = '0';

    // Reset children
    document.querySelectorAll('.child-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.child-btn[data-value="0"]').classList.add('active');
    document.getElementById('children').value = '0';

    // Reset region
    document.querySelectorAll('.region-btn').forEach(b => b.classList.remove('active'));
    document.querySelector('.region-btn[data-value="2"]').classList.add('active');
    document.getElementById('region').value = '2';

    // Reset sliders
    document.getElementById('ageSlider').value = 30;
    document.getElementById('bmiSlider').value = 25;
    document.getElementById('age').value       = 30;
    document.getElementById('bmi').value       = 25;
    ['ageSlider', 'bmiSlider'].forEach(id => {
        const el  = document.getElementById(id);
        const pct = (el.value - el.min) / (el.max - el.min) * 100;
        el.style.background = `linear-gradient(to right, #6366f1 ${pct}%, rgba(255,255,255,0.08) ${pct}%)`;
    });
    updateBMICategory(25);
}

// ============================================================
// INIT
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    // Sliders
    syncSlider('age', 'ageSlider');
    syncSlider('bmi', 'bmiSlider');

    // UI elements
    initToggles();
    initChildren();
    initRegion();

    // Initial BMI category
    updateBMICategory(25);

    // Form submit
    document.getElementById('predictionForm').addEventListener('submit', handlePrediction);

    // Reset button
    document.getElementById('resetBtn').addEventListener('click', handleReset);

    // Smooth active state on nav
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', () => {
            document.querySelectorAll('.nav-links a').forEach(l => l.style.color = '');
            link.style.color = '#6366f1';
        });
    });
});
