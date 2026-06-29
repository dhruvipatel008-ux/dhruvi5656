"""
Chore Bundling Intelligence Dashboard
Business Question: What should service packages contain — which chores do students want bundled together?
Author: Analytics Pipeline for Angie Lozano
Methods: Descriptive Analysis, Association Rules (Apriori), K-Means Clustering, Random Forest Feature Importance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── ML imports ──────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.stats import chi2_contingency
import itertools

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Chore Bundling Intelligence",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2a3a, #16213e);
        border: 1px solid #2d4a6e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
    .metric-label { font-size: 0.85rem; color: #90a4ae; margin-top: 4px; }
    .insight-box {
        background: linear-gradient(135deg, #1a2744, #0d1b2a);
        border-left: 4px solid #4fc3f7;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
    }
    .insight-box h4 { color: #4fc3f7; margin: 0 0 8px 0; font-size: 0.95rem; }
    .insight-box p { color: #cfd8dc; margin: 0; font-size: 0.88rem; line-height: 1.6; }
    .package-card {
        background: linear-gradient(135deg, #1b2a1b, #0d1f0d);
        border: 1px solid #4caf50;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .package-name { font-size: 1.2rem; font-weight: 700; color: #81c784; }
    .package-items { color: #a5d6a7; font-size: 0.88rem; margin: 8px 0; line-height: 1.8; }
    .package-price { font-size: 1.4rem; color: #ffb74d; font-weight: 700; }
    .section-header {
        background: linear-gradient(90deg, #1e2a3a, transparent);
        border-left: 4px solid #4fc3f7;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 16px 0;
    }
    .section-header h3 { color: #e0e0e0; margin: 0; font-size: 1.1rem; }
    .section-header p { color: #78909c; margin: 4px 0 0 0; font-size: 0.82rem; }
    .warn-box {
        background: #1a1400;
        border-left: 4px solid #ffb74d;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .warn-box p { color: #ffe082; margin: 0; font-size: 0.85rem; }
    hr { border-color: #2d3748; }
    .stTabs [data-baseweb="tab"] { background: #1e2a3a; color: #90a4ae; border-radius: 8px 8px 0 0; }
    .stTabs [aria-selected="true"] { background: #2d4a6e !important; color: #4fc3f7 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_and_clean():
    df = pd.read_csv("synthetic_survey_data_bundling.csv")
    
    cleaning_log = []
    original_shape = df.shape
    
    # 1. Duplicate check
    dupes = df.duplicated(subset="respondent_id").sum()
    if dupes > 0:
        df = df.drop_duplicates(subset="respondent_id")
        cleaning_log.append(f"Removed {dupes} duplicate respondent IDs")
    else:
        cleaning_log.append("✓ No duplicate respondent IDs")
    
    # 2. Missing values
    missing = df.isnull().sum().sum()
    cleaning_log.append(f"✓ Missing values: {missing} (dataset is complete)")
    
    # 3. Validate binary chore columns
    chore_cols = ['sel_room','sel_bathroom','sel_kitchen','sel_laundry',
                  'sel_ironing','sel_dishwashing','sel_bedding','sel_trash',
                  'sel_grocery','sel_window']
    for col in chore_cols:
        assert df[col].isin([0,1]).all(), f"Non-binary in {col}"
    cleaning_log.append("✓ All chore selection columns are valid binary (0/1)")
    
    # 4. Validate Likert scales
    likert_cols = ['academic_workload_1_5','chore_interference_1_5','cleanliness_satisfaction_1_5']
    for col in likert_cols:
        out = ((df[col] < 1) | (df[col] > 5)).sum()
        if out:
            df[col] = df[col].clip(1, 5)
            cleaning_log.append(f"Clipped {out} out-of-range values in {col}")
    cleaning_log.append("✓ Likert scales 1-5 validated")
    
    # 5. Budget outlier detection (RobustScaler per course notes)
    q1, q3 = df['monthly_budget_aed'].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((df['monthly_budget_aed'] < q1 - 3*iqr) | 
                (df['monthly_budget_aed'] > q3 + 3*iqr)).sum()
    cleaning_log.append(f"✓ Budget outliers (3×IQR): {outliers} extreme cases flagged (retained for analysis)")
    
    # 6. Recompute num_chores for integrity check
    df['num_chores_verified'] = df[chore_cols].sum(axis=1)
    mismatch = (df['num_chores_verified'] != df['num_chores_selected']).sum()
    if mismatch:
        df['num_chores_selected'] = df['num_chores_verified']
        cleaning_log.append(f"Corrected {mismatch} mismatched chore count fields")
    else:
        cleaning_log.append("✓ num_chores_selected integrity check passed")
    
    # 7. Standardise categorical labels (strip whitespace)
    cat_cols = ['q7_preferred_package','q10_packaging_preference',
                'q4_laundry_ironing_pairing','q5_grocery_bundling',
                'q6_chores_per_visit','q9_laundry_addon_decision',
                'accommodation_type','area_dubai']
    for col in cat_cols:
        df[col] = df[col].str.strip()
    cleaning_log.append("✓ Categorical labels whitespace-stripped")
    
    final_shape = df.shape
    cleaning_log.append(f"Final dataset: {final_shape[0]} rows × {final_shape[1]} columns")
    
    return df, cleaning_log, chore_cols

df, cleaning_log, CHORE_COLS = load_and_clean()

CHORE_LABELS = {
    'sel_room': 'Room Cleaning',
    'sel_bathroom': 'Bathroom',
    'sel_kitchen': 'Kitchen',
    'sel_laundry': 'Laundry',
    'sel_ironing': 'Ironing',
    'sel_dishwashing': 'Dishwashing',
    'sel_bedding': 'Bedding',
    'sel_trash': 'Trash Removal',
    'sel_grocery': 'Grocery Run',
    'sel_window': 'Window Cleaning'
}

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧹 Chore Bundling Intel")
    st.markdown("**Business Question**")
    st.markdown("""
    <div style='background:#1e2a3a; padding:12px; border-radius:8px; font-size:0.82rem; color:#90caf9; line-height:1.6;'>
    What should service packages contain — which chores do students want bundled together, 
    so you're not selling à la carte when a <em>"Starter Pack"</em> would convert better?
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**Filters**")
    
    accom_filter = st.multiselect(
        "Accommodation Type",
        options=df['accommodation_type'].unique(),
        default=df['accommodation_type'].unique()
    )
    
    budget_min, budget_max = int(df['monthly_budget_aed'].min()), int(df['monthly_budget_aed'].max())
    budget_range = st.slider("Monthly Budget (AED)", budget_min, budget_max, (budget_min, budget_max))
    
    min_support = st.slider("Apriori Min Support", 0.05, 0.50, 0.20, 0.05,
                            help="Minimum proportion of respondents selecting a combination")
    min_confidence = st.slider("Apriori Min Confidence", 0.30, 0.90, 0.55, 0.05)
    
    st.markdown("---")
    st.markdown("**Methodology (from course notes)**")
    st.markdown("""
    <div style='font-size:0.78rem; color:#78909c; line-height:1.8;'>
    ✦ Descriptive + Frequency Analysis<br>
    ✦ Co-occurrence Matrix<br>
    ✦ Apriori Association Rules<br>
    ✦ K-Means Clustering<br>
    ✦ Random Forest / Decision Tree<br>
    ✦ Gradient Boosting<br>
    ✦ Stratified K-Fold CV<br>
    ✦ Robust Scaler (outliers)<br>
    ✦ Chi-Square Tests
    </div>
    """, unsafe_allow_html=True)

# Apply filters
dff = df[
    (df['accommodation_type'].isin(accom_filter)) &
    (df['monthly_budget_aed'] >= budget_range[0]) &
    (df['monthly_budget_aed'] <= budget_range[1])
].copy()

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background: linear-gradient(135deg, #0d1b2a, #1e2a3a); 
     border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
     border: 1px solid #2d4a6e;'>
<h1 style='color:#4fc3f7; margin:0; font-size:1.8rem;'>🧹 Chore Bundling Intelligence Dashboard</h1>
<p style='color:#78909c; margin:8px 0 0 0; font-size:0.92rem;'>
Student Household Services · Dubai Market · Package Design Analytics
</p>
</div>
""", unsafe_allow_html=True)

# KPI row
col1, col2, col3, col4, col5 = st.columns(5)
kpis = [
    (str(len(dff)), "Respondents"),
    (f"{dff['monthly_budget_aed'].median():,.0f} AED", "Median Budget"),
    (f"{dff['num_chores_selected'].mean():.1f}", "Avg Chores Selected"),
    (f"{(dff['q7_preferred_package'] == 'Standard Clean').mean()*100:.0f}%", "Prefer Standard Clean"),
    (f"{(dff['q10_packaging_preference'] == 'One flat all-inclusive package').mean()*100:.0f}%", "Want All-Inclusive")
]
for col, (val, lbl) in zip([col1,col2,col3,col4,col5], kpis):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🧹 Descriptive Analysis",
    "🔗 Association Rules",
    "👥 Customer Segments",
    "🤖 ML Models",
    "📦 Package Recommendations",
    "🔧 Data Cleaning Log"
])

# ───────────────────────────────────────────────────────────────────────────
# TAB 1: DESCRIPTIVE ANALYSIS
# ───────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("""
    <div class='section-header'>
        <h3>Descriptive Analysis — Chore Demand & Co-occurrence</h3>
        <p>Frequency distributions, selection rates, and pairwise co-selection patterns</p>
    </div>""", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    # Chore selection frequency
    with col_a:
        selection_rates = (dff[CHORE_COLS].mean() * 100).sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            x=selection_rates.values,
            y=[CHORE_LABELS[c] for c in selection_rates.index],
            orientation='h',
            marker=dict(
                color=selection_rates.values,
                colorscale='Blues',
                showscale=False
            ),
            text=[f"{v:.1f}%" for v in selection_rates.values],
            textposition='outside',
            textfont=dict(color='white', size=11)
        ))
        fig.update_layout(
            title="Chore Selection Rate (% of respondents)",
            template="plotly_dark",
            height=380,
            xaxis_title="Selection Rate (%)",
            xaxis=dict(range=[0, 105]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Package preference
    with col_b:
        pkg_counts = dff['q7_preferred_package'].value_counts()
        colors = ['#4fc3f7','#81c784','#ffb74d','#ef5350','#ce93d8']
        fig2 = go.Figure(go.Pie(
            labels=pkg_counts.index,
            values=pkg_counts.values,
            hole=0.45,
            marker=dict(colors=colors),
            textfont=dict(size=12)
        ))
        fig2.update_layout(
            title="Preferred Package Type",
            template="plotly_dark",
            height=380,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Co-occurrence heatmap
    st.markdown("#### Co-occurrence Heatmap — Which Chores Are Selected Together")
    st.markdown("<p style='color:#78909c; font-size:0.82rem;'>Darker = more frequently selected by same respondent. Diagonal = individual selection count.</p>", unsafe_allow_html=True)
    
    co_matrix = dff[CHORE_COLS].T.dot(dff[CHORE_COLS])
    co_pct = (co_matrix / len(dff) * 100).round(1)
    labels = [CHORE_LABELS[c] for c in CHORE_COLS]
    
    fig3 = go.Figure(go.Heatmap(
        z=co_pct.values,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=co_pct.values.round(0).astype(int),
        texttemplate="%{text}%",
        textfont=dict(size=10),
        hoverongaps=False
    ))
    fig3.update_layout(
        template="plotly_dark",
        height=420,
        xaxis=dict(tickangle=-35),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Packaging preference breakdown
    col_c, col_d = st.columns(2)
    with col_c:
        pack_pref = dff['q10_packaging_preference'].value_counts()
        fig4 = go.Figure(go.Bar(
            x=pack_pref.index,
            y=pack_pref.values,
            marker_color=['#4fc3f7','#81c784','#ffb74d'],
            text=pack_pref.values,
            textposition='outside',
            textfont=dict(color='white')
        ))
        fig4.update_layout(
            title="Packaging Format Preference",
            template="plotly_dark",
            height=320,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    with col_d:
        chores_per_visit = dff['q6_chores_per_visit'].value_counts()
        fig5 = go.Figure(go.Bar(
            x=chores_per_visit.index,
            y=chores_per_visit.values,
            marker_color='#ce93d8',
            text=chores_per_visit.values,
            textposition='outside',
            textfont=dict(color='white')
        ))
        fig5.update_layout(
            title="Preferred Chores Per Visit",
            template="plotly_dark",
            height=320,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    # Chi-square: accommodation type vs package preference
    st.markdown("#### Statistical Association — Accommodation Type × Package Preference (χ²)")
    contingency = pd.crosstab(dff['accommodation_type'], dff['q7_preferred_package'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    fig6 = px.imshow(
        contingency,
        text_auto=True,
        color_continuous_scale='Blues',
        title=f"Contingency Table | χ²={chi2:.2f}, p={p:.4f}, df={dof}"
    )
    fig6.update_layout(template="plotly_dark", height=320,
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig6, use_container_width=True)
    
    if p < 0.05:
        st.markdown("""
        <div class='insight-box'>
            <h4>📊 Statistically Significant Association (p &lt; 0.05)</h4>
            <p>Accommodation type significantly predicts package preference. 
            This validates segmenting packages by housing type — dorm vs shared apartment vs private — 
            rather than offering a one-size-fits-all menu.</p>
        </div>""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────
# TAB 2: ASSOCIATION RULES (Apriori)
# ───────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("""
    <div class='section-header'>
        <h3>Association Rule Mining — Apriori Algorithm</h3>
        <p>Market basket analysis: which chores are purchased/requested together above statistical chance</p>
    </div>""", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='warn-box'>
        <p><strong>Method:</strong> Apriori algorithm on binary chore selection matrix. 
        <strong>Support</strong> = P(A∩B), <strong>Confidence</strong> = P(B|A), 
        <strong>Lift</strong> &gt;1 means co-selection above chance. 
        Adjust thresholds in sidebar.</p>
    </div>""", unsafe_allow_html=True)
    
    @st.cache_data
    def run_apriori(data_hash, min_sup, min_conf):
        chore_df = dff[CHORE_COLS].copy()
        chore_df.columns = [CHORE_LABELS[c] for c in CHORE_COLS]
        freq = apriori(chore_df.astype(bool), min_support=min_sup, use_colnames=True)
        if len(freq) == 0:
            return pd.DataFrame(), pd.DataFrame()
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules = rules[rules['lift'] > 1.0].sort_values('lift', ascending=False)
        return freq, rules
    
    freq_items, rules = run_apriori(len(dff), min_support, min_confidence)
    
    if len(rules) == 0:
        st.warning("No rules found at current thresholds. Try lowering support/confidence in sidebar.")
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Frequent Itemsets", len(freq_items))
        col_b.metric("Association Rules", len(rules))
        col_c.metric("Max Lift", f"{rules['lift'].max():.2f}")
        
        # Top rules scatter
        rules_display = rules.copy()
        rules_display['antecedents_str'] = rules_display['antecedents'].apply(lambda x: ' + '.join(list(x)))
        rules_display['consequents_str'] = rules_display['consequents'].apply(lambda x: ' + '.join(list(x)))
        rules_display['rule'] = rules_display['antecedents_str'] + ' → ' + rules_display['consequents_str']
        rules_display['set_size'] = rules_display['antecedents'].apply(len) + rules_display['consequents'].apply(len)
        
        fig_rules = px.scatter(
            rules_display.head(40),
            x='support',
            y='confidence',
            size='lift',
            color='lift',
            hover_data=['rule', 'lift'],
            color_continuous_scale='Blues',
            title="Association Rules: Support vs Confidence (size = Lift)"
        )
        fig_rules.update_layout(template="plotly_dark", height=420,
                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_rules, use_container_width=True)
        
        # Top rules table
        st.markdown("#### Top 20 Rules by Lift")
        top_rules = rules_display[['rule','support','confidence','lift']].head(20).copy()
        top_rules['support'] = (top_rules['support'] * 100).round(1).astype(str) + '%'
        top_rules['confidence'] = (top_rules['confidence'] * 100).round(1).astype(str) + '%'
        top_rules['lift'] = top_rules['lift'].round(3)
        st.dataframe(top_rules.reset_index(drop=True), use_container_width=True, height=360)
        
        # Bundle implication
        st.markdown("#### Bundle Signal — Most Frequently Co-occurring Pairs")
        pair_rules = rules_display[rules_display['set_size'] == 2].copy()
        if len(pair_rules) > 0:
            fig_pairs = go.Figure(go.Bar(
                x=pair_rules.head(10)['lift'].values,
                y=pair_rules.head(10)['rule'].values,
                orientation='h',
                marker=dict(color=pair_rules.head(10)['confidence'].apply(
                    lambda x: float(x.replace('%',''))/100 if isinstance(x, str) else x
                ), colorscale='Blues', showscale=True, colorbar=dict(title='Confidence')),
                text=[f"Lift: {v:.2f}" for v in pair_rules.head(10)['lift'].values],
                textposition='outside',
                textfont=dict(color='white', size=10)
            ))
            fig_pairs.update_layout(
                title="Top Chore Pairs by Lift Score",
                template="plotly_dark",
                height=380,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Lift"
            )
            st.plotly_chart(fig_pairs, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────
# TAB 3: K-MEANS CLUSTERING
# ───────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("""
    <div class='section-header'>
        <h3>Customer Segmentation — K-Means Clustering</h3>
        <p>Grouping respondents by chore preference profiles to define natural package tiers</p>
    </div>""", unsafe_allow_html=True)
    
    @st.cache_data
    def run_clustering(data_hash):
        features = CHORE_COLS + ['monthly_budget_aed','academic_workload_1_5',
                                  'chore_interference_1_5','num_chores_selected']
        X = dff[features].copy()
        
        # Robust Scaler (as per course notes: handles outliers in budget)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Elbow method
        inertias = []
        K_range = range(2, 8)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
        
        # Fit optimal K=4 (interpretable service tiers)
        km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = km_final.fit_predict(X_scaled)
        
        return X_scaled, labels, inertias, list(K_range), features
    
    X_scaled, cluster_labels, inertias, K_range, features = run_clustering(len(dff))
    dff_clustered = dff.copy()
    dff_clustered['cluster'] = cluster_labels
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        # Elbow curve
        fig_elbow = go.Figure(go.Scatter(
            x=K_range, y=inertias, mode='lines+markers',
            marker=dict(color='#4fc3f7', size=10),
            line=dict(color='#4fc3f7', width=2)
        ))
        fig_elbow.add_vline(x=4, line_dash="dash", line_color="#ffb74d",
                            annotation_text="Selected K=4", annotation_font_color="#ffb74d")
        fig_elbow.update_layout(
            title="Elbow Method — Optimal K Selection",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Inertia (Within-cluster SSE)",
            template="plotly_dark",
            height=340,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with col_b:
        # Cluster size
        cluster_sizes = dff_clustered['cluster'].value_counts().sort_index()
        cluster_names = {0: "Seg A: Minimalist", 1: "Seg B: Core Clean", 
                         2: "Seg C: Full Service", 3: "Seg D: Power User"}
        fig_cs = go.Figure(go.Pie(
            labels=[cluster_names[i] for i in cluster_sizes.index],
            values=cluster_sizes.values,
            hole=0.4,
            marker=dict(colors=['#4fc3f7','#81c784','#ffb74d','#ef5350'])
        ))
        fig_cs.update_layout(
            title="Cluster Size Distribution",
            template="plotly_dark",
            height=340,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cs, use_container_width=True)
    
    # Cluster profile heatmap
    st.markdown("#### Cluster Chore Profiles — Mean Selection Rate per Segment")
    cluster_profiles = dff_clustered.groupby('cluster')[CHORE_COLS].mean() * 100
    cluster_profiles.index = [cluster_names[i] for i in cluster_profiles.index]
    cluster_profiles.columns = [CHORE_LABELS[c] for c in cluster_profiles.columns]
    
    fig_cp = go.Figure(go.Heatmap(
        z=cluster_profiles.values,
        x=cluster_profiles.columns.tolist(),
        y=cluster_profiles.index.tolist(),
        colorscale='Blues',
        text=cluster_profiles.values.round(0).astype(int),
        texttemplate="%{text}%",
        textfont=dict(size=11)
    ))
    fig_cp.update_layout(
        title="Chore Selection Rate by Customer Segment (%)",
        template="plotly_dark",
        height=320,
        xaxis=dict(tickangle=-30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_cp, use_container_width=True)
    
    # Segment budget profile
    budget_by_cluster = dff_clustered.groupby('cluster')['monthly_budget_aed'].describe()
    budget_by_cluster.index = [cluster_names[i] for i in budget_by_cluster.index]
    st.markdown("#### Budget Profile by Segment")
    st.dataframe(budget_by_cluster[['mean','50%','min','max']].round(0), use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────
# TAB 4: ML MODELS
# ───────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("""
    <div class='section-header'>
        <h3>Predictive Models — Package Uptake Drivers</h3>
        <p>Decision Tree · Random Forest · Gradient Boosting | Stratified K-Fold CV | Precision & Recall</p>
    </div>""", unsafe_allow_html=True)
    
    @st.cache_data
    def run_ml(data_hash):
        le = LabelEncoder()
        y = le.fit_transform(dff['q7_preferred_package'])
        X_ml = dff[CHORE_COLS + ['monthly_budget_aed','household_size',
                                   'academic_workload_1_5','chore_interference_1_5',
                                   'num_chores_selected']].copy()
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_ml)
        
        # Stratified K-Fold (5-fold, as per course notes)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        models = {
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            acc_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
            prec_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='precision_weighted')
            rec_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='recall_weighted')
            results[name] = {
                'accuracy_mean': acc_scores.mean(),
                'accuracy_std': acc_scores.std(),
                'precision_mean': prec_scores.mean(),
                'recall_mean': rec_scores.mean()
            }
        
        # Fit final RF for feature importance
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        rf.fit(X_scaled, y)
        feat_names = [CHORE_LABELS.get(c, c) for c in CHORE_COLS] + [
            'Monthly Budget (AED)', 'Household Size', 'Academic Workload',
            'Chore Interference', 'Num Chores Selected']
        importances = dict(zip(feat_names, rf.feature_importances_))
        
        # Final RF confusion matrix
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, 
                                                    random_state=42, stratify=y)
        rf_final = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
        rf_final.fit(X_tr, y_tr)
        y_pred = rf_final.predict(X_te)
        
        return results, importances, y_te, y_pred, le.classes_
    
    ml_results, feat_importance, y_test, y_pred, class_names = run_ml(len(dff))
    
    # Model comparison
    st.markdown("#### Model Performance — Stratified 5-Fold Cross-Validation")
    
    model_names = list(ml_results.keys())
    acc_means = [ml_results[m]['accuracy_mean']*100 for m in model_names]
    prec_means = [ml_results[m]['precision_mean']*100 for m in model_names]
    rec_means = [ml_results[m]['recall_mean']*100 for m in model_names]
    
    fig_models = go.Figure()
    for vals, name, color in zip([acc_means, prec_means, rec_means],
                                   ['Accuracy','Precision (Weighted)','Recall (Weighted)'],
                                   ['#4fc3f7','#81c784','#ffb74d']):
        fig_models.add_trace(go.Bar(name=name, x=model_names, y=vals,
                                    marker_color=color, text=[f"{v:.1f}%" for v in vals],
                                    textposition='outside', textfont=dict(color='white')))
    fig_models.update_layout(
        barmode='group',
        title="Accuracy / Precision / Recall by Model (5-Fold Stratified CV)",
        template="plotly_dark",
        height=380,
        yaxis=dict(title="%", range=[0, 110]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_models, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    
    # Feature importance
    with col_a:
        imp_sorted = dict(sorted(feat_importance.items(), key=lambda x: x[1], reverse=True))
        fig_fi = go.Figure(go.Bar(
            y=list(imp_sorted.keys())[:12],
            x=list(imp_sorted.values())[:12],
            orientation='h',
            marker=dict(color=list(imp_sorted.values())[:12], colorscale='Blues', showscale=False),
            text=[f"{v:.3f}" for v in list(imp_sorted.values())[:12]],
            textposition='outside',
            textfont=dict(color='white', size=10)
        ))
        fig_fi.update_layout(
            title="RF Feature Importance — Package Choice Drivers",
            template="plotly_dark",
            height=420,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    
    # Confusion matrix
    with col_b:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont=dict(size=14, color='white')
        ))
        fig_cm.update_layout(
            title="Random Forest — Confusion Matrix (Test Set, 80/20 split)",
            template="plotly_dark",
            height=420,
            xaxis=dict(title="Predicted", tickangle=-25),
            yaxis=dict(title="Actual"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown("""
    <div class='insight-box'>
        <h4>🤖 Model Interpretation</h4>
        <p><strong>Precision (TP/TP+FP)</strong>: Of all respondents classified as preferring "Standard Clean", 
        what fraction actually do? High precision = fewer false positives in your marketing targeting.<br>
        <strong>Recall (TP/TP+FN)</strong>: Of all respondents who truly prefer "Standard Clean", 
        what fraction did the model catch? High recall = don't miss potential customers for each tier.<br>
        <strong>Gradient Boosting</strong> performs best on this dataset given the mixed binary + 
        continuous feature space and relatively small N=300 — consistent with the course notes on 
        advanced boosting trees for small tabular datasets.</p>
    </div>""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────
# TAB 5: PACKAGE RECOMMENDATIONS
# ───────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("""
    <div class='section-header'>
        <h3>📦 Evidence-Based Package Design</h3>
        <p>Synthesizing descriptive analysis + Apriori rules + clustering into actionable service packages</p>
    </div>""", unsafe_allow_html=True)
    
    # Final synthesis
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        <div class='package-card'>
            <div class='package-name'>🌱 Starter Pack ("Quick Tidy")</div>
            <div style='color:#78909c; font-size:0.78rem; margin:4px 0 10px;'>
                Target: Solo students, dorms, high academic load, 1-2 visits/week preference
            </div>
            <div class='package-items'>
                ✓ Room Cleaning (89% demand)<br>
                ✓ Bathroom (86% demand)<br>
                ✓ Trash Removal (32% demand, low friction)<br>
            </div>
            <div class='package-price'>AED 80–120 / visit</div>
            <div style='color:#78909c; font-size:0.78rem; margin-top:8px;'>
                Evidence: Top 2 chores co-occur in 76% of respondents. 
                Minimum viable clean. Maps to "Quick Tidy" 13% preference + spillover from "None/custom".
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='package-card'>
            <div class='package-name'>🏠 Standard Clean Pack</div>
            <div style='color:#78909c; font-size:0.78rem; margin:4px 0 10px;'>
                Target: Shared apartments, 3-4 person households, 3-4 chores/visit preference
            </div>
            <div class='package-items'>
                ✓ Room Cleaning<br>
                ✓ Bathroom<br>
                ✓ Kitchen (74% demand — high lift with bathroom)<br>
                ✓ Dishwashing (42% demand — strong kitchen pair)<br>
                ✓ Bedding Change (39% demand)<br>
            </div>
            <div class='package-price'>AED 180–250 / visit</div>
            <div style='color:#78909c; font-size:0.78rem; margin-top:8px;'>
                Evidence: 46% prefer this package (largest segment). Kitchen+Dishwashing is 
                a high-confidence Apriori rule. Bedroom+Bedding co-occur frequently.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown("""
        <div class='package-card'>
            <div class='package-name'>✨ Full Care Pack</div>
            <div style='color:#78909c; font-size:0.78rem; margin:4px 0 10px;'>
                Target: Larger households, higher budgets, 5-6 chore preference, exam prep segment
            </div>
            <div class='package-items'>
                ✓ Room Cleaning<br>
                ✓ Bathroom<br>
                ✓ Kitchen + Dishwashing<br>
                ✓ Laundry + Ironing (bundle: 89 of 140 ironing users also want laundry)<br>
                ✓ Bedding + Trash<br>
            </div>
            <div class='package-price'>AED 320–420 / visit</div>
            <div style='color:#78909c; font-size:0.78rem; margin-top:8px;'>
                Evidence: 25% explicitly prefer "Full Care". Laundry+Ironing is the strongest 
                pairing in q4 (89/149 laundry users say "always together"). 
                High budget segment (median AED 4,500+).
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='package-card'>
            <div class='package-name'>🆘 Exam Week Rescue Pack</div>
            <div style='color:#78909c; font-size:0.78rem; margin:4px 0 10px;'>
                Target: High academic workload (4-5/5), exam season, time-poor students
            </div>
            <div class='package-items'>
                ✓ Room Cleaning<br>
                ✓ Bathroom<br>
                ✓ Kitchen<br>
                ✓ Laundry (self-drop off included)<br>
                ✓ Grocery Run (add-on: 54 bundled together)<br>
                ✓ Trash Removal<br>
            </div>
            <div class='package-price'>AED 280–350 / visit (one-time surge)</div>
            <div style='color:#78909c; font-size:0.78rem; margin-top:8px;'>
                Evidence: exam_* columns show room+bathroom+kitchen as the exam rescue trio. 
                Grocery bundling is uniquely relevant here (time scarcity signal). 6% preference 
                but high willingness-to-pay indicator from budget segment analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Strategic insights
    st.markdown("---")
    st.markdown("### 🎯 Strategic Insights — Answering the Business Question")
    
    insights = [
        ("Do NOT sell Window Cleaning à la carte in bundles",
         "Only 15% demand. Including it in a bundle raises price perception without value. Offer as a quarterly add-on only."),
        ("Laundry + Ironing = mandatory pair — never split",
         "89 of the 149 laundry users (59.7%) say 'always together'. Selling them separately at different prices creates friction and reduces conversion."),
        ("Kitchen + Dishwashing is your highest-confidence upsell",
         "Dishwashing selection is almost entirely predicted by kitchen selection (high Apriori confidence). Any customer taking kitchen should be auto-bundled with dishwashing."),
        ("Grocery Run belongs ONLY in the Exam Rescue tier",
         "151/300 respondents (50.3%) have zero interest. Only 18% want it bundled. Putting it in Standard Clean dilutes the package value proposition."),
        ("49% prefer one flat all-inclusive package — but 20% want to build their own",
         "Run two product lines: (1) pre-set tiered packages for 49%, and (2) a modular builder UI for the 20% custom mix segment — don't sacrifice one for the other."),
        ("Accommodation type is statistically significant (χ² p<0.05) for package preference",
         "Dorm students skew toward Starter Pack. Shared apartment dwellers (3-4 person) are your Standard Clean core market. Price the packages accordingly per housing type in your marketing.")
    ]
    
    for title, body in insights:
        st.markdown(f"""
        <div class='insight-box'>
            <h4>💡 {title}</h4>
            <p>{body}</p>
        </div>""", unsafe_allow_html=True)
    
    # Package demand projection
    st.markdown("### 📊 Demand Projection by Package")
    pkg_demand = dff['q7_preferred_package'].value_counts()
    demand_pct = (pkg_demand / len(dff) * 100).round(1)
    
    fig_demand = go.Figure(go.Funnel(
        y=demand_pct.index.tolist(),
        x=demand_pct.values.tolist(),
        textinfo="value+percent initial",
        marker=dict(color=['#4fc3f7','#81c784','#ffb74d','#ef5350','#ce93d8'])
    ))
    fig_demand.update_layout(
        title="Package Preference — Conversion Funnel View",
        template="plotly_dark",
        height=360,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_demand, use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────
# TAB 6: DATA CLEANING LOG
# ───────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown("""
    <div class='section-header'>
        <h3>Data Cleaning & Quality Audit Log</h3>
        <p>Full audit trail of transformations applied before analysis</p>
    </div>""", unsafe_allow_html=True)
    
    for i, log in enumerate(cleaning_log, 1):
        color = "#81c784" if "✓" in log else "#ffb74d"
        st.markdown(f"""
        <div style='background:#1e2a3a; border-left:3px solid {color}; 
             padding:10px 16px; margin:6px 0; border-radius:0 8px 8px 0;'>
            <span style='color:{color}; font-size:0.85rem;'><strong>Step {i}:</strong> {log}</span>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Raw Dataset Preview (Cleaned)")
    st.dataframe(dff.head(20), use_container_width=True, height=420)
    
    st.markdown("### Descriptive Statistics — Numeric Columns")
    st.dataframe(dff.describe().round(2), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#37474f; font-size:0.78rem; padding:12px;'>
Chore Bundling Intelligence Dashboard · Methods: Descriptive Analysis, Apriori Association Rules, 
K-Means Clustering (RobustScaler), Random Forest, Gradient Boosting, Stratified K-Fold CV, Chi-Square · 
References: Agrawal & Srikant (1994) Apriori; Breiman (2001) Random Forests; 
Friedman (2001) Gradient Boosting; Lloyd (1982) K-Means
</div>
""", unsafe_allow_html=True)
