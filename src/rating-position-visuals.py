import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Streamlit Setup ───────────────────────────────────────────────────────────
st.set_page_config(page_title="FC 24 Ultimate Dashboard", page_icon="⚽", layout="wide")

# ── Global UI Colors ──────────────────────────────────────────────────────────
BG, CARD, GRID = "#0d1117", "#161b22", "#21262d"
PINK, AMBER, GREEN = "#ff4081", "#f39c12", "#00e676"

# ── Load Data & Train Models (Cached for speed) ───────────────────────────────
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    filename = os.path.join(project_root, 'data', 'processed', 'cleaned_male_fc_24_players.csv')
    return pd.read_csv(filename)

@st.cache_resource
def train_and_evaluate(_df_model):
    X_reg = _df_model.drop(columns=['overall_rating', 'best_position', 'name'])
    y_reg = _df_model['overall_rating']
    X_class = _df_model.drop(columns=['best_position', 'overall_rating', 'name'])
    y_class = _df_model['best_position']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

    scaler_r, scaler_c = StandardScaler(), StandardScaler()
    X_train_r_sc = scaler_r.fit_transform(X_train_r)
    X_test_r_sc = scaler_r.transform(X_test_r)
    X_train_c_sc = scaler_c.fit_transform(X_train_c)
    X_test_c_sc = scaler_c.transform(X_test_c)

    reg_models = {
        "Linear Regression": LinearRegression(),
        "Ridge (α=10)": Ridge(alpha=10.0),
        "Polynomial (degree=2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    }
    class_models = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    reg_results, class_results = {}, {}
    
    for name, model in reg_models.items():
        model.fit(X_train_r_sc, y_train_r)
        preds = model.predict(X_test_r_sc)
        
        # Save coefficients if it's Ridge to show Feature Importance later
        coefs = None
        if name == "Ridge (α=10)":
            coefs = model.coef_
            
        reg_results[name] = {
            "r2": r2_score(y_test_r, preds),
            "mse": mean_squared_error(y_test_r, preds),
            "preds": preds,
            "actual": y_test_r.values,
            "coefs": coefs
        }

    for name, model in class_models.items():
        model.fit(X_train_c_sc, y_train_c)
        preds = model.predict(X_test_c_sc)
        labels = sorted(y_class.unique())
        class_results[name] = {
            "acc": accuracy_score(y_test_c, preds),
            "cm": confusion_matrix(y_test_c, preds, labels=labels),
            "labels": labels,
        }

    return reg_results, class_results, X_reg.columns.tolist()

# Load everything into memory
with st.spinner("Loading data and training models..."):
    df_model = load_data()
    reg_results, class_results, feature_names = train_and_evaluate(df_model)

# ── Sidebar Interface & Customization ─────────────────────────────────────────
st.sidebar.title("⚙️ Dashboard Controls")

# 1. Color Customizer!
st.sidebar.markdown("### 🎨 Visual Customization")
custom_color = st.sidebar.color_picker("Pick a Primary Theme Color", "#00e5ff")

# 2. Navigation
st.sidebar.markdown("### 🧭 Navigation")
view_mode = st.sidebar.radio(
    "Choose a View:",
    [
        "1. Dataset Explorer (Power BI Style)", 
        "2. ML Models Summary", 
        "3. Regression Analysis", 
        "4. Classification Analysis"
    ]
)

st.title("⚽ FC 24 Ultimate ML Dashboard")
st.markdown("---")

# ── VIEW 1: Dataset Explorer (Power BI Style) ─────────────────────────────────
if view_mode == "1. Dataset Explorer (Power BI Style)":
    st.header("🔍 Dataset Explorer")
    st.write("Analyze the raw data before it gets fed into the machine learning models. Choose features below to see how they correlate.")
    
    # Show raw data toggle
    if st.checkbox("Show Raw Data Table"):
        st.dataframe(df_model.head(100), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Interactive Scatter Plot")
        x_axis = st.selectbox("X-Axis Feature", feature_names, index=feature_names.index('pace') if 'pace' in feature_names else 0)
        y_axis = st.selectbox("Y-Axis Feature", ['overall_rating'] + feature_names, index=0)
        
        # Plotly Express makes Power BI style charts easy
        fig_scatter = px.scatter(
            df_model, x=x_axis, y=y_axis, color="best_position", 
            hover_data=['name'], opacity=0.6,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_scatter.update_layout(paper_bgcolor=BG, plot_bgcolor=CARD, font=dict(color="white"))
        fig_scatter.update_xaxes(gridcolor=GRID)
        fig_scatter.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig_scatter, width="stretch")

    with col2:
        st.subheader("Feature Distribution (Histogram)")
        hist_feature = st.selectbox("Select Feature for Histogram", ['overall_rating'] + feature_names, index=0)
        
        fig_hist = px.histogram(
            df_model, x=hist_feature, nbins=40, 
            color_discrete_sequence=[custom_color]
        )
        fig_hist.update_layout(paper_bgcolor=BG, plot_bgcolor=CARD, font=dict(color="white"))
        fig_hist.update_xaxes(gridcolor=GRID)
        fig_hist.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig_hist, width="stretch")

# ── VIEW 2: ML Models Summary ─────────────────────────────────────────────────
elif view_mode == "2. ML Models Summary":
    st.header("🏆 Overall Metrics Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Regression (Predicting Rating)")
        r_names = list(reg_results.keys())
        r2s = [reg_results[n]["r2"] for n in r_names]
        
        # Using the custom color!
        fig_r = go.Figure(go.Bar(x=r2s, y=r_names, orientation="h", marker_color=custom_color, text=[f"{v:.4f}" for v in r2s], textposition="auto"))
        fig_r.update_layout(title="R² Scores (Higher is Better)", paper_bgcolor=BG, plot_bgcolor=CARD, font=dict(color="white"))
        st.plotly_chart(fig_r, width="stretch")

    with col2:
        st.markdown("### Classification (Predicting Position)")
        c_names = list(class_results.keys())
        accs = [class_results[n]["acc"] for n in c_names]
        
        # Using the custom color!
        fig_c = go.Figure(go.Bar(x=accs, y=c_names, orientation="h", marker_color=custom_color, text=[f"{v:.3f}" for v in accs], textposition="auto"))
        fig_c.update_layout(title="Accuracy Scores (Higher is Better)", paper_bgcolor=BG, plot_bgcolor=CARD, font=dict(color="white"))
        st.plotly_chart(fig_c, width="stretch")

# ── VIEW 3: Regression Analysis (Added Feature Importance) ────────────────────
elif view_mode == "3. Regression Analysis":
    st.header("📈 Regression Deep-Dive")
    
    # TOP TABS for sub-views
    tab1, tab2 = st.tabs(["Actual vs Predicted", "Feature Importance (Effect on Prediction)"])
    
    with tab1:
        selected_reg = st.selectbox("Select Regression Model", list(reg_results.keys()) + ["Overlay All"])
        fig = go.Figure()
        
        if selected_reg == "Overlay All":
            for i, (name, color) in enumerate(zip(reg_results.keys(), [custom_color, PINK, GREEN])):
                actual = reg_results[name]["actual"]
                preds = reg_results[name]["preds"]
                fig.add_trace(go.Scatter(x=actual, y=preds, mode="markers", marker=dict(color=color, opacity=0.5), name=name))
        else:
            actual = reg_results[selected_reg]["actual"]
            preds = reg_results[selected_reg]["preds"]
            fig.add_trace(go.Scatter(x=actual, y=preds, mode="markers", marker=dict(color=custom_color, opacity=0.6), name=selected_reg))

        lim = [int(actual.min()) - 1, int(actual.max()) + 1]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="white", dash="dash"), name="Perfect fit"))
        
        fig.update_layout(height=600, paper_bgcolor=BG, plot_bgcolor=CARD, font=dict(color="white"), xaxis_title="Actual Rating", yaxis_title="Predicted Rating")
        fig.update_xaxes(gridcolor=GRID)
        fig.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig, width="stretch")
        
    with tab2:
        st.subheader("Which features impact the Overall Rating the most?")
        st.write("This chart extracts the internal mathematical weights from the Ridge Regression model.")
        
        coefs = reg_results["Ridge (α=10)"]["coefs"]
        
        # Sort features by absolute importance
        coef_df = pd.DataFrame({"Feature": feature_names, "Importance": coefs})
        coef_df['Absolute Importance'] = coef_df['Importance'].abs()
        coef_df = coef_df.sort_values(by="Absolute Importance", ascending=True).tail(15) # Show top 15
        
        fig_imp = px.bar(
            coef_df, x="Importance", y="Feature", orientation='h',
            color_discrete_sequence=[custom_color]
        )
        fig_imp.update_layout(height=600, paper_bgcolor=BG, plot_bgcolor=CARD, font=dict(color="white"))
        fig_imp.update_xaxes(gridcolor=GRID)
        fig_imp.update_yaxes(gridcolor=GRID)
        st.plotly_chart(fig_imp, width="stretch")

# ── VIEW 4: Classification Analysis ───────────────────────────────────────────
elif view_mode == "4. Classification Analysis":
    st.header("🎯 Classification Models: Confusion Matrices")
    selected_class = st.selectbox("Select Classification Model", list(class_results.keys()))
    
    res = class_results[selected_class]
    cm = res["cm"]
    labels = res["labels"]
    
    # Normalized the confusion matrix by row (actual position)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    hover_text = []
    for r in range(len(labels)):
        row_text = []
        for c in range(len(labels)):
            pct = cm_normalized[r, c] * 100
            row_text.append(f"Actual: {labels[r]}<br>Predicted: {labels[c]}<br>Count: {cm[r, c]}<br>Accuracy: {pct:.1f}%")
        hover_text.append(row_text)
    
    # Create a dynamic color scale based on user's custom color!
    custom_colorscale = [[0, CARD], [1, custom_color]]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized, 
        x=labels, 
        y=labels, 
        colorscale=custom_colorscale,
        zmin=0, zmax=1,
        customdata=cm,
        text=hover_text,
        hoverinfo="text",
        texttemplate="%{customdata}", 
        textfont={"size":12, "color":"white"}
    ))
    
    fig.update_layout(
        title=f"{selected_class} (Accuracy: {res['acc']:.3f})",
        xaxis_title="Predicted Position", 
        yaxis_title="Actual Position",
        height=700, paper_bgcolor=BG, plot_bgcolor=CARD, font=dict(color="white")
    )
    st.plotly_chart(fig, width="stretch")

# ── Run Normally Hack ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    from streamlit.web import cli as stcli
    from streamlit.runtime import exists
    
    if not exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())