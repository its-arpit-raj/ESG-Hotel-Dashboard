import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy.optimize import linprog

# --- 1. DATASET GENERATION ---
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_firms = 24
    years = list(range(2013, 2020))
    
    data = []
    for firm_id in range(1, n_firms + 1):
        base_esg = np.clip(np.random.normal(57.3, 15.0), 10, 95)
        base_size = np.random.normal(8.0, 1.3)
        base_leverage = np.random.uniform(0.2, 0.8)
        base_cash = np.random.uniform(0.02, 0.15)
        
        for year in years:
            esg = np.clip(base_esg + np.random.normal(0, 3), 10, 95)
            env = np.clip(esg + np.random.normal(0, 5), 10, 95)
            soc = np.clip(esg + np.random.normal(0, 5), 10, 95)
            gov = np.clip(esg + np.random.normal(0, 8), 10, 95)
            
            gdp = np.random.normal(2.4, 1.4)
            inf = np.random.normal(1.7, 1.4)
            unemp = np.random.normal(7.4, 2.0)
            
            risk = np.clip(100 - (0.8 * esg) + np.random.normal(0, 5), 0, 100)
            
            eff_target = 0.40 + (0.003 * soc) + (0.002 * env) + (0.0001 * gov) - (0.001 * risk)
            eff_target = np.clip(eff_target + np.random.normal(0, 0.05), 0.1, 1.0)
            
            capex = np.random.lognormal(mean=5.0, sigma=0.5)
            xsga = np.random.lognormal(mean=5.5, sigma=0.5)
            revenue = (capex + xsga) * eff_target * 4.0 
            
            data.append({
                'Firm_ID': f'Hotel_{firm_id:02d}', 'Year': year,
                'ESG_Score': round(esg, 2), 'ENV': round(env, 2), 
                'SOC': round(soc, 2), 'GOV': round(gov, 2),
                'Risk_Score': round(risk, 2), 'Revenue': round(revenue, 2),
                'CAPEX': round(capex, 2), 'XSGA': round(xsga, 2),
                'Leverage': round(base_leverage, 2), 'Size': round(base_size, 2), 
                'Cash': round(base_cash, 2), 'GDP_Growth': round(gdp, 2), 
                'Inflation': round(inf, 2), 'Unemployment': round(unemp, 2)
            })
            
    return pd.DataFrame(data)

# --- 2. DEA SOLVER ---
@st.cache_data
def calculate_dea(df):
    inputs = df[['CAPEX', 'XSGA']].values
    outputs = df[['Revenue']].values
    n = len(df)
    
    eff_scores = []
    for i in range(n):
        c = np.array([1.0] + [0.0] * n)
        A_ub_in = np.hstack((-inputs[i].reshape(-1, 1), inputs.T))
        b_ub_in = np.zeros(inputs.shape[1])
        A_ub_out = np.hstack((np.zeros((outputs.shape[1], 1)), -outputs.T))
        b_ub_out = -outputs[i]
        A_ub = np.vstack((A_ub_in, A_ub_out))
        b_ub = np.concatenate((b_ub_in, b_ub_out))
        A_eq = np.array([[0.0] + [1.0] * n])
        b_eq = np.array([1.0])
        bounds = [(0, None)] * (n + 1)
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            eff_scores.append(round(res.x[0], 4))
        else:
            eff_scores.append(np.nan)
            
    df['Efficiency'] = eff_scores
    return df

# --- SETUP & CALCULATIONS ---
st.set_page_config(layout="wide", page_title="ESG & Efficiency Simulator")
raw_df = generate_data()
df = calculate_dea(raw_df)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module:", [
    "1️⃣ Data Transparency", 
    "2️⃣ DEA Efficiency Analysis", 
    "3️⃣ ESG Performance Analysis", 
    "4️⃣ ESG → Risk → Efficiency Model", 
    "5️⃣ Business Impact Simulator"
])

# --- PAGE 1: DATA TRANSPARENCY ---
if page == "1️⃣ Data Transparency":
    st.title("Dataset Overview & Transparency")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average ESG", round(df['ESG_Score'].mean(), 2))
    col2.metric("Average Efficiency", round(df['Efficiency'].mean(), 3))
    col3.metric("Average Risk", round(df['Risk_Score'].mean(), 2))
    col4.metric("Most Efficient Hotel", df.loc[df['Efficiency'].idxmax(), 'Firm_ID'])
    
    st.markdown("---")
    
    st.subheader("Data Filters")
    col_f1, col_f2, col_f3 = st.columns(3)
    selected_firm = col_f1.multiselect("Select Firm(s):", df['Firm_ID'].unique(), default=df['Firm_ID'].unique()[:3])
    selected_year = col_f2.slider("Select Year Range:", int(df['Year'].min()), int(df['Year'].max()), (2013, 2019))
    esg_range = col_f3.slider("ESG Score Range:", int(df['ESG_Score'].min()), int(df['ESG_Score'].max()), (10, 95))
    
    filtered_df = df[
        (df['Firm_ID'].isin(selected_firm)) & 
        (df['Year'] >= selected_year[0]) & (df['Year'] <= selected_year[1]) &
        (df['ESG_Score'] >= esg_range[0]) & (df['ESG_Score'] <= esg_range[1])
    ]
    
    st.dataframe(filtered_df)
    st.download_button("Download Dataset (CSV)", filtered_df.to_csv(index=False), "hotel_esg_data.csv", "text/csv")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe().T[['mean', 'std', 'min', 'max']])
    with col_b:
        st.subheader("Correlation Matrix")
        corr = df[['Efficiency', 'ESG_Score', 'Risk_Score', 'Revenue', 'CAPEX', 'XSGA']].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)

# --- PAGE 2: DEA EFFICIENCY ---
elif page == "2️⃣ DEA Efficiency Analysis":
    st.title("Data Envelopment Analysis (DEA)")
    
    with st.expander("View DEA Mathematical Model"):
        st.latex(r"\theta = \min \theta")
        st.latex(r"\text{Subject to:}")
        st.latex(r"\sum_{j=1}^n \lambda_j x_{vj} \le \theta x_{vo}")
        st.latex(r"\sum_{j=1}^n \lambda_j y_{rj} \ge y_{ro}")
        st.latex(r"\sum_{j=1}^n \lambda_j = 1 \quad \text{(VRS)}")
        st.latex(r"\lambda_j \ge 0")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(df, x="Efficiency", nbins=20, title="Distribution of Efficiency Scores", color_discrete_sequence=['indigo'])
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        avg_eff = df.groupby('Firm_ID')['Efficiency'].mean().reset_index().sort_values('Efficiency', ascending=False)
        fig_bar = px.bar(avg_eff, x='Firm_ID', y='Efficiency', title="Average Efficiency by Hotel", color='Efficiency')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.subheader("DEA Frontier (Inputs vs Revenue)")
    df['Total_Inputs'] = df['CAPEX'] + df['XSGA']
    df['Is_Efficient'] = np.where(df['Efficiency'] >= 0.99, 'Efficient (Frontier)', 'Inefficient')
    fig_scatter = px.scatter(df, x='Total_Inputs', y='Revenue', color='Is_Efficient', hover_data=['Firm_ID', 'Efficiency'], title="Production Frontier Map")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- PAGE 3: ESG PERFORMANCE ---
elif page == "3️⃣ ESG Performance Analysis":
    st.title("ESG Dimensions & Efficiency")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_esg_dist = px.histogram(df, x="ESG_Score", title="Histogram of ESG Scores", color_discrete_sequence=['green'])
        st.plotly_chart(fig_esg_dist, use_container_width=True)
    with col2:
        esg_melted = df.melt(id_vars=['Firm_ID'], value_vars=['ENV', 'SOC', 'GOV'], var_name='Pillar', value_name='Score')
        fig_box = px.box(esg_melted, x='Pillar', y='Score', color='Pillar', title="ENV vs SOC vs GOV Distribution")
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("ESG Impact on Efficiency")
    fig_esg_eff = px.scatter(df, x="ESG_Score", y="Efficiency", trendline="ols", title="ESG Score vs Operational Efficiency", opacity=0.7)
    st.plotly_chart(fig_esg_eff, use_container_width=True)

# --- PAGE 4: ESG -> RISK -> EFFICIENCY MODEL ---
elif page == "4️⃣ ESG → Risk → Efficiency Model":
    st.title("Mediation Analysis: Risk Management")
    
    st.markdown("### Theoretical Framework")
    st.markdown("**ESG Investment** ➔ Lowers **Risk** ➔ Improves **Operational Efficiency**")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_r1 = px.scatter(df, x="ESG_Score", y="Risk_Score", trendline="ols", title="Higher ESG → Lower Risk", color_discrete_sequence=['red'])
        st.plotly_chart(fig_r1, use_container_width=True)
    with col2:
        fig_r2 = px.scatter(df, x="Risk_Score", y="Efficiency", trendline="ols", title="Lower Risk → Higher Efficiency", color_discrete_sequence=['purple'])
        st.plotly_chart(fig_r2, use_container_width=True)

    st.subheader("Regression Estimation (OLS)")
    
    X = sm.add_constant(df[['ESG_Score', 'Risk_Score', 'Leverage', 'Size', 'Cash', 'GDP_Growth', 'Inflation', 'Unemployment']])
    y = df['Efficiency']
    model = sm.OLS(y, X).fit()
    
    st.text(model.summary().as_text())

# --- PAGE 5: BUSINESS IMPACT SIMULATOR ---
elif page == "5️⃣ Business Impact Simulator":
    st.title("Interactive Business Simulator")
    st.markdown("Adjust the parameters below to see how specific ESG strategies alter Risk, Operational Efficiency, and **Financial Revenue**.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Firm Inputs")
        
        sim_mode = st.radio("Simulation Mode:", ["Granular E, S, G", "Overall ESG Score"])
        
        if sim_mode == "Overall ESG Score":
            sim_esg = st.slider("Target ESG Score", 10, 100, 57)
            # Extrapolate components equally for math if overall is chosen
            sim_env, sim_soc, sim_gov = sim_esg, sim_esg, sim_esg
        else:
            sim_env = st.slider("Environmental (E) Score", 10, 100, 55)
            sim_soc = st.slider("Social (S) Score", 10, 100, 58)
            sim_gov = st.slider("Governance (G) Score", 10, 100, 58)
            # Calculate overall ESG from the three pillars
            sim_esg = (sim_env + sim_soc + sim_gov) / 3
            st.info(f"**Calculated Overall ESG Score: {sim_esg:.1f}**")
            
        st.markdown("---")
        st.markdown("**Financial Inputs (Cost Factors)**")
        st.info("💡 *Increasing costs won't change your efficiency multiplier, but it will scale your final revenue volume!*")
        sim_capex = st.slider("CAPEX ($ Millions)", 50, 500, 193)
        sim_xsga = st.slider("Operating Cost (XSGA in $M)", 100, 800, 322)
        
    with col2:
        st.subheader("Simulated Financial & Operational Outcomes")
        
        # 1. Math for Risk
        pred_risk = np.clip(100 - (0.8 * sim_esg), 0, 100)
        
        # 2. Math for Efficiency (Applying specific weights from the research)
        pred_eff = np.clip(0.40 + (0.003 * sim_soc) + (0.002 * sim_env) + (0.0001 * sim_gov) - (0.001 * pred_risk), 0.1, 1.0)
        
        # 3. Math for Financial Output (Revenue) tied to CAPEX/XSGA
        pred_rev = pred_eff * (sim_capex + sim_xsga) * 4.0 
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Risk", f"{pred_risk:.1f}", delta=f"{pred_risk - df['Risk_Score'].mean():.1f} vs Avg", delta_color="inverse")
        c2.metric("Predicted Efficiency", f"{pred_eff:.3f}", delta=f"{pred_eff - df['Efficiency'].mean():.3f} vs Avg")
        c3.metric("Projected Revenue", f"${pred_rev:,.1f}M", delta=f"${pred_rev - df['Revenue'].mean():,.1f}M vs Avg")
        
        st.markdown("---")
        st.markdown("### The Underlying Equations")
        st.latex(r"\text{ESG Score} = \frac{\text{ENV} + \text{SOC} + \text{GOV}}{3}")
        st.latex(r"\text{Risk Score} = 100 - (0.8 \times \text{ESG})")
        st.latex(r"\text{Efficiency} (\theta) = \beta_0 + \beta_1(\text{SOC}) + \beta_2(\text{ENV}) + \beta_3(\text{GOV}) - \beta_4(\text{Risk})")
        st.latex(r"\text{Projected Revenue} = \theta \times (\text{CAPEX} + \text{XSGA}) \times \text{Industry Scale Factor}")

        st.markdown("### Strategic Insight")
        if sim_esg > 70:
            st.success("🟢 **Strong ESG Profile:** By maintaining high ESG standards, your operational risk is suppressed. Your CAPEX and Operating Costs are transformed into Revenue at maximum efficiency.")
        elif sim_esg < 40:
            st.error("🔴 **Vulnerable ESG Profile:** Low ESG exposure has driven up your Risk Score. As a result, your inputs (CAPEX/XSGA) are suffering from extreme DEA inefficiency, leaving Revenue on the table.")
        else:
            st.warning("🟡 **Average ESG Profile:** You are performing at industry standard. Shifting budget into Environmental and Social pillars will systematically decrease risk and boost the revenue multiplier of your current inputs.")