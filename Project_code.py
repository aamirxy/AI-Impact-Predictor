# ======================
# AI IMPACT PREDICTOR - FINAL POLISHED VERSION
# ======================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap

# Set up the page
st.set_page_config(
    page_title="AI Impact Predictor Pro",
    page_icon="🤖",
    layout="wide"
)

# Add a title and introduction
st.title("🤖 AI Impact Predictor Pro")
st.markdown("""
**Understand how AI will impact your business**  
This tool predicts AI's effect on revenue and jobs using real-world data from multiple industries and countries.
""")

# Load and prepare data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Global_AI_Content_Impact_Dataset.csv')
        
        # Clean percentage columns
        percent_cols = ['AI Adoption Rate (%)', 'Job Loss Due to AI (%)', 
                       'Revenue Increase Due to AI (%)', 'Human-AI Collaboration Rate (%)']
        for col in percent_cols:
            if df[col].dtype == object:
                df[col] = df[col].str.replace('%', '').astype(float)
        
        # Calculate efficiency score (revenue gain per 1% job loss)
        df['Efficiency Score'] = df['Revenue Increase Due to AI (%)'] / (df['Job Loss Due to AI (%)'] + 1)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Only proceed if data loaded successfully
if not df.empty:
    # Train model
    @st.cache_resource
    def train_models(_df):
        features = ['AI Adoption Rate (%)', 'Human-AI Collaboration Rate (%)', 
                   'Job Loss Due to AI (%)', 'Industry', 'Regulation Status']
        target = 'Revenue Increase Due to AI (%)'
        
        X = pd.get_dummies(_df[features])
        y = _df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(model)
        return model, explainer, X.columns

    model, explainer, feature_names = train_models(df)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Predictor", "Tool Analysis", "Country Trends", "Yearly Patterns"])

    with tab1:
        st.header("📈 Custom Impact Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            industry = st.selectbox("Industry", sorted(df['Industry'].unique()))
            adoption = st.slider(
                "AI Adoption Level (%)", 
                min_value=0,
                max_value=100,
                value=50,
                help="Percentage of processes using AI"
            )
            
        with col2:
            collaboration = st.slider(
                "Human-AI Collaboration Score", 
                min_value=0,
                max_value=100,
                value=50,
                help="How effectively your team works with AI (0-100)"
            )
            regulation = st.selectbox(
                "Regulation Environment",
                options=sorted(df['Regulation Status'].unique()),
                help="Government oversight of AI in your region"
            )
        
        if st.button("Calculate AI Impact", help="Generate predictions based on your inputs"):
            # Calculate job impact considering all factors
            industry_avg = df[df['Industry']==industry].agg({
                'Job Loss Due to AI (%)': 'median',
                'Revenue Increase Due to AI (%)': 'median',
                'Human-AI Collaboration Rate (%)': 'median'
            })
            
            regulation_factors = {
                'Lenient': 1.2,
                'Moderate': 1.0,
                'Strict': 0.8
            }
            
            # Job impact formula
            base_impact = industry_avg['Job Loss Due to AI (%)']
            adoption_factor = adoption / 100
            regulation_factor = regulation_factors.get(regulation, 1.0)
            collaboration_benefit = 1 - (collaboration / 200)  # 100% collaboration halves the impact
            
            job_impact = -base_impact * adoption_factor * regulation_factor * collaboration_benefit
            
            # Prepare model input
            input_data = {
                'AI Adoption Rate (%)': adoption,
                'Human-AI Collaboration Rate (%)': collaboration,
                'Job Loss Due to AI (%)': abs(job_impact),  # Model expects positive value
                'Industry': industry,
                'Regulation Status': regulation
            }
            
            input_df = pd.DataFrame([input_data])
            input_processed = pd.get_dummies(input_df).reindex(columns=feature_names, fill_value=0)
            
            # Make predictions
            revenue_pred = model.predict(input_processed)[0]
            shap_values = explainer.shap_values(input_processed)
            
            # Display results
            st.subheader("📊 Prediction Results")
            
            # Metrics with explanations
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Expected Revenue Change",
                    f"+{revenue_pred:.1f}%",
                    help="Predicted annual revenue growth from AI adoption",
                    delta_color="normal"
                )
                st.markdown(f"""
                **Industry Comparison:**  
                • Average: +{industry_avg['Revenue Increase Due to AI (%)']:.1f}%  
                • Top 25%: +{df[df['Industry']==industry]['Revenue Increase Due to AI (%)'].quantile(0.75):.1f}%
                """)
                
            with col2:
                st.metric(
                    "Estimated Job Impact",
                    f"{job_impact:.1f}%",
                    help="Net change in workforce requirements (negative means reduction)",
                    delta_color="inverse"
                )
                st.markdown(f"""
                **Industry Comparison:**  
                • Average: -{industry_avg['Job Loss Due to AI (%)']:.1f}%  
                • Top 25%: -{df[df['Industry']==industry]['Job Loss Due to AI (%)'].quantile(0.75):.1f}%
                """)
            
            # Impact breakdown
            st.subheader("🔍 Impact Breakdown")
            with st.expander("How these predictions are calculated"):
                st.markdown(f"""
                **Job Impact Formula:**  
                `Base Impact × Adoption % × Regulation Factor × Collaboration Benefit`  
                
                - **Base Impact:** {base_impact:.1f}% (industry median)  
                - **Your Adoption:** {adoption}% → ×{adoption_factor:.2f}  
                - **Regulation ({regulation}):** ×{regulation_factor:.2f}  
                - **Collaboration ({collaboration}/100):** ×{collaboration_benefit:.2f}  
                """)
                
                st.markdown("""
                **Good to know:**  
                • Higher collaboration scores significantly reduce job impacts  
                • Strict regulation slows both gains and disruptions  
                • The most successful companies balance high adoption with strong collaboration
                """)
            
            # Key influencing factors
            st.subheader("📌 Key Influencing Factors")
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Impact': shap_values[0]
            }).sort_values('Impact', key=abs, ascending=False).head(3)
            
            for _, row in feature_importance.iterrows():
                feature_name = row['Feature']
                impact = row['Impact']
                
                if "Industry" in feature_name:
                    desc = "Your Industry"
                    detail = f"The {industry} sector typically sees:"
                    stats = df[df['Industry']==industry].agg({
                        'Revenue Increase Due to AI (%)': ['mean', 'std'],
                        'Job Loss Due to AI (%)': ['mean']
                    })
                elif "Adoption" in feature_name:
                    desc = "AI Adoption Level"
                    detail = "Every 10% increase in adoption:"
                    stats = "+{:.1f}% revenue impact".format(impact * 10)
                elif "Collaboration" in feature_name:
                    desc = "Team Collaboration"
                    detail = "Collaboration benefits:"
                    stats = "Reduces job impact by {:.1f}%".format(abs(impact) * 10)
                elif "Regulation" in feature_name:
                    desc = "Regulation Level"
                    detail = f"{regulation} regulation tends to:"
                    stats = df[df['Regulation Status']==regulation].agg({
                        'Revenue Increase Due to AI (%)': ['mean'],
                        'Job Loss Due to AI (%)': ['mean']
                    })
                
                with st.expander(f"{desc} → {impact:.1f}% impact"):
                    st.markdown(f"{detail}")
                    if isinstance(stats, pd.DataFrame):
                        st.dataframe(stats.style.format("{:.1f}%"))
                    else:
                        st.markdown(stats)

    with tab2:
        st.header("🛠️ AI Tools Comparison")
        
        # Tool performance metrics
        tool_stats = df.groupby('Top AI Tools Used').agg({
            'Revenue Increase Due to AI (%)': ['mean', 'count'],
            'Job Loss Due to AI (%)': 'mean',
            'Human-AI Collaboration Rate (%)': 'mean',
            'Efficiency Score': 'mean'
        }).sort_values(('Efficiency Score', 'mean'), ascending=False)
        
        st.markdown("""
        ### Tool Performance Metrics
        **Efficiency Score** = (Revenue Increase) / (Job Loss + 1)  
        Higher scores indicate better results with less workforce disruption
        """)
        
        st.dataframe(
            tool_stats.style.format({
                ('Revenue Increase Due to AI (%)', 'mean'): "{:.1f}%",
                'Job Loss Due to AI (%)': "{:.1f}%",
                'Human-AI Collaboration Rate (%)': "{:.1f}%",
                'Efficiency Score': "{:.2f}"
            }),
            use_container_width=True,
            height=600
        )
        
        # Visual comparison
        st.markdown("""
        ### Tool Performance Visualization
        """)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x='Revenue Increase Due to AI (%)',
            y='Job Loss Due to AI (%)',
            hue='Top AI Tools Used',
            size='Human-AI Collaboration Rate (%)',
            sizes=(50, 200),
            alpha=0.7,
            ax=ax
        )
        ax.set_title("Revenue vs Job Impact by AI Tool")
        st.pyplot(fig)
        
        st.markdown("""
        **How to read this:**  
        • Ideal tools are in the top-left (high revenue, low job impact)  
        • Bubble size shows collaboration rates  
        • Color indicates different tools
        """)

    with tab3:
        st.header("🌍 Country Comparison")
        
        # Country performance metrics
        country_stats = df.groupby('Country').agg({
            'Revenue Increase Due to AI (%)': 'mean',
            'Job Loss Due to AI (%)': 'mean',
            'AI Adoption Rate (%)': 'mean',
            'Human-AI Collaboration Rate (%)': 'mean',
            'Regulation Status': lambda x: x.mode()[0],
            'Top AI Tools Used': lambda x: x.mode()[0]
        }).sort_values('Revenue Increase Due to AI (%)', ascending=False)
        
        st.markdown("""
        ### National AI Adoption Metrics
        """)
        st.dataframe(
            country_stats.style.format({
                'Revenue Increase Due to AI (%)': "{:.1f}%",
                'Job Loss Due to AI (%)': "{:.1f}%",
                'AI Adoption Rate (%)': "{:.1f}%",
                'Human-AI Collaboration Rate (%)': "{:.1f}%"
            }),
            use_container_width=True,
            height=600
        )
        
        # Regulation impact analysis
        st.markdown("""
        ### Regulation Impact Analysis
        """)
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.boxplot(
            data=df,
            x='Regulation Status',
            y='Revenue Increase Due to AI (%)',
            ax=ax[0]
        )
        ax[0].set_title("Revenue Impact by Regulation")
        
        sns.boxplot(
            data=df,
            x='Regulation Status',
            y='Job Loss Due to AI (%)',
            ax=ax[1]
        )
        ax[1].set_title("Job Impact by Regulation")
        
        st.pyplot(fig)
        
        st.markdown("""
        **Key Findings:**  
        • **Lenient regulation:** Higher revenue potential but greater job disruption risk  
        • **Strict regulation:** More stable employment but slower revenue growth  
        • **Moderate regulation:** Best balance for most industries
        """)

    with tab4:
        st.header("📅 Yearly Trends")
        
        # Global trends
        yearly_stats = df.groupby('Year').agg({
            'Revenue Increase Due to AI (%)': 'mean',
            'AI Adoption Rate (%)': 'mean',
            'Job Loss Due to AI (%)': 'mean'
        })
        
        st.markdown("""
        ### Global AI Adoption Trends
        """)
        fig, ax = plt.subplots(figsize=(10, 5))
        yearly_stats.plot(ax=ax, marker='o')
        ax.set_title("AI Impact Evolution (2020-2025)")
        ax.set_ylabel("Percentage Change")
        st.pyplot(fig)
        
        st.markdown(f"""
        **Annual Growth Rates:**  
        • Adoption: +{(yearly_stats['AI Adoption Rate (%)'].iloc[-1] - yearly_stats['AI Adoption Rate (%)'].iloc[0])/5:.1f}% per year  
        • Revenue Impact: +{(yearly_stats['Revenue Increase Due to AI (%)'].iloc[-1] - yearly_stats['Revenue Increase Due to AI (%)'].iloc[0])/5:.1f}% per year  
        • Job Impact: Changed by {(yearly_stats['Job Loss Due to AI (%)'].iloc[-1] - yearly_stats['Job Loss Due to AI (%)'].iloc[0])/5:.1f}% per year
        """)
        
        # Industry-specific trends
        selected_industry = st.selectbox(
            "Select Industry for Detailed Trends",
            options=sorted(df['Industry'].unique()),
            key='industry_trend_selector'
        )
        
        industry_data = df[df['Industry']==selected_industry]
        
        st.markdown(f"""
        ### {selected_industry} Sector Trends
        """)
        fig, ax = plt.subplots(figsize=(10, 5))
        industry_data.groupby('Year').agg({
            'Revenue Increase Due to AI (%)': 'mean',
            'AI Adoption Rate (%)': 'mean'
        }).plot(ax=ax, marker='o')
        ax.set_title(f"{selected_industry} AI Adoption & Impact")
        st.pyplot(fig)
        
        st.markdown(f"""
        **Current Status ({df['Year'].max()}):**  
        • Adoption Rate: {industry_data['AI Adoption Rate (%)'].mean():.1f}%  
        • Revenue Impact: +{industry_data['Revenue Increase Due to AI (%)'].mean():.1f}%  
        • Most Used Tool: {industry_data['Top AI Tools Used'].mode()[0]}
        """)

    # Final summary
    st.markdown("""
    ---
    ### How to Use These Insights
    1. **Start with predictions** - Get customized estimates for your situation
    2. **Compare tools** - See which AI tools deliver the best results
    3. **Learn from countries** - Find regions with similar regulatory environments
    4. **Track trends** - Understand how AI impacts evolve over time
    
    **Remember:** These are data-driven estimates based on patterns in the dataset.  
    Your actual results may vary based on implementation quality and other factors.
    """)

else:
    st.error("Please check your dataset and try again. The application cannot run without proper data.")