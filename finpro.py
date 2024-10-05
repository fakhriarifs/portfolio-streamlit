import streamlit as st
st.title('Superstore Customer Dataset')
st.write('Superstore Customer Segmentation for Future Marketing Campaigns')

import pandas as pd
df = pd.read_csv('marketing_campaign.csv', sep=';')
df = df.drop(columns='ID')
dfa = df.dropna()
dfa = dfa.drop_duplicates()
dfa['Dt_Customer'] = pd.to_datetime(dfa['Dt_Customer'], dayfirst=True)
today = pd.to_datetime('2014-10-04')
dfa['Age'] = today.year - dfa['Year_Birth']
dfa['Member_Period'] = (today - dfa['Dt_Customer']).dt.days
dfa = dfa.drop(columns=['Year_Birth', 'Dt_Customer'])
dfa = dfa.drop(columns=['Z_CostContact', 'Z_Revenue'])
st.write('This is the dataset of customer that will be used:')
st.write(dfa)

education_mapping = {
    'Basic': 0,
    'Graduation': 1,
    'Master': 2,
    '2n Cycle': 2,
    'PhD': 3
}

dfa['Education'] = dfa['Education'].map(education_mapping)

marital_status_mapping = {
    'Single': 0,
    'Together': 1,
    'Married': 2,
    'Divorced': 3,
    'Widow': 4,
    'Alone': 5,
    'Absurd': 6,
    'YOLO': 7
}

dfa['Marital_Status'] = dfa['Marital_Status'].map(marital_status_mapping)
num = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

dfb = dfa.copy()
scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(dfb)

kmeans = KMeans(n_clusters=4, random_state=42).fit(scaled_data)

df_scaled = pd.DataFrame(scaled_data, columns=dfb.columns)

df_scaled['clusters'] = kmeans.labels_
dfb['clusters'] = kmeans.labels_

grouped_summary = dfb.groupby('clusters').agg(['mean', 'median'])

import plotly.express as px

st.title("Customer Clustering")

# Tabs for visualizations
tab1, tab2 = st.tabs(["Visualizations", "Data Summary"])

with tab1:
    # Grouping the DataFrame by clusters and calculating the mean for each feature
    cluster_summary = dfb.groupby('clusters').mean().reset_index()

    # Select box for choosing a column for visualization
    metric = st.selectbox("Select a Metric to Visualize:", cluster_summary.columns[1:])  # Exclude 'clusters' column

    # Radio button to choose the type of chart
    chart_type = st.radio("Select Chart Type:", ("Bar Chart", "Pie Chart"))

    if chart_type == "Bar Chart":
        # Bar chart for the selected metric by cluster
        fig1 = px.bar(
            cluster_summary, 
            x='clusters', 
            y=metric, 
            title=f'Average {metric} by Cluster',
            labels={'clusters': 'Cluster', metric: f'Average {metric}'},
            color='clusters',
            text=metric
        )
        st.plotly_chart(fig1)

    elif chart_type == "Pie Chart":
        # Pie chart for the selected metric
        fig2 = px.pie(
            cluster_summary, 
            names='clusters', 
            values=metric, 
            title=f'Average {metric} Proportion by Cluster',
            labels={'clusters': 'Cluster', metric: f'Proportion of Average {metric}'}
        )
        st.plotly_chart(fig2)

with tab2:
    # Data Summary for each cluster
    st.write("Data Summary by Cluster:")

    # Grouping the DataFrame by clusters and calculating the mean for each feature
    cluster_summary = dfb.groupby('clusters').mean().reset_index()
    st.write(cluster_summary)

    # Optional: If you want to provide the option to view raw data per cluster
    cluster_selection = st.selectbox("Select a Cluster to View Raw Data:", dfb['clusters'].unique())
    
    # Filter the DataFrame for the selected cluster
    cluster_data = dfb[dfb['clusters'] == cluster_selection]
    
    st.write(f"Raw Data for Cluster {cluster_selection}:")
    st.write(cluster_data)