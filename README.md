# Customer Segmentation using Machine Learning

## Project Overview

This project applies unsupervised machine learning to segment customers based on annual income and spending behavior.

The project uses the Mall Customer Segmentation dataset containing 200 customer records. K-Means clustering is used as the primary clustering algorithm, while silhouette analysis is used to evaluate different cluster configurations.

An interactive Streamlit dashboard is included to explore customer segments, summary statistics, and customer-level information.

## Objectives

- Analyze customer income and spending behavior.
- Clean and prepare the dataset.
- Standardize numerical features.
- Determine a suitable number of clusters.
- Apply K-Means clustering.
- Evaluate clustering using silhouette scores.
- Create descriptive customer segments.
- Generate professional visualizations.
- Build an interactive Streamlit dashboard.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Streamlit
- Google Colab
- GitHub

## Dataset

The project uses the Mall Customer Segmentation Dataset.

Dataset information:

- 200 customer records
- Customer ID
- Genre
- Age
- Annual Income (k$)
- Spending Score (1-100)

The clustering model uses Annual Income (k$) and Spending Score (1-100) as its primary features.

## Machine Learning Workflow

Raw Dataset
-> Data Loading
-> Data Cleaning
-> Feature Selection
-> Feature Scaling
-> Cluster Evaluation
-> K-Means Clustering
-> Cluster Profiling
-> Segment Naming
-> Visualization
-> Streamlit Dashboard

## Data Preprocessing

The preprocessing stage includes loading the dataset, checking and removing duplicate records, selecting relevant numerical features, and standardizing the selected features using StandardScaler.

## Clustering

K-Means clustering is used to group customers with similar income and spending characteristics.

Model settings:

- Initialization: k-means++
- Random state: 42
- n_init: 10

Different cluster counts are evaluated before selecting the preferred configuration.

## Model Evaluation

Two evaluation methods are included:

### Elbow Curve

The Elbow Curve examines model inertia for different numbers of clusters and helps identify a suitable clustering range.

### Silhouette Score

Silhouette Score evaluates cluster separation and cohesion. Scores are calculated for multiple values of K, and the highest-scoring configuration is used as the preferred clustering solution.

## Customer Segments

The project generates descriptive segment names based on average income and spending behavior:

- High-Value Customers
- High-Income Low-Spenders
- Budget High-Spenders
- Budget Low-Spenders
- Average Customers

These are descriptive interpretations of the clustering results and are not supervised predictions.

## Visualizations

The project includes:

- Elbow Curve
- Silhouette Score Analysis
- Customer Cluster Visualization
- Income vs Spending analysis

## Streamlit Dashboard

The dashboard provides:

- Total customer count
- Average customer age
- Average annual income
- Average spending score
- Customer segment distribution
- Income vs spending visualization
- Segment summary
- Customer-level data
- Segment filtering

## Project Structure

customer-segmentation-ml/
    app.py
    requirements.txt
    README.md
    .gitignore
    data/
        Mall_Customers.csv
    notebooks/
        customer_segmentation.ipynb
    src/
        preprocessing.py
        clustering.py
        visualization.py
    outputs/
        dashboard_data.csv
        elbow_curve.png
        silhouette_analysis.png
        customer_clusters.png
    report/
        Customer_Segmentation_Project_Report.docx

## Installation

Clone the repository and install the required packages:

pip install -r requirements.txt

## Run the Dashboard

Use:

streamlit run app.py

## Key Learning Outcomes

This project demonstrates practical experience with:

- Data preprocessing
- Feature selection
- Feature scaling
- Unsupervised machine learning
- K-Means clustering
- Silhouette analysis
- Customer profiling
- Data visualization
- Streamlit dashboard development
- Modular Python project organization
- GitHub project management

## Limitations

This analysis is based on a relatively small and static customer dataset. It uses only a limited set of customer characteristics and does not include historical transactions, purchase frequency, product-level information, or time-series behavior.

Therefore, the resulting clusters should be interpreted as exploratory customer segments rather than definitive business classifications.

## Future Improvements

Possible future improvements include:

- Larger real-world datasets
- Additional customer behavior features
- Alternative clustering algorithms
- Automated model comparison
- Database integration
- Real-time data processing
- Cloud deployment and monitoring

## Author

Sahil Kumar

Machine Learning and Data Analytics Project

## License

This project is intended for educational, portfolio, and demonstration purposes.
