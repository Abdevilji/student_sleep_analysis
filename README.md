# Student Sleep Analysis Using Big Data (Spark & Hadoop)

This project leverages Big Data tools and machine learning techniques to analyze student sleep patterns, classify healthy and unhealthy sleep behaviors, and predict future trends. By utilizing Hadoop for data processing and Apache Spark MLlib for machine learning, the project handles large-scale datasets efficiently, providing actionable insights into student well-being.

## Key Features

- **Data Processing**: Hadoop is used for storing and processing large-scale sleep data.  
- **Feature Engineering**:  
  - The dataset was transformed to include a binary label:  
    - `1`: Sleep hours ≥ 8 (Healthy sleep).  
    - `0`: Sleep hours < 8 (Insufficient sleep).  
- **Machine Learning Models**: Implemented multiple machine learning techniques, including:  
  - **Gradient Boosted Trees (GBT)** for high-performance classification.  
  - Other classification models such as Logistic Regression and Decision Trees for comparative analysis.  
- **Scalable Architecture**: Combines the power of Hadoop and Spark to manage and process large datasets efficiently.  

## Objectives

- Analyze student sleep data to identify patterns and classify sleep behavior.  
- Provide a predictive model to assess the impact of sleep habits on well-being.  
- Support further research and interventions aimed at improving student health.

## Tools & Technologies

- **Hadoop**: Distributed data storage and processing.  
- **Apache Spark**: Machine learning and data analysis using Spark MLlib.  
- **Python/Scala**: For implementing data pipelines and machine learning models.

## Model Performance

The evaluation of various machine learning models, including **Gradient Boosted Trees**, **Logistic Regression**, and **Decision Trees**, is documented in detail. Metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC** are included to compare model performance.  

For a comprehensive view of the results and visualizations, refer to the **Performance Analysis PPT** uploaded to this repository.  

## Usage

1. Preprocess and store raw sleep data using Hadoop.  
2. Use Spark for feature engineering and data transformation.  
3. Train and evaluate machine learning models to classify sleep behavior.  
4. Compare model performance and fine-tune the best-performing model.  

## Results

The machine learning models demonstrated reliable performance, with Gradient Boosted Trees achieving the highest accuracy among all methods. Detailed insights, including the impact of sleep habits on health, are outlined in the accompanying presentation.
