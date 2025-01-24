# Student Sleep Analysis Using Big Data (Spark & Hadoop)

This project analyzes the sleep patterns of students to predict and classify healthy sleep behavior using Big Data tools. By leveraging Spark MLlib and Hadoop, we process and analyze large datasets efficiently to gain insights into student well-being.

## Key Features

- **Data Processing**: Utilizes Hadoop for storing and processing sleep data at scale.  
- **Feature Engineering**: Sleep hours are analyzed, and a binary label is created:
  - `1`: Sleep hours ≥ 8 (Healthy sleep)
  - `0`: Sleep hours < 8 (Insufficient sleep)  
- **Modeling**: Implements Gradient Boosted Trees (GBT) using Spark MLlib to classify and predict sleep behaviors.  
- **Big Data Integration**: Combines Spark and Hadoop to handle large-scale datasets, ensuring scalability and efficiency.  

## Objectives

- Classify students based on sleep hours to identify those with healthy or unhealthy sleep patterns.  
- Provide actionable insights into student sleep behaviors for further research or applications in well-being initiatives.

## Tools & Technologies

- **Hadoop**: Distributed data storage and processing.  
- **Apache Spark**: Machine learning and data analysis using Spark MLlib.  
- **Python/Scala**: Implementation of data pipelines and machine learning models.

## Usage

1. Preprocess sleep data using Hadoop for distributed storage.  
2. Use Spark to create and transform datasets for analysis.  
3. Train Gradient Boosted Trees for classification tasks.  
4. Evaluate model performance and predict sleep behavior labels.
