# Banknote Authentication Project

This project uses machine learning to identify whether banknotes are real or fake. We use two methods, K-Nearest Neighbors (KNN) and Support Vector Machines (SVM), to classify the banknotes.

## Steps in the Project

### 1. Preparing the Data
   - The banknote features are loaded from a file into a table.
   - The data is split into 80% for training the models and 20% for testing them.

### 2. Data Processing
   - The features of the banknotes are standardized. This helps in making our models work better.

### 3. Model Training
   - **KNN Model:** This model looks at the 5 closest points to decide if a banknote is real or fake.
   - **SVM Model:** This model uses a technique called Support Vector Machines for classification.

### 4. Predictions
   - Both models predict on the test data.
   - We also see how the models perform on a new sample of data.

### 5. Evaluating the Models
   - We use confusion matrices to see how well our models are doing.
   - We also calculate accuracy, precision, and recall for a detailed evaluation.

### 6. Visualizing Results
   - We plot the confusion matrices to visually inspect model performance.

## Key Tools Used
- **Pandas and NumPy:** For data handling.
- **Matplotlib:** For plotting graphs.
- **Scikit-learn:** For building and evaluating the models.


## Conclusion
This simple setup helps in classifying banknotes as real or fake using machine learning, demonstrating basic but powerful techniques in security and authentication.

