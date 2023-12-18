# Project Title
Medical Chatbot for Disease Prediction

**Prepared for:**
UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang

**Author Name:**
Sai Thanmai Peddader Pally

**Author's Links:**
- [GitHub Profile](https://github.com/DATA-606-2023-FALL-TUESDAY/Peddader-Pally_Sai-Thanmai)
- [LinkedIn Profile](www.linkedin.com/in/sai-thanmai-peddader-pally-8110721b6)

**Additional Links:**
- [PowerPoint Presentation](Medical%20Chatbot%20ppt.pptx)
- [YouTube Video] (https://youtu.be/KvKgJPZnQmg) 

## 1. What is it about?
The project centers around the development of a healthcare chatbot leveraging machine learning. The primary focus is on utilizing a dataset containing information about various diseases and their associated symptoms. The dataset, obtained from Kaggle, consists of two main files: "testing.csv" and "training.csv," with the latter containing data on 41 diseases and their corresponding symptoms.

### 1.1 Why does it matter?
Healthcare chatbots and symptom checkers play a crucial role in providing accessible and preliminary healthcare guidance. With the increasing popularity of telemedicine, such tools serve to assist individuals in assessing symptoms and identifying potential health issues. Moreover, efficient symptom checkers can support healthcare professionals during patient consultations by providing additional context and relevant information.

### 1.2 What are your research questions?
1. **Effectiveness of Machine Learning Model:**
   - How effective is the machine learning model trained on this dataset in accurately identifying diseases based on reported symptoms?
2. **User-Friendliness Optimization:**
   - How can the chatbot be optimized for user-friendliness and ease of understanding for individuals with varying levels of medical knowledge?

## 2. Data Sources:
The dataset is obtained from Kaggle, consisting of two primary files: "testing.csv" and "training.csv."

### 2.1 Data Details:
- Data Size: 30KB
- Data Shape: 133 Columns, 4921 Rows
- Time Period: Not specified

### 2.2 Each Row Represents:
The values in the dataset represent whether a specific symptom is associated with a disease.

### 2.3 Data Dictionary:
- Columns Name: Symptoms names (132 columns)
- Data Type: Binary (1 for symptom presence, 0 for absence)
- Definition: The dataset reflects the presence or absence of symptoms in each training example, aiming to associate symptoms with disease names.
- Potential Values: Categorical variables (symptoms) are binary, with '1' indicating symptom presence and '0' indicating symptom absence.

### 2.4 Target/Label:
- Target/Label Variable: Disease (prognosis) column

### 2.5 Features/Predictors:
- Variables/Columns: 132 symptom columns (symptom names)
- Data Type: Binary (1 for symptom presence, 0 for absence)
- Use in ML Model:
  - Input (X): Symptom columns
  - Output (y): Disease (prognosis) column
  - This structure allows for training a machine learning model to predict diseases based on the presence or absence of symptoms, where the column names represent the symptoms.

## 3. Data Exploration
The purpose of this data exploration is to analyze and understand the dataset for the development of a healthcare chatbot. The primary focus is on the target variable (Disease) and selected features (Symptoms). The key objectives include generating summary statistics, creating visualizations, and assessing the need for data cleansing.

The dataset, consisting of 4921 rows and 133 columns, is loaded for analysis. The initial assessment provides a foundation for subsequent exploration.

To streamline the dataset, all columns except the target variable (Disease) and symptom features are dropped. This ensures a more focused analysis.

Descriptive statistics are computed to gain insights into key variables, facilitating a better understanding of the dataset's characteristics.

## 4. Visualizations:
Visual representations using Plotly Express are employed to enhance comprehension.

### 4.1 Disease Distribution:
In this visualization, an equal distribution of every symptom in the dataset is observed, indicating a balanced representation of symptoms across diseases.

### 4.2 Symptom Frequency:
Fatigue emerges as the most common symptom in the dataset, as evident from the clear dominance in the bar chart, highlighting its prevalence.

### 4.3 Correlation Heatmap:
The heatmap illustrates the correlation between all symptoms in the dataset. The visual confirms the existence of correlations, offering a comprehensive overview of symptom relationships.

### 4.4 Correlation for Top 20 Symptoms:
The correlation heatmap for the top 20 symptoms distinctly shows correlations among them, providing focused insights into the relationships of the most frequent symptoms.

### 4.5 Grouped Bar Chart:
The grouped bar chart representing Symptoms Associated with Top 5 Diseases reveals that tuberculosis has the highest number of associated symptoms, offering valuable insights into prevalent symptom patterns for this disease.
# 5. Further Data Manipulation
An exploration is made to determine if additional data manipulation (splitting, merging, pivoting, melting) is required based on the current analysis.

## 5.1 Augmentation with Additional Data
The potential need for bringing in external data sources, such as population or socioeconomic data, is explored to enhance the dataset. The resulting dataset is validated to ensure it adheres to the principles of tidy data, where each row represents one unique observation, and each column represents one unique property of that entity. This comprehensive data exploration lays the groundwork for subsequent steps in the development of the healthcare chatbot.

# 6. Data Splitting
The dataset undergoes a split into training and testing sets using the `train_test_split` function from `sklearn.model_selection`. The features (X) comprise all columns except the target variable (prognosis), while the target variable (y) is derived from the prognosis column. The split ratio is set at 80% for training and 20% for testing, ensuring a balanced division for model evaluation.

## 6.1 Random Forest Model Training
For predictive analytics, the Random Forest Classifier is chosen due to its suitability for datasets with numerous features and resilience against overfitting. The classifier is configured with 100 decision trees (n_estimators=100) and trained using the designated training set.

## 6.2 Model Evaluation
The accuracy of the trained model is assessed using the testing set. The `score` method of the classifier quantifies the proportion of correctly predicted outcomes. In this case, the model exhibits a certain accuracy level in classifying diseases based on reported symptoms.

## 6.3 Overall Accuracy
The Random Forest model exhibits remarkable accuracy, achieving a perfect score of 1.0 on the testing set. This implies that every prediction made by the model aligns with the actual outcomes.

## 6.4 Classification Report
The detailed classification report provides insights into the model's performance across various disease categories. Here are key metrics for selected diseases:
| Disease | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| (vertigo) Paroymsal Positional Vertigo | 1.00 | 1.00 | 1.00 |
| AIDS | 1.00 | 1.00 | 1.00 |
| Acne | 1.00 | 1.00 | 1.00 |
| Alcoholic hepatitis | 1.00 | 1.00 | 1.00 |
| Allergy | 1.00 | 1.00 | 1.00 |
... (truncated for brevity)

## 6.5 Interpretation
- **Precision:** The model shows perfect precision (1.00) for all diseases, indicating that when it predicts a disease, it is correct.
- **Recall:** The recall values of 1.00 imply that the model identifies all instances of each disease in the test set.
- **F1-Score:** The harmonic mean of precision and recall is also perfect (1.00), demonstrating a balanced trade-off between precision and recall. The model's outstanding performance, as reflected in the accuracy and detailed classification report, suggests its effectiveness in accurately identifying diseases based on reported symptoms. The high precision and recall values indicate a robust and reliable predictive capability across various medical conditions.

# 7. Web App Development
## 7.1 Gradio Implementation
To enhance user accessibility and interaction with the trained Random Forest model, a web application has been developed using Gradio. Gradio simplifies the deployment of machine learning models by providing a user-friendly interface. The application allows users to input symptoms and receive predictions for possible diseases.

## 7.2 Implementation Steps
1. **Model Loading:** The trained Random Forest model (`random_forest_model.joblib`) is loaded into the web application.
2. **Symptom List Retrieval:** The list of symptoms is extracted from the dataset to create a user-friendly interface for symptom selection.
3. **Prediction Function:** A prediction function is defined to take user-selected symptoms as input, convert them into a binary input vector, and predict the associated disease using the loaded model.
4. **Gradio Interface:** Gradio is utilized to create a graphical interface where users can select symptoms through checkboxes. The interface dynamically adjusts based on the available symptoms.

## 7.3 User Interaction
Users can interact with the web app by selecting symptoms relevant to their condition. Upon submitting the selected symptoms, the model predicts the associated disease and presents the result on the interface. This interactive web app provides a seamless and intuitive way for individuals to obtain preliminary insights into potential health issues based on reported symptoms. The use of Gradio ensures simplicity and ease of use for a diverse audience with varying levels of medical knowledge.

# 8. Conclusion
## 8.1 Summary of Work
The project revolves around the development of a healthcare chatbot using a Random Forest model trained on a dataset containing information about various diseases and their associated symptoms. The machine learning model demonstrates exceptional accuracy in predicting diseases based on reported symptoms, achieving a perfect score on the testing set. The model is integrated into a web app using Gradio, providing a user-friendly interface for individuals to interact with the predictive capabilities.

## 8.2 Potential Applications
1. **Accessible Healthcare Guidance:** The healthcare chatbot serves as a preliminary guide for individuals to assess their symptoms and obtain initial insights into potential health issues.
2. **Support for Healthcare Professionals:** During patient consultations, the chatbot can provide additional context and support healthcare professionals in the decision-making process.
3. **Telemedicine Enhancement:** With the growing popularity of telemedicine, the chatbot enhances telehealth services by assisting individuals in evaluating symptoms remotely.

## 8.3 Limitations
1. **Binary Symptom Representation:** The model's reliance on binary symptom representation (present/absent) may oversimplify the complexity of symptom severity.
2. **Dataset Specificity:** The model's performance is contingent on the characteristics of the specific dataset used for training, limiting its generalizability to diverse populations.
3. **Web App Dependency:** The web app's effectiveness relies on user input and may not replace professional medical advice. It serves as a supportive tool rather than a definitive diagnostic tool.

## 8.4 Lessons Learned
1. **Feature Importance:** Understanding feature importance in the Random Forest model was crucial for interpreting the roles of symptoms in disease prediction.
2. **Web App Development:** Developing a user-friendly web app using Gradio enhanced the model's accessibility and usability for a wider audience.
3. **Model Evaluation:** Thorough model evaluation, including precision, recall, and F1-score, provided a comprehensive assessment of predictive performance.

## 8.5 Future Research Direction
1. **Enhanced Symptom Representation:** Investigate more nuanced symptom representation to capture severity and frequency, allowing for a more refined prediction model.
2. **Integration of Additional Data Sources:** Explore the integration of external data sources, such as demographic or environmental factors, to enhance predictive capabilities.
3. **Continuous Model Training:** Implement strategies for continuous model training to adapt to evolving medical knowledge and emerging diseases.
4. **User Feedback Integration:** Incorporate user feedback into the model to improve its accuracy and relevance over time.

In conclusion, while the current project demonstrates promising results in disease prediction and user interaction, ongoing research and improvements are essential to address limitations and enhance the
