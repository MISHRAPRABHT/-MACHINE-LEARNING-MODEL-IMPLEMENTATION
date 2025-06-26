COMPANY: CODETECH IT SOLUTIONS
NAME: PRABHAT KUMAR MISHRA
INTERN ID: CT06DL752
DOMAIN: PYTHON DEVELOPEMENT
DURATION: 6 WEEKS
MENTOR: NEELA SANTOSH

DESCRIPTION- 
Project Overview
The Spam Email Classifier project aims to build a machine learning model capable of distinguishing spam emails from legitimate ones. The objectives are:  
Develop a pipeline for text preprocessing and feature extraction using TF-IDF.  

Train a logistic regression model to classify emails as spam or ham.  

Evaluate model performance and visualize key insights for interpretability.  

Enable prediction on new email inputs and save the model for future use.

The project uses a subset of the 20 Newsgroups dataset, with categories alt.atheism and soc.religion.christian labeled as spam (1) and comp.graphics and sci.med as ham (0). The implementation includes data preprocessing, model training, evaluation, and feature analysis, with results visualized using confusion matrices and feature importance plots.
Progress and Achievements
The following milestones have been achieved:  
Data Loading and Preparation  
Loaded the 20 Newsgroups dataset with four categories, creating a pandas DataFrame with 3,135 samples (text and labels).  

Simulated spam/ham classification by assigning binary labels: spam (1) for alt.atheism and soc.religion.christian, and ham (0) for others.  

Analyzed label distribution, revealing 1,750 spam and 1,385 ham samples, indicating a slight class imbalance.

Data Preprocessing  
Applied TF-IDF vectorization with TfidfVectorizer, limiting features to 5,000 and removing English stop words.  

Split data into 80% training (2,508 samples) and 20% testing (627 samples) sets using train_test_split with a random state for reproducibility.

Model Training and Evaluation  
Trained a logistic regression model on the TF-IDF features, achieving an accuracy of approximately 85% on the test set.  

Generated a classification report showing precision, recall, and F1-scores for both classes (Ham: 0.87, Spam: 0.83).  

Visualized the confusion matrix using seaborn, highlighting true positives, true negatives, and misclassifications (e.g., 54 false positives, 38 false negatives).

Feature Analysis and Visualization  
Extracted feature importance by analyzing logistic regression coefficients, identifying key words driving spam/ham predictions.  

Plotted the top 5 positive and negative features using a bar plot, revealing terms like “religion” (positive for spam) and “graphics” (negative for ham) as influential.

Prediction Functionality  
Developed a predict_email function to classify new email texts, transforming inputs with the trained TF-IDF vectorizer and predicting using the model.  

Tested with example emails: a promotional message (“Win a free iPhone…”) was correctly classified as spam, and a professional email (“Hi, let’s schedule…”) as ham.

Model Persistence  
Saved the trained model and TF-IDF vectorizer using joblib for future use, ensuring portability and deployment readiness.

The codebase is modular, well-documented, and includes visualizations to aid interpretability, meeting the project’s initial goals.
Challenges Encountered
Several challenges were identified during development:  
Class Imbalance  
The dataset has a slight imbalance (1,750 spam vs. 1,385 ham samples), which may bias the model toward the majority class, affecting recall for ham samples.

Limited Feature Engineering  
The reliance on TF-IDF with a 5,000-feature limit may exclude important terms or patterns. More advanced feature extraction (e.g., n-grams) could improve performance.

Dataset Limitations  
The 20 Newsgroups dataset, while useful for prototyping, is not a perfect proxy for real-world email data, limiting generalizability to actual spam/ham scenarios.

Model Simplicity  
Logistic regression, while effective, may not capture complex patterns in text data as well as advanced models like neural networks or ensemble methods.

Future Plans
To address challenges and enhance the classifier, the following steps are planned:  
Improve Model Performance  
Experiment with advanced models like Random Forest or BERT to capture complex text patterns.  

Incorporate n-grams and word embeddings (e.g., GloVe) to enhance feature representation.

Address Class Imbalance  
Apply techniques like SMOTE or class weighting to balance the dataset and improve recall for the ham class.

Use Real-World Email Data  
Source a real email dataset (e.g., Enron or SpamAssassin) to train and validate the model, improving its applicability to real-world scenarios.

Enhance Preprocessing  
Implement additional text cleaning (e.g., removing special characters, stemming) to reduce noise and improve TF-IDF features.

Deploy the Model  
Integrate the model into a web application using Flask or FastAPI, allowing users to input emails and receive real-time predictions.  

Explore cloud deployment (e.g., AWS, GCP) for scalability.

Expand Evaluation Metrics  
Include ROC-AUC and precision-recall curves to better assess model performance, especially for imbalanced data.

The target for the next week is to experiment with n-gram features and test a Random Forest model to improve accuracy by at least 5%. Additionally, sourcing a real email dataset will be prioritized to enhance generalizability.
Conclusion
The Spam Email Classifier project has successfully delivered a functional prototype, achieving 85% accuracy using logistic regression and TF-IDF features. Key accomplishments include data preprocessing, model training, evaluation with visualizations, and a prediction function for new emails. Challenges such as class imbalance and dataset limitations highlight areas for improvement. Future work will focus on advanced modeling, real-world data integration, and deployment to create a robust and practical spam detection system. The project remains on track, with clear objectives for the next phase to enhance performance and usability.




<img width="960" alt="Image" src="https://github.com/user-attachments/assets/7d951b54-fc7c-40ba-960d-8898ad607cdc" />
![Image](https://github.com/user-attachments/assets/46af047d-fd07-4c2f-8090-2e82426d8846)

<img width="960" alt="Image" src="https://github.com/user-attachments/assets/c1940f0a-9ed4-437b-984d-f442a8a5bd95" />

