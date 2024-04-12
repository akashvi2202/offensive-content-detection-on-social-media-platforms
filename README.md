# offensive-content-detection-on-social-media-platforms
Project Methodology:
1. Data Collection:
Datasets Used:
Kaggle Hate Speech and Offensive Language Dataset: Annotated tweets for hate speech and offensive language.
Faker Generated Synthetic Dataset: Synthetic tweets mimicking real-world data generated using the faker library.
Hybrid Dataset (Faker-Kaggle Dataset): Synthetic tweets closely resembling Kaggle data annotated for hate speech and offensive content.

3. Data Preprocessing:
Balancing Dataset: Employed techniques like undersampling and oversampling to balance the dataset, reducing bias and improving representation.
Text Preprocessing Steps:
Tokenization: Divided input text into discrete tokens.
Stop Words Removal: Eliminated common but irrelevant words to enhance model performance.
Lowercasing: Standardized text to lowercase to ensure uniformity.
Lemmatization: Reduced words to their base form to capture core meaning.

5. Feature Extraction:
TF-IDF (Term Frequency-Inverse Document Frequency): Represented text samples as vectors, highlighting term importance based on frequency.
One-Hot Encoding: Converted documents into binary vectors to indicate word presence in the vocabulary.

7. Model Evaluation:
Machine Learning (ML) Models Used:
Logistic Regression
Naive Bayes
Random Forest
Bagging
AdaBoost
Deep Learning (DL) Models Used:
Artificial Neural Networks (ANN)
Convolutional Neural Networks (CNN)
Long Short-Term Memory (LSTM)
Bidirectional LSTM (BiLSTM)

9. Performance Evaluation:
Training and Testing Data Split: Used a 70%-30% split for training and testing data.
Evaluation Metrics: Assessed models based on accuracy, precision, recall, and F1 score across different datasets.
Comparative Analysis: Compared performance of ML and DL models on Kaggle hate speech dataset, Faker-generated dataset, and Faker-Kaggle hybrid dataset.

Results and Discussion:
Performance Metrics:
Kaggle Hate Speech Dataset:

ML Models (e.g., Adaboost) achieved high accuracies due to imbalanced class handling.
DL Models (e.g., CNN) excelled in capturing local patterns in tweets.
Faker Generated Dataset:

ML Models (e.g., Random Forest) demonstrated robust generalization in synthetic data.
DL Models (e.g., ANN) performed well in learning complex patterns.
Faker-Kaggle Hybrid Dataset:

ML Models (e.g., Random Forest) continued to perform well due to versatility in handling imbalanced data.
DL Models (e.g., CNN, LSTM) exhibited competitive accuracy in capturing patterns.
Conclusion:
The project highlighted the importance of selecting appropriate algorithms based on dataset characteristics. ML models like Adaboost excelled with imbalanced classes, while DL models like CNN and LSTM were effective in capturing complex patterns. The implementation of a predictive and block system enhanced the practical utility of offensive content detection.
