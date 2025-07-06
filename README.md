# ChatGPT Review Sentiment Analysis  
This project uses Natural Language Processing (NLP) techniques to analyze and classify user sentiment from reviews of the ChatGPT Android application.

## STAR Methodology

### Situation  
- Started with three Kaggle datasets of ChatGPT app reviews (US region)  
- Exploratory analysis revealed severe class imbalance:  
  - Overwhelming majority of positive reviews (4-5 star ratings)  
  - Limited negative reviews  
- Risked training a model biased toward majority class with poor negative feedback identification  

### Task  
Build a text classification model to accurately categorize reviews as:  
- **Positive (1)**  
- **Negative (0)**  

Key objectives:  
1. Clean and preprocess review text  
2. Mitigate class imbalance  
3. Vectorize text for ML models  
4. Train/compare classification algorithms  
5. Optimize and validate final model on unseen data  

### Action  
Implemented an NLP pipeline:  

#### Data Loading and Consolidation  
- Merged low-rating reviews (<3 stars) from validation/test datasets into training data  
- Removed 3-star reviews to sharpen positive/negative distinction  

#### Advanced Text Preprocessing  
Applied transformations to review content:  
1. Lowercase conversion  
2. Punctuation removal  
3. Abbreviation expansion (e.g., "btw" → "by the way")  
4. Stopword removal using NLTK  

#### Intelligent Emoji Handling  
- Removed emojis when accompanied by text  
- Converted emoji-only reviews to text descriptions (e.g., 👍 → "thumbs up")  

#### Vectorization and Modeling  
- Transformed text using `TfidfVectorizer` (limited to 5000 features)  
- Trained/evaluated multiple classifiers:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - XGBoost  

#### Optimization and Final Validation  
1. Selected `RandomForestClassifier` for performance-interpretability balance  
2. Adjusted decision threshold from 0.5 → 0.6 to improve negative-class recall  
3. Validated against completely new test dataset (unused in training/tuning)  

### Result  
Achieved a robust sentiment classifier:  
- **95.6% accuracy** on unseen test data (149k+ reviews)  
- **72% recall for negative class** (significant improvement from threshold adjustment)  
- Confusion matrix confirms strong predictive capability for both classes  
- Model demonstrates generalization ability with no overfitting  
- Ready for deployment or adaptation to similar contexts  

The model's performance on completely new data demonstrates that it is not overfitted and that the preprocessing and imbalance mitigation techniques were successful. The model is ready for potential deployment or for application in other similar contexts.

