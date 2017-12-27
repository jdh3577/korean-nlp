# nlp-kor-machine-learning
Korean text classification using conventional machine learning methods

-Word Vector
  - Bag-of-Words (CountVectorizer, HashingVectorizer)
  - TF-IDF

-Classifier Methods
  - Naive Bayes (BernoulliNB, MultinomialNB)
  - SGDClassifier
  - KNeighborsClassifier
  - NearestCentroid
  - SVM (LinearSVC)
  - RandomForest
  - PassiveAggressiveClassifier
  - Perceptron
  - RidgeClassifier
  - Ensemble (hyperparameter random search)
    - hard voting: RandomForest, Naive Bayes, SVM, XGBoost, Ridge
    - soft voting: RandomForest, Naive Bayes, XGBoost
