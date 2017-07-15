Analysis based on the Kaggle Dataset from https://www.kaggle.com/dalpozz/creditcardfraud

Features information had already been transformed by the data owner through PCA

After doing some EDA, it was quickly apparent that the data needed to be engineered further, namely rescaled and a technique needed to be chosen to deal with the imbalance between the amount of labels for fraud (there was too few) which would throw off machine learning algorithms.  The approach I chose was to randomly eliminate some of the negative outcomes bringing the dataset closer to 50:50 fraud/not fraud
I trained and applied a logistic regression model to the test set and came out with a model that was acceptable.  

I then tried another well-known classification model, KNN to try and predict fraud.  In preparing this model's creation, I also including some code to attempt to brute force an optimal K value and demonstrate this trade off graphically.  My model then chooses the optimized K to make predictions.  The model results were similar to the logreg model on under sampled data.

I was curious so I played with the dataset further by applying the trained model to the entire dataset, resulting in a very poor F1 value for fraud detection.

I next wanted to try the opposite approach to balancing out the imbalance between fraud and non fraud occurrences and bootstrap more fraud entries to bring the ratio to an acceptable 5:1 non fraud/fraud.  I bootstrapped the data from the original dataset and I again trained a logistic regression model on this data and the results had the same f1 score for fraud occurrences and a better f1 for non-fraud.

I became more curious about the PCA values so I used LASSO to attempt to find the 'most important' features. And I further looked graphically at the covariance between the features in the dataset.  I used Seaborn to display this matrix in some pretty colors.
