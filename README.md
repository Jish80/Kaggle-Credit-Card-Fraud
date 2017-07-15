Analysis based on the Kaggle Dataset from https://www.kaggle.com/dalpozz/creditcardfraud

Data is has already been transfored through PCA
After doing some EDA, it was quickly apparent that the data needed to be engineered further, namely rescaled and a technique needed to be chosen to deal with the imbalance between the amount of labels for fraud (there was too few) which would throw off machine learning algorithms.  The approach I chose was to randomly eliminate some of the negative outcomes bringing the dataset closer to 50:50
I applied logistic regression and came out with a model that was ok prediction wise.  I further played with the dataset applying LASSO on the principle componants to see which were the most important.  Further I constructed another classification model to see if I could get better results than logreg, in this case I used KNN

