## COMP-551
Course Work for Applied Machine Learning.

Reddit Comment Popularity Score Predictor

In this paper we make use of a linear regression model using both closed form and gradient descent methods to estimate the popularity score of a Reddit post. We trained our model on shallow bag of common words (n-BOW), and other features. We start the analysis with varying lengths of bag of words as well as including interaction terms and transforming the feature space. Additionally, we explored various common text-processing practices such as bigrams, trigrams, character n-grams, and stopword removal. We compare these model’s performance on the validation sets based on their mean squared error and adjusted R2 coefficients. We remark that our model is not overfitting nor underfitting, because we had a lower MSE for our validation set than training set, while having a reasonable number of parameters. We find that using 67-BOW plus additional features and interactions helped us achieve the lowest MSE of 0.960 and highest adjusted R2 of 0.264 under the closed form solution on the validation set; and had an MSE of 1.298, adjusted R2 of 0.186 when run on the test set. Finally, we compare the performance of the gradient descent and closed form solutions and find that the gradient descent is less stable and slower on average than the closed form solution.




