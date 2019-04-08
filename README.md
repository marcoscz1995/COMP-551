## COMP-551
Course Work for Applied Machine Learning.

Reddit Comment Popularity Score Predictor

In this paper we make use of a linear regression model using both closed form and gradient descent methods to estimate the popularity score of a Reddit post. We trained our model on shallow bag of common words (n-BOW), and other features. We start the analysis with varying lengths of bag of words as well as including interaction terms and transforming the feature space. Additionally, we explored various common text-processing practices such as bigrams, trigrams, character n-grams, and stopword removal. We compare these model’s performance on the validation sets based on their mean squared error and adjusted R2 coefficients. We remark that our model is not overfitting nor underfitting, because we had a lower MSE for our validation set than training set, while having a reasonable number of parameters. We find that using 67-BOW plus additional features and interactions helped us achieve the lowest MSE of 0.960 and highest adjusted R2 of 0.264 under the closed form solution on the validation set; and had an MSE of 1.298, adjusted R2 of 0.186 when run on the test set. Finally, we compare the performance of the gradient descent and closed form solutions and find that the gradient descent is less stable and slower on average than the closed form solution.

IMDB Sentiment Analysis

Sentiment analysis is an ongoing and widespread practice both in academia and industry. The need
for adequate text analyzers is an ongoing challenge. In this paper we approach a binary sentiment analysis by
predicting movie reviews sentiment as positive or negative on the IMDB Movie Review Dataset. We make
use of various feature extraction processes such as N-grams and term frequency * inverse document
frequency (TF*IDF). We then assess and compare the performance of Bernoulli Naive Bayes (BNB), support
vector machine (SVM), logistic regression (LREG), extreme gradient boosting (XGB), dense shallow neural
network (NN), and bidirectional LSTM (biLSTM) on this data set. Finally, we compare and analyze each
model’s performance on our sentiment analysis task.

Modified MNIST number prediction

In this paper we exploit ResNet's increased accuracy to predict images on the modied MNIST data set.
However, due to computational restrictions we are forced to look for novel ways to cut on computational
costs while maintaining a desired level of accuracy. This accuracy-cost trade-o is addressed by the SE-block
architectures that improve accuracy on shallower models at only a nominal increase in computational cost
(Hu et al, 2018). By including SE-blocks in our shallow networks we were able to achieve high levels of
accuracy comparable to deeper networks while using less computational power. In particular, we explore the
use of residual neural networks (ResNet 34, ResNet 50, ResNet 101, ResNet 152, Wide ResNet), Squeezeand-
Excitation Networks (SE-ResNet 56), and a custom 22-layer CNN for the objective. We found that the
SE-ResNet 56 has the greatest accuracy on the validation set, with an accuracy of 97.48%. The SE-ResNet
56 also achieved an accuracy of 97.56% on the test set. We also observed that the SE-ResNet converged
much more rapidly than the other models, with a validation accuracy of 91.64% after only 2 epochs.
Hu et al. (2018) showed that by adding SE blocks to ResNet's, as SE-ResNet-50, it had comparable
validation errors as the deeper ResNet-101 but at half the computational cost. They show that SE-blocks
are 
exible enought to be added to any model at only a nominal cost. Indeed, by adding SE-blocks to ResNets,
Hu at al. (2018) received rst place using this method in the ILSVRC 2017 classication competition.




