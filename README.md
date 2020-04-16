# Career Optimization Utilizing Census Data

## Introduction

Massive census datasets can reveal interesting aspects of the human condition and trends based on pure statistical data. Many previous studies have been conducted using these statistics. Interesting analyses to note include the definition of “rural” in the U.S. and business performance between female-owned and male-owned businesses [1, 2]. We are utilizing the UCI Machine Learning Repository’s Census Income Data Set [3] for further analysis of career development and what is a best-suited job based on statistical analysis of many attributes. We propose the development of optimized machine learning algorithms to predict and enhance users’ job selections to provide people with a variety of choices to eliminate growing career selection uncertainty. This project is driven by the economics concept of comparative advantage. Said to be the most important concept by Dr. Emily Oster of Brown, comparative advantage is the ability of an individual or group to carry out a particular economic activity (such as making a specific product) more efficiently than another activity.

## Data
Dont pick census data

### Description

Our source, as mentioned above, is the UCI Machine Learning Repository's Census Income Data Set. There are a total of 48,842 datapoints, with 15 continuous and categorical features associated with each datapoint:

1. age
2. workclass
3. fnlgwt
4. education
5. education-num
6. marital-status
7. occupation
8. relationship
9. race
10. sex
11. capital-gain
12. capital-loss
13. hours-per-week
14. native-country
15. income (whether the individual made <=50k or >50k)

The original purpose of this dataset was to predict the 15th feature, which is listed as part of our dataset for our project.

### Pre-Processing

#### Irrelevant Features

We decided to drop the fnlgwt feature, as it was irrelevant to this analysis.

#### One-Hot Encoding

The dataset is comprised of both numerical and categorical data. For a few classifiers, it is hard to use categorical data so we perform one-hot encoding. One-Hot encoding converts every categorical feature into new feature comprised of the values giving each new feature either 0 or 1. 

For example, the workclass feature has seven values: 'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', and 'Without-pay'. After One-Hot encoding, the workclass feature gets removed and gets replaced by 'workclass_Federal-gov', 'workclass_Local-gov', 'workclass_Never-worked', 'workclass_Private', 'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc', 'workclass_State-gov', and 'workclass_Without-pay' features. 

| workclass |
|-----------| 
| State-gov |

is converted to 

| workclass_Federal-gov | workclass_Local-gov | workclass_Never-worked | workclass_Private | workclass_Self-emp-inc | workclass_Self-emp-not-inc | workclass_State-gov | workclass_Without-pay |
|-----------------------|---------------------|------------------------|-------------------|------------------------|----------------------------|---------------------|-----------------------|
| 0                     | 0                   | 0                      | 0                 | 0                      | 0                          | 1                   | 0                     |

#### Standardization

We used standardized the dataset by using scikit-learn's StandardScaler basically normalizing then scaling every datapoint
by the standard deviation.


## Unsupervised Learning

### DBSCAN

First, after we pre-process our data with the exception of one-hot encoding (to avoid mixing categorical and continuous features together), we run DBSCAN on the data to identify and remove outliers. We use MinPts = 20. To find Eps, we use the elbow method where we plotted the 20th-nearest-neighbor distance for each datapoint. 

![Elbow plot to determine Eps for DBSCAN.](/plots/dbscan_elbow.png)

With this graphical method, we identified Eps = 1.571. With these two parameters, we are able to group our datapoints into two clusters and identify 44 noisy datapoints (labeled as -1 by the scikit-learn DBSCAN function) that we then delete from the dataset.

![Density Plot.](/plots/dbscan_results.png)

### GMM

Talk about GMM

## Supervised Learning

### SVM

Support Vector Machine is a form of supervised learning that classifies linearly separable data. Through methods such as one against all and using kernels, we are able to perform multi-classification on data that is not linearly separable. In order to classify the census data, a third degree polynomial kernel was chosen because the data is not linearly separable and because it is less computationally expensive then other kernels. 

In order to get the best performance out of our classifier, we must tune the parameters C and gamma. In SVM, C is the regularization parameter which determines the size of the margin between classifications. If C is large then it will create a smaller margin and increase classification accuracy. A smaller C will create a larger margin and increase computation speed but may misclassify more. Gamma is the kernel coefficient and it determines how well the classification fits the data. The high value for gamma will mean overfitting, while a low value will result in a gamma that is not representative of the training data. 

A grid search method was used in order to choose appropriate parameters for the classifier. In this method, possible C and gamma terms were specified and were systematically checked to determine the accuracy that they produced. In the end, the parameters that produced the best accuracy while being computationally efficient was C = 1 gamma = 0.054. 


### Decision Tree

### Random Forest

### Linear Regression

Regression is used to estimate the relationship between the training variables and the outcome. It can also be used for predicting the value based on the training set. We decided to use Linear Regression to figure out if we can build a linear function to help predict the different features. Due to the overfitting problem of Linear Regression, we also try Ridge and Lasso Regression to regularize the linear function so it does not overfit the training data.
For regression, there are three hyperparameters: polynomial degree, alpha, and maximum iterations. A normal linear function is a line and the line was not accomodating all the data points so the polynomial degree converts the PCA components into higher degree functions. As for the other hyperparameters, Ridge and Lasso regression trains use coordinate descent and the hyperparameters define how long to train the regression model . From the graphs below, we figured out that the dataset works perfectly for polynomial degree = 4, alpha=0.1, and max-iterations=2000.

![HyperParameter Tuning for Polynomial Degree.](/plots/Linear_HyperParameter_Polynomial_Degree.png )
![HyperParameter Tuning for Alpha.](/plots/Linear_HyperParameter_Alpha.png)
![HyperParameter Tuning for Max_iterations.](/plots/Linear_HyperParameter_Max_iterations.png)

## Results

## Conclusion

## Distribution of Work

## References

[1] Ratcliffe, Michael, et al. "Defining rural at the US Census Bureau." American community survey and geography brief  (2016): 8.

[2] Fairlie, Robert W., and Alicia M. Robb. "Gender differences in business performance: evidence from the Characteristics of Business Owners survey." Small Business Economics 33.4 (2009): 375.

[3] Kohavi, R. (1994). UCI Center for Machine Learning and Intelligent Systems [Census Income Data Set]. Retrieved from: http://mlr.cs.umass.edu/ml/datasets/Census+Income

