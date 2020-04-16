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

#### Standardization

We standaridize the data.

#### One-Hot Encoding

Categorical data to continuous (ish)

## Unsupervised Learning

### DBSCAN

Talk abour DBSCAN

[Here is an elbow plot we used to determine Eps for DBSCAN](/plots/dbscan_elbow.png "Elbow method to find Eps")

### GMM

Talk about GMM

## Supervised Learning

### SVM

### Decision Tree

### Random Forest

### Linear Regression

## Results

## Conclusion

## Distribution of Work

## References

[1] Ratcliffe, Michael, et al. "Defining rural at the US Census Bureau." American community survey and geography brief  (2016): 8.

[2] Fairlie, Robert W., and Alicia M. Robb. "Gender differences in business performance: evidence from the Characteristics of Business Owners survey." Small Business Economics 33.4 (2009): 375.

[3] Kohavi, R. (1994). UCI Center for Machine Learning and Intelligent Systems [Census Income Data Set]. Retrieved from: http://mlr.cs.umass.edu/ml/datasets/Census+Income

