# Career Optimization Utilizing Census Data

## Project Outline
1. [Introduction](#introduction)
2. [Data](#data)
    1. [Description](#description)
    2. [Pre-Processing](#pre-processing)
    3. [Data Exploration](#data-exploration)
3. [Unsupervised Learning](#unsupervised-learning)
    1. [DBSCAN](#dbscan)
    2. [Gaussian Mixture Model (GMM)](#gmm)
4. [Supervised Learning](#supervised-learning)
    1. [Support Vector Machine (SVM)](#svm)
    2. [Decision Trees](#decisiontrees)
    3. [Random Forests](#randomforests)
    4. [Linear Regression](#linearregression)
5. [Results](#results)
    1. [Sex](#sex)
    2. [Occupation](#occupation)
    3. [Relationship](#relationship)
    4. [Workclass](#workclass)
    5. [Education](#education)
6. [Conclusion](#conclusion)
7. [Distribution of Work](#distribution)
8. [References](#references)

## Introduction <a name="introduction"></a>

Massive census datasets can reveal interesting aspects of the human condition and trends based on pure statistical data. Many previous studies have been conducted using these statistics. Interesting analyses to note include the definition of “rural” in the U.S. and business performance between female-owned and male-owned businesses [1, 2]. We are utilizing the UCI Machine Learning Repository’s Census Income Data Set [3] for further analysis of career development and what is a best-suited job based on statistical analysis of many attributes. We propose the development of optimized machine learning algorithms to predict and enhance users’ job selections to provide people with a variety of choices to eliminate growing career selection uncertainty. While this was the main objective of our proposed project, we expanded to be able to predict multiple catagorical labels across our data set including: education, sex, relationship, workclass and occupation. This project is driven by the economics concept of comparative advantage. Said to be the most important concept by Dr. Emily Oster of Brown, comparative advantage is the ability of an individual or group to carry out a particular economic activity (such as making a specific product) more efficiently than another activity. While we were orginally primarily interested in occupation prediction and if there were secondary occupation's people may prefer over their optimal, ranked preferences were difficult to acertain. We pivoted to focus on documenting an thourough analysis of of mutiple approcess across mutliple labels.

## Data <a name="data"></a>

### Description  <a name="description"></a>

Our source, as mentioned above, is the UCI Machine Learning Repository's Census Income Data Set. There are a total of 32,561 datapoints, with 15 continuous and categorical features associated with each datapoint:

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

### Pre-Processing <a name="pre-processing"></a>

#### Irrelevant Features

The fnlwgt feature is the number of people that the census believes the datapoint represents based on the total population. We thought the feature does not help predict other features because different datapoints can represent the same number of people so we dropped the fnlwgt feature. 

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

We standardized the dataset by using scikit-learn's StandardScaler, which removes the mean and scales the data to unit variance.

### Data Exploration <a name="data_exploration"></a>

A preliminary exploration of this massive dataset was required. All variables were plotted against each other to determine where interesting points existed. Below is an example of some of the preliminary analysis.

![Prelim Data plot.](/plots/prelim_data.png)

## Unsupervised Learning <a name="unsupervised-learning"></a>

### DBSCAN <a name="dbscan"></a>

First, after we pre-process our data with the exception of one-hot encoding (to avoid mixing categorical and continuous features together), we run DBSCAN on the data to identify and remove outliers, increasing the purity of our dataset [4]. We use MinPts = 20. To find Eps, we use the elbow method where we plotted the 20th-nearest-neighbor distance for each datapoint. 

![Elbow plot to determine Eps for DBSCAN.](/plots/all_dbscan_elbow.JPG)

With this graphical method, we identified Eps = 1.571. With these two parameters, we are able to group our datapoints into two clusters and identify 44 noisy datapoints (labeled as -1 by the scikit-learn DBSCAN function) that we then delete from the dataset.

![Density Plot.](/plots/all_dbscan_results.JPG)

### GMM <a name="gmm"></a>

Upon removing the outliers from the dataset, we run a second clustering algorithm, GMM (Gaussian Mixture Models), to group datapoints into clusters that we can represent with univariate Gaussian distributions. We do this particular algorithm to group similar points together in order to gap-fill the unspecified labels in each categorical feature based on a majority vote of intracluster points. In literature, more complex methods are implemented for this very task, but for this project, a majority vote suffices [5].

To select how many components we would like to cluster our data into, we perform GMM from 1 component to 21 components, calculate the BIC (Bayesian Information Criterion) value for each of the models, and choose the one with the lowest BIC. In this case, we chose n_components = 19.

![BIC plot.](/plots/all_gmm_bic.JPG)

## Supervised Learning <a name="supervised-learning"></a>

For supervised learning, we removed a single feature from the data and used it as a label. Then we used four different algorithms (SVM, Decision Trees, Random Forest, and Linear Regression) to see how well the algorithms could classify data for a specific label. The labels that we looked at were relationship, workclass, sex, occupation, and education.

### SVM <a name="svm"></a>

Support Vector Machine is a form of supervised learning that classifies linearly separable data. Through methods such as one against all [6] and using kernels, we are able to perform multi-classification on data that is not linearly separable. In order to classify the census data, a third degree polynomial kernel was chosen because the data is not linearly separable and because it is less computationally expensive then other kernels. 

In order to get the best performance out of our classifier, we must tune the parameters C and gamma. In SVM, C is the regularization parameter which determines the size of the margin between classifications. If C is large then it will create a smaller margin and increase classification accuracy. A smaller C will create a larger margin and increase computation speed but may misclassify more. Gamma is the kernel coefficient and it determines how well the classification fits the data. The high value for gamma will mean overfitting, while a low value will result in a gamma that is not representative of the training data. 

A grid search method was used in order to choose appropriate parameters for the classifier. In this method, possible C and gamma terms were specified and were systematically checked to determine the accuracy that they produced. In the end, the parameters that produced the best accuracy while being computationally efficient was C = 1 gamma = 0.054. 

| Accuracy (%) | gamma = 0.001      | 0.01      | 0.025   | 0.054   |
| -------------- |:------:| -----:| ---:| ---:|
|C = 0.001   | 40.48    |   40.48| 40.48| 43.88| 
|C = 0.01    | 40.48    |   40.48| 43.85| 62.08| 
|C = 0.1    |  40.48   |   41.46| 62.07| 68.97|  
|C = 1    | 40.48   |   59.2| 68.96| 69.78|



### Decision Tree <a name="decisiontrees"></a>

Decision trees are a type of decision analysis that utilizes a tree-like model comprised of nodes and leaves, where the node represents the condition that will split the outcome and the leaves represent the outcomes. They are commonly used in machine learning techniques, especially for data mining, because of their robustness for missing data, quick computation time, and efficient avoidance of noise. Its branching capabilities cleanly separate complex, nonlinear data into linear boundaries. 

While the branches of the tree can become quite computationally expensive, there are a number of parameters that can be tuned to maintain the efficiency and accuracy of decision trees. Parameter tuning includes maximum depth, the minimum number of samples required to split nodes, and quality measurement of the split. The maximum depth to search in the tree as the deeper the tree, the more splits it has [7].

A decision tree classifier was utilized and trained on this dataset. We tuned using the maximum depth parameter and minimum of sample splits parameter discussed above. Maximum depth was examined from [1, 30] and minimum of sample splits was ranged from [0.0001, 1.0]. Iterating through these parameters separately, the following results were obtained.

![Decision Tree Tuning.](/plots/all_decisiontree.JPG)


Iterating through these parameters together, we found that the highest accuracy of correctly labeling was achieved at a maximum depth of 10 and a minimum of sample splits at 0.0001.




### Random Forest <a name="randomforests"></a>
Random Forests are an extension of Decision Trees. Decision trees will not make mistakes with their training data and may overfit (assuming unlimited depth). The model will not only learn the training set but also the noise of the system itself. With unlimited depth, ther is unlimited flexibility so the tree can keep growing until it has exactly one leaf node for every single observation, perfectly classifying them all. The solution to this it to limit the depth which reduces variance but increases bias. An alternative is to combine many decision trees into a single ensemble model known as a Random Forest. For each tree there is a random sample taken to create it [8]. 

Typically samples are drawn with replacement, known as bootstrapping. With this method, each tree might have high variance with respect to a specific set of the training data but overall the entire forest will have lower variance with the cost of increasing bias. Once trained the model can average the predictions of each tree which is a method known as bagging (bootstrap aggregating). There is also a method of voting as an alternative [9].

When splitting nodes, a random subset of features is selected. This is conveniently set to the square root of the number of features (scikit learn default is the same). In regression tasks it is also common to consider all features at all nodes. 
Overall, Random Forests are hard to beat in terms of performance, can perform regression and classification tasks, adn are often (depending on implementation) are quick to train. However, the downsides are that they take more time on the prediction side and are sometimes slow in real time implementation. For data including categorical variables with different number of levels (such as the data set we selected), random forests are biased in favor of those attributes with more levels. Could possibly translate the categorical datas into one-hot form. Solutions to this issue are an active area of research [10].

There are several key hyperparameters that influence performance in addition to those already discussed (max depth, max features). The key ones include the max depth, min_impurity split, bootstrap, min samples split, etc. While most of these values are held at their default values, we tuned specifically the number of trees and depth of trees to give better performance while balancing computation time. Finally, to increase runtimes scikit can edit the number of processors you're using to run the system (n_jobs = -1, using all parallelization). The variables were parametrized and a limited number are presented. Higher numbers were discarded from the search as their computation time was too high.

| Accuracy (%) |  PCA # ->     |1      |2    |4    |**8**    |10   |15   |20   |25   |30
| -------------- |:------:| -----:| ---:| ---:| ---:| ---:| ---:| ---:| ---:| ---:|
|#oT = 50     | MD = 2    |   43.5| 54.5| 61.5| 59.2| 60.4| 55.1| 55.0| 55.5| 55.4|
|#oT = 100    | MD = 2    |   43.6| 55.5| 61.7| 59.0| 58.8| 55.5| 56.0| 56.9| 56.3|
|#oT = 100    | MD = 4    |   44.0| 59.0| 63.3| 64.1| 64.2| 63.4| 63.4| 63.5| 63.0| 
|#oT = 100    | MD = 6    |   45.1| 60.8| 66.2| 66.3| 67.2| 66.1| 66.4| 67.1| 66.7|
|#oT = 100    | MD = 8    |   44.9| 62.0| 68.0| 68.8| 69.3| 68.7| 69.4| 70.3| 70.2|
|**#oT = 100**    | **MD = 10**   |   45.1| 62.6| 69.3| 70.5| 71.4| 70.7| 71.4| 72.3| 72.4|
|#oT = 1000   | MD = 10   |   45.0| 62.8| 69.3| 70.8| 71.2| 70.7| 71.6| 72.4| 72.5|

After testing out different hyperparameters, across algorithms, we found an elbow point to occur around using 8 principle compoents. From here we selected the number of trees (#oT) to be 100 and a max depth of 10. While increasing both these numbers increase the accuracy, it is at the cost of computation time on the training side. 


### Linear Regression <a name="linearregression"></a>

Regression is used to estimate the relationship between the training variables and the outcome. It can also be used for predicting the value based on the training set. We decided to use Linear Regression to figure out if we can build a linear function to help predict the different features. Due to the overfitting problem of Linear Regression, we also try Ridge[11] and Lasso Regression [12] to regularize the linear function so it does not overfit the training data. However, Lasso ends up always having lower accuracy while Ridge has same accuracy as Linear so we only need Linear for our final results.

For regression, there are two hyperparameters: alpha, and maximum iterations. Ridge and Lasso regression trains use coordinate descent and the hyperparameters define how long to train the regression model. From the graphs below, we figured out that the dataset works best for alpha=0.1 and max-iterations=2000.

Also, most of our predicted features are categorical features. Compared to the above classifiers, regression does not work if we just map our categorical feature onto numbers so we decided to do a multi-class linear regression. Multi-class linear regression means encoding the categorical feature using one-hot encoding then creating a linear regression **Li(x)** for each one-hot encoding feature. When it comes to predicting for a datapoint **x**, we just maximize over all the linear regression on the datapoint **max_{i}  Li(x)** and output the class that had the maximum value given by their linear regression.

![Regression HyperParameter Tuning.](/plots/all_regression.JPG)

## Results and Discussion <a name="results"></a>
Below are the results of PCA analysis of each of the supervised approaches previously discussed. While we were particularly interested in occupation, 


### Sex <a name="sex"></a>

![Sex Results.](/plots/all_Sex.JPG)
The trends in the PCA analysis indicate that, generally, random forests perform the best but SVM catchs up given enough principle components. Overall, all four classifiers have above 80% accuracy for 20 pca components. Moreover, the confusion matrix indicates that there is a high chance of correctly classifying male's as male's (normalized value = 0.87) but a bit lower on correctly classifying female's (0.60). Additionally, there are a greater number of females incorrectly classified as males (0.40) than males classified as females (0.13).


### Occupation <a name="occupation"></a>

![Occupation Results.](/plots/all_Occupation.JPG)

For occupation, the trends continue with random forests performing the best again. However,  the four classifiers have a lower accuracy compared to other features coming with a maximum accuracy of 35%. Moreover, the confusion matrix indicates that most of the data was classified as ‘Craft-repair’ that could be due to most of the datapoints being ‘Craft-repair’.

### Relationship <a name="relationship"></a>

![Relationship Results.](/plots/all_Relationship.JPG)

This data contained an appropriate spread of different relationship statuses for this specific label. The accuracy of classifying relationship increases as the number of PCA component increases. However, a plateau is noticeable at around a PCA value of 8. For time-efficiency, a PCA value of 8 can be considered the best for this label. Overall, each algorithm performs well in classifying relationship except for linear regression. The best algorithm at a PCA component of 8 is Random Forests, achieving an accuracy of about 73%.

### Workclass <a name="workclass"></a>

![Workclass Results.](/plots/all_Workclass.JPG)

As seen above, accuracy of classifying workclass varied significantly for different algorithms and PCA values. Linear Regression and Random forest performed well acheiving accurarcies of 75% or above. SVM started off well but then as the PCA value grew its accuracy dropped. Lastly, Decision Trees hovered consistently around 74% accuracy.

It appears that all the algorithms had some succes in correcly classifying the data, however their "success" may be largely attributed to the data. A majority of the data had the label of "private" which meant that this person worked in the private sector. Since a majority of our training data had the label "private", our trained model will classify a majority of the test data as "private" as well (as seen in the confusion matrix below). This occurance goes to show the importance of evenly spread data that has a lot of variety. Without these traits in a data, our models are highly susceptible to our data bias.


### Education <a name="education"></a>

![Education Results.](/plots/all_Education.JPG)

Once again, the trends continue with random forests performing the best again. Random Forests have plateaus after 8 components with an accuracy of 42% while the other classifiers range between 38% - 40%. This makes education the second lowest feature to classify based on the census data. Based on the confusion matrix, the low accuracy can be explained by the skewness of the data. The data has 10490 ‘HS-grads’ that Is twice the second most label that is 7283 ‘some-college’.  We can see that ‘some-college’ is predicted correctly but the other labels are predicted as ‘HS-grads’ causing the low accuracy.

## Conclusion <a name="conclusion"></a>

In this project, we use supervised and unsupervised learning methods to perform optimized machine learning classification of census data from 1994. With unsupervised learning, we were able to detect outliers in our data using DBSCAN and gap-fill unspecificied labels using GMM. Then, with supervised learning, we built tuned models of SVMs, Decision Trees, Random Forests, and Linear Regression. We performed analyses on workclass, sex, education, occupation, and relationship features. Our results concluded that 8 was the optimal choice of number of PCA components. Additionally, we found that our tuned Random Forest performed the best across all features. Using these findings, we can further drive comparative testing and provide predictions on information of persons living in the United States in 1994.

## Distribution of Work <a name="distribution"></a>

## References - Everyone <a name="references"></a>

[1] Ratcliffe, Michael, et al. "Defining rural at the US Census Bureau." American community survey and geography brief  (2016): 8.

[2] Fairlie, Robert W., and Alicia M. Robb. "Gender differences in business performance: evidence from the Characteristics of Business Owners survey." Small Business Economics 33.4 (2009): 375.

[3] Kohavi, R. (1994). UCI Center for Machine Learning and Intelligent Systems [Census Income Data Set]. Retrieved from: http://mlr.cs.umass.edu/ml/datasets/Census+Income

[4] A. Ajiboye and al., “Anomaly Detection in Dataset for Improved Model Accuracy Using DBSCAN Clustering Algorithm,” African Journal of Computing and ICTs, vol. 8, pp. 39–46, 2015.

[5] Melchior, P. and Goulding, A. D. (2018) Filling the gaps: Gaussian mixture models from noisy, truncated or incomplete samples. Astronomy and Computing, 25, 183–194.

[6] Ben Aisen. "A Comparison of Multiclass SVM Methods", Dec 15, 2006. https://courses.media.mit.edu/2006fall/mas622j/Projects/aisen-project/

[7] Mantovani, Rafael G., et al. "Hyper-parameter tuning of a decision tree induction algorithm." 2016 5th Brazilian Conference on Intelligent Systems (BRACIS). IEEE, 2016.

[8] A. Painsky and S. Rosset, "Cross-Validated Variable Selection in Tree-Based Methods Improves Predictive Performance," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 11, pp. 2142-2153, 1 Nov. 2017.

[9] Strobl, Carolin & Boulesteix, Anne-Laure & Augustin, Thomas. (2007). Unbiased split selection for classification trees based on the Gini Index. Computational Statistics & Data Analysis. 52. 483-501. 10.1016/j.csda.2006.12.030.

[10] Deng, Houtao & Runger, George & Tuv, Eugene. (2011). Bias of Importance Measures for Multi-valued Attributes and Solutions. Lecture Notes in Computer Science. 6792. 293-300. 10.1007/978-3-642-21738-8_38.

[11] Arthur E. Hoerl, Robert W. Kannard & Kent F. Baldwin (1975) Ridge regression:some simulations, Communications in Statistics, 4:2, 105-123, DOI: 10.1080/03610927508827232

[12] R. Tibshirani (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267-288.
