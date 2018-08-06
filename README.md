# auto1_study
auto1_study
Preperad by Omer Adiguzel

Here you can find the copy of the text that i send you via email.



Data Science Challenge
Question 1 (10 Points)
List as many use cases for the dataset as possible.

-	First, we can forecast the price variable
-	We can forecast symbolling
-	The relationships between variables and the target variable
-	Creating new class for the data, using k-means clustering

Question 2 (10 Points)
Auto1 has a similar dataset (yet much larger...) 
Pick one of the use cases you listed in question 1 and describe how building a statistical model based on the dataset could best be used to improve Auto1’s business.

-	I picked the first one, forecasting the price. 
-	First, I am going to use the famous linear regression, then random forest and lastly I am going to perform extreme gradient boosting method.

-	After analyzing the results, xgb or linear regression can be used by depending on the accuracy. I will explain them step by step later.

Question 3 (20 Points)
Implement the model you described in question 2 in R or Python. The code has to retrieve the data, train and test a statistical model, and report relevant performance criteria. 

When submitting the challenge, send us the link for a Git repository containing the code for analysis and the potential pre-processing steps you needed to apply to the dataset. You can use your own account at github.com or create a new one specifically for this challenge if you feel more comfortable.

Ideally, we should be able to replicate your analysis from your submitted source-code, so please explicit the versions of the tools and packages you are using (R, Python, etc).

Done.

Question 4 (60 Points)
A. Explain each and every of your design choices (e.g., preprocessing, model selection, hyper parameters, evaluation criteria). Compare and contrast your choices with alternative methodologies. 

Data Preprocessing
-	I used some functions to analyze the data easily.
-	Since I know the metadata, I identified the column classes
-	When we checked the Percent missing data by feature graph, we see some variables with missing values. I replaced them by their means.
-	And then separated the data by the class of variables, (numeric, char)
-	Next, I checked the histograms and density plots of the data to get a clear idea what we are dealing with. The frequencies, kurtosis, skewness etc..
-	I checked the data if there is a variable with 0 variance, so I can get it out.
First thing I did was the principal component analysis, 
-	As you can see, even the 7 of the variables can explain the ~93% of the variance. 
-	after finding the principal components, I separated them into 5 clusters using k means algorithm
Second algorithm is the linear regression:
-	First, I created a data partition for train and test with the proportion of 80 – 20
-	Regular linear regression gave a nice result; Adj-R square is ~96% that means our regression curves fits our data well. Also in this manner, 
-	By looking at the F value that is big enough to tell us there is a relationship between predictor and response variables.
-	Looking at p values, for the bigger p valued variables, this means that they are not significant, so next time we can extract them from our model.
-	Graph-1: for our model to be good, here we should see a random pattern for the residual distribution, in our case so it is.
-	Graph-2: QQ-Plot, being on a straight diagonal line shows us that the residuals are nearly normally distributed, and also this is fine for us.
-	Graph-3: and this graph shows us that how our residuals are spread and whether the residuals have an equal variance or not.
-	Graph-4: In this graph, we can identify the abnormal data, here they are 110, 35 and 8. If we extract them from data, our regression coefficients will be effected significantly.
-	When it comes to prediction, our linear regression model halts, kinda overfitting issue and data quality.
Third algorithm is random forest:
-	Here I applied cross validation to find the best parameters for the random forest algorithm.
-	Grid search found mtry = 13 for the model. As you can see In the graph also, after a certain point, error stays the same, so we defined there as a cutting point.
-	We first performed random forests, the outcomes, number of trees = 500, and variance explained ~89%. Not bad. After analyzing the graph, I just realized that 60 is enough for the number of the trees, because again error does not change significantly after that number.
-	Rebuilt the model, and the results are nice when it comes to prediction.
-	Please check the graph, and you can see how beautiful it is. (nicely fitted)
-	Also here I calculated the variable importance in the random forests model, graphs show us nicely the importance of the variables.

Lastly, the cool extreme gradient boosting;
-	First create a correlation matrix, so we can extract the highly correlated variables.
-	Here I choosed 90% for the cutoff value, and thus I extracted just 1 variable; highwat-mpg
-	Again data partitioning
-	After here, I am preparing the data for the xgb algorithm, for the categorical variables, did the one hot encoding. And I finally ready for the algorithm.
-	Before beginning the model, I again used hyperparameter tuning for the better model.
-	And after finding the best parameters, I created the model and made the prediction.

The winner is random forests, at the end you can see the graphs and the errors sum.

B. Describe how you would improve the model in Question 3 if you had more time.

First, in kmm, I try the find the optimal clusters by using scree plot, elbow method.
In random forest and xgb, after finding the variable importance, I would extract the variables that has low importance.
And for the regression, I woult try some non-linear methods and adding some Gaussian noise to the data to deal with the overfitting problem.
And also if I had more data, I would try neural networks.

Ömer Adiguzel
