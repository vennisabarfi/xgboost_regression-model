# used for when you are trying to predict continuous data like height, weight, temperature etc.
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 


california = fetch_california_housing() # dataset we're using
x,y = california.data, california.target 
xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.15)

# model
xgbr = XGBRegressor(verbosity=0) # set to zero to avoid printing training process to console
# print(xgbr) # print output on the model

# fit model on training data
xgbr.fit(xtrain, ytrain)
score = xgbr.score(xtrain, ytrain) # check training score
print("Training score: ", score)

# k-fold cross-validation: model is evaluated on different subsets of data
# you can use this to evaluate different models and pick the best one.
# cross validation:identify average training score
cv_score = cross_val_score(xgbr, xtrain,ytrain, cv=10) #model, input features for training, output features for training, how many times the model is trained and evaluated
print("CV mean score: ", cv_score.mean()) # higher mean score means better performance

# predict test data
ypred = xgbr.predict(xtest)

# accuracy
# Measure how well the model'sprediction align with the actual values
mse = mean_squared_error(ytest,ypred) #lower means better model performance. 
print("MSE: ", mse)
print("RMSE: ", mse*(1/2.0)) # root mean squared: avg diff between values predicted by a model and the actual values

# visualize
x_ax = range(len(ytest))
plt.plot(x_ax, ytest, label ="original")
plt.plot(x_ax, ypred, label="predicted")
plt.title("California test and predicted data")
plt.legend()
plt.show()