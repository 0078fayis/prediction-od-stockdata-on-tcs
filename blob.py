#predsiction on tcs

import pandas as pd #for data works
import matplotlib.pyplot as plt #data visualization
import quandl #dtock api for fetch data
from sklearn.linear_model import LinearRegression

# with quand api  fectch 1 month data of TCS for prediction
quandl.api_config.api_key=''  #enter key
stock_data = quandl.get('NSE/TCS', start_date='2018-12-01', end_date='2018-12-31')
print(stock_data) #see data

#convert this data to dataframe
dataset=pd.DataFrame(stock_data)
print(dataset.head())
#convert dataframe into csv
dataset.to_csv('tcs.csv')

#read csv
data=pd.read_csv('tcs.csv')
print(data.head())

#check null values

print(data.isnull().sum())

#see correlation between data
import seaborn as sns #plot correlation
plt.figure(1,figsize=(17,8))
cor=sns.heatmap(data.corr(),annot=True)

#Now we have to divide data in Dependent and Independent variable
#We can see Date column is useul for our prediction but for simplicity we have to remove it
# because date format is not proper

#Now we have to predict open price so this column is our dependent variable
#because open price depend on High,Low,Close,Last,Turnover etc...

#select our features
x=data.loc[:,'High':'Turnover (Lacs)']
y=data.loc[:,'Open']

print(x.head())
print(y.head())

# we have to split data in training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

# fit our LinearRegression Model
lr=LinearRegression()
lr.fit(x_train,y_train)
lr.score(x_train,y_train)

##I given a test data of random day

Test_data = [[2017.0 ,1979.6 ,1990.00 ,1992.70 ,2321216.0 ,46373.71]]
prediction = lr.predict(Test_data)
print(prediction)

#On that day TCS open on 1998.0 price and our model predicted price is 1999.15884851
# so we can near to the prediction







