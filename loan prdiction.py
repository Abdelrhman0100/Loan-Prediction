# import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# read datasets
data = pd.read_csv(r'C:\Users\Abdelrhman\Desktop\loan prdiction\train_ctrUa4K.csv')
print(data.head())   
print('\n\n Data Columns : \n {0} '.format(data.columns))

# encode the target    
encoder = LabelEncoder()
data.Loan_Status = encoder.fit_transform(data.Loan_Status)


# drop null values
data.dropna(how='any', inplace=True)

#train test split
train, test = train_test_split(data,test_size=0.2, random_state=0)

#seperate the target and the independant variable
x_train = train.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
y_train = train['Loan_Status']

x_test = test.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
y_test = test['Loan_Status']
print(x_train.head())


#encode the data
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

# create model object
model = LogisticRegression()
model.fit(x_train, y_train)
predict = model.predict(x_test)

# score
print('\n\nAccuracy Score on test data : \n\n')
print(accuracy_score(y_test,predict))



