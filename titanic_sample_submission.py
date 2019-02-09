import pandas as pd
train = pd.read_csv("E:\csvdhf5xlsxurlallfiles/titanic_train.csv")
train.head()
test = pd.read_csv(r"E:\csvdhf5xlsxurlallfiles/titanic_test.csv")
test.head()
#drop features that are not going to use
train = train.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
test = test.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
print(train.head())
print(test.head())
for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
    print(df)
train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)
features = ['Pclass', 'Age', 'Sex_binary']
target = 'Survived'
train[features].head()
train[target].head().values
#Using Classifiers
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(train[features], train[target])
#predictions
predictions = clf.predict(test[features])
print(predictions)
submissions = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':predictions})
submissions.head(10)
file_name = 'E:\csvdhf5xlsxurlallfiles/titanic_sample_submission.csv'
submissions.to_csv(file_name, index=False)
print('Saved file:'+file_name)