
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import re




pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', None)


common_path=r"..\data\titanic"
test_data_path=r"..\data\titanic\test.csv"
train_data_path=r"..\data\titanic\train.csv"
use_columns=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","NameTitle"]

name_title_mapping={ "Mr":1,"Mrs":2,"Miss":3,"Master":4,"Don":5,"Rev":6,"Dr":7,"Mme":8,
                     "Ms":9,"Major":10,"Lady":11,"Sir":12,"Mlle":13,"Col":14,"Capt":15,
                     "the Countess":16,"Jonkheer":17}

def get_survived():
    data = pd.read_csv(train_data_path)
    return data["Survived"]



def get_columns():
    data=pd.read_csv(train_data_path)
    data["FamilySize"] = data["SibSp"] + data["Parch"]
    data["NameLength"] = data["Name"].apply(lambda x: len(x))
    data["NameTitle"] = data["Name"].apply(name_title)

    test=pd.read_csv(test_data_path)
    test["FamilySize"] = test["SibSp"] + test["Parch"]
    test["NameLength"] = test["Name"].apply(lambda x: len(x))
    test["NameTitle"] = test["Name"].apply(name_title)

    test["Fare"] = test["Fare"].fillna(test["Fare"].mean())

    return  data.loc[:,use_columns],test.loc[:,use_columns]



def name_title(name):
    s = re.search(", .*\.", name)
    if s != None:
        title = s.group()
        title = title[1:-1].strip()
        if title in name_title_mapping.keys():
            return name_title_mapping[title]

    return 0



def sex_embarked():
    data,test = get_columns()
    data.loc[:, "Embarked"] = data.loc[:, "Embarked"].fillna("S")
    se=data.loc[:,["Sex","Embarked"]]
    test_se=test.loc[:,["Sex","Embarked"]]



    onehot = OneHotEncoder(categories='auto')
    tse = onehot.fit_transform(se).toarray()
    test_tse=onehot.transform(test_se).toarray()

    data = pd.concat([data, pd.DataFrame(tse)], axis=1)
    data.columns = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","NameTitle",'x0_female','x0_male','x1_C','x1_Q','x1_S']
    data = data.drop("Sex", axis=1)
    data = data.drop("Embarked", axis=1)

    test = pd.concat([test, pd.DataFrame(test_tse)], axis=1)

    test.columns = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","NameTitle",'x0_female','x0_male','x1_C','x1_Q','x1_S']
    test = test.drop("Sex", axis=1)
    test = test.drop("Embarked", axis=1)

    return data,test



def fillage():
    data,test=sex_embarked()

    y = data.loc[:, "Age"]
    x = data.loc[:, data.columns != "Age"]

    y_test = y.loc[y.isnull()]
    y_train = y.loc[y.notnull()]
    x_test = x.loc[y_test.index, :]
    x_train = x.loc[y_train.index, :]

    rfr = RandomForestRegressor(n_estimators=100,random_state=0)
    rfr.fit(x_train, y_train)
    rfr_x_predict = rfr.predict(x_test)

    data.loc[data.loc[:, "Age"].isnull(), "Age"] = rfr_x_predict

    test_y = test.loc[:, "Age"]
    test_x = test.loc[:, test.columns != "Age"]

    test_y_test = test_y.loc[test_y.isnull()]
    test_x_test = test_x.loc[test_y_test.index, :]
    # print(test_x_test)

    test_predict=rfr.predict(test_x_test)
    test.loc[test.loc[:, "Age"].isnull(), "Age"] = test_predict

    return data,test


def get_xy():
    x,test=fillage()
    y=get_survived()

    return x,y,test



if __name__ == '__main__':
    # data=get_columns()
    # print(data.describe())

    # sex_embarked()

    # data=fillage()
    # print(data)

    x,y,test=get_xy()

    print(test)


