
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.feature_selection import SelectKBest,f_classif


import pandas
import numpy as np
import matplotlib.pyplot as plt
import re
import os

pandas.set_option('display.max_columns', 1000)
pandas.set_option('display.width', 1000)
pandas.set_option('display.max_colwidth', 1000)
pandas.set_option('display.max_rows', None)

common_path=r"D:\data\titanic"
test_data_path=r"D:\data\titanic\test.csv"
train_data_path=r"D:\data\titanic\train.csv"




use_columns=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","FamilySize","NameLength","NameTitle"]

name_title_mapping={ "Mr":1,"Mrs":2,"Miss":3,"Master":4,"Don":5,"Rev":6,"Dr":7,"Mme":8,
                     "Ms":9,"Major":10,"Lady":11,"Sir":12,"Mlle":13,"Col":14,"Capt":15,
                     "the Countess":16,"Jonkheer":17}


MEDIAN_AGE=28
MEDIAN_FARE=14.4542

def load_data(data_path):
    train_data=pandas.read_csv(data_path)
    train_data["Age"] = train_data["Age"].fillna(MEDIAN_AGE)
    train_data["Fare"] = train_data["Fare"].fillna(MEDIAN_FARE)

    train_data.loc[train_data["Sex"] == 'male', "Sex"] = 0
    train_data.loc[train_data["Sex"] == 'female', 'Sex'] = 1

    train_data["Embarked"] = train_data["Embarked"].fillna("S")
    train_data.loc[train_data["Embarked"] == 'S', "Embarked"] = 0
    train_data.loc[train_data["Embarked"] == 'C', "Embarked"] = 1
    train_data.loc[train_data["Embarked"] == 'Q', "Embarked"] = 2
    train_data["FamilySize"]=train_data["SibSp"]+train_data["Parch"]
    train_data["NameLength"]=train_data["Name"].apply(lambda x:len(x))
    train_data["NameTitle"]=train_data["Name"].apply(name_title)



    return train_data

def name_title(name):
    s=re.search(", .*\.",name)
    if s!=None:
        title=s.group()
        title=title[1:-1].strip()
        # print(title)
        if title in name_title_mapping.keys():
            return name_title_mapping[title]

    return 0

def save_data():
    train_data = load_data(train_data_path)
    x, y = train_data[use_columns], train_data["Survived"]

    train_x, test_x, train_y, test_y = train_test_split(x, y)
    print(train_y)
    train_x.to_csv(os.path.join(common_path,"train_x.csv"),index=None,header=True)
    train_y.to_csv(os.path.join(common_path,"train_y.csv"),index=None,header=True)
    test_x.to_csv(os.path.join(common_path,"test_x.csv"),index=None,header=True)
    test_y.to_csv(os.path.join(common_path,"test_y.csv"),index=None,header=True)

def load_train_data():

    train_x=pandas.read_csv(os.path.join(common_path, "train_x.csv"))
    train_y=pandas.read_csv(os.path.join(common_path, "train_y.csv"))
    test_x=pandas.read_csv(os.path.join(common_path, "test_x.csv"))
    test_y=pandas.read_csv(os.path.join(common_path, "test_y.csv"))

    train_x.to_csv()
    return train_x,train_y,test_x,test_y


def clf_model():
    train_x, train_y, test_x, test_y = load_train_data()
    x = train_x.append(test_x)
    y = train_y.append(test_y)

    clf = DecisionTreeClassifier(random_state=25,max_depth=3)
    clf.fit(train_x, train_y)
    # s=clf.score(test_x,test_y)
    # 交叉验证
    s = cross_val_score(clf, x, y, cv=10)
    print(s.mean())


def clf_predict():
    train_x, train_y, test_x, test_y = load_train_data()


    clf = DecisionTreeClassifier(random_state=25,max_depth=3)
    clf.fit(train_x, train_y)

    test_data=load_data(test_data_path)
    x = test_data[use_columns]

    r=clf.predict(x)

    for id,survived in zip(test_data["PassengerId"],r):
        print(id,survived)



def clf_find_args():
    '''
    使用for循环寻找最优参数
    :return:
    '''
    train_x, train_y, test_x, test_y = load_train_data()
    x = train_x.append(test_x)
    y = train_y.append(test_y)

    ts = []
    cs = []

    for i in range(1, 11):
        print(i)
        clf = DecisionTreeClassifier(random_state=25
                                     , max_depth=i)
        clf.fit(train_x, train_y)

        train_score = clf.score(train_x, train_y)
        # s=clf.score(test_x,test_y)
        # 交叉验证
        cross_score = cross_val_score(clf, x, y, cv=10).mean()

        ts.append(train_score)
        cs.append(cross_score)

    print(max(cs))
    line = range(1, 11)
    plt.plot(line, ts, color="red", label="train")
    plt.plot(line, cs, color="blue", label="test")
    plt.xticks(line)
    plt.legend()
    plt.show()




if __name__ == '__main__':
    # save_data()

    train_x, train_y, test_x, test_y=load_train_data()
    x=train_x.append(test_x)
    y=train_y.append(test_y)
    #
    # clf_model()
    # clf_predict()

    param={
        "criterion":("gini","entropy" ),
        # "splitter":("best","random"),
        "max_depth":[*range(1,11)],
        # "min_samples_leaf":[*range(1,50,5)],
        # "min_impurity_decrease":np.linspace(0,0.5,50)

    }
    clf = DecisionTreeClassifier(random_state=25)
    gs=GridSearchCV(clf,param,cv=10)
    gs.fit(x,y)
    print("++++++++++++")
    print(gs.best_score_)
    print("++++++++++++")
    print(gs.best_params_)






