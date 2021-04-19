

from  sklearn.preprocessing import MinMaxScaler
from  sklearn.preprocessing import StandardScaler
from  sklearn.preprocessing import LabelEncoder
from  sklearn.preprocessing import OrdinalEncoder
from  sklearn.preprocessing import OneHotEncoder
from  sklearn.preprocessing import Binarizer
from  sklearn.preprocessing import KBinsDiscretizer

from  sklearn.ensemble import RandomForestClassifier
from  sklearn.ensemble import RandomForestRegressor
from  sklearn.model_selection import  cross_val_score


import pandas as pd

'''
    数据预处理：
        归一化、正则化、标准化、缺失值处理、类别数据（onehot）、连续值分段
'''


def test1():
    '''
        归一化
            将数据缩放到0到1之间
                通常可以使用:（x-min(x)）/(max(x)-min(x))来计算
    :return:
    '''

    data=[[-1,2],[-0.5,6],[0,10],[1,18]]

    scaler=MinMaxScaler()
    scaler.fit(data)
    result=scaler.transform(data)
    print(result)



def test2():
    '''
        标准化
            将数据按均值中心化，然后再按标准差缩放。
                通常使用（x-mean（x））/std(x)

    :return:
    '''
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

    ss=StandardScaler()
    ss.fit(data)
    stddata=ss.transform(data)

    print(stddata)
    print(stddata.mean())
    print(stddata.std())



def test3():
    '''
        缺失值处理
            通常可以使用中位数、平均值、众数等方式来填充
            填充可以用pandas来处理
    :return:
    '''
    train=pd.read_csv(r"C:\File\workspace\python\sklearns\data\titanic\train.csv")
    #显示所有的行列
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(train.head())

    print(train.describe())

    new_train=train.loc[:,"Age"].fillna(train.loc[:,"Age"].median())
    print(new_train.describe())



def test4():
    '''
        缺失值填充
            除了可以使用上述方式外，还可以使用随机森林来填补
    :return:
    '''
    train = pd.read_csv(r"C:\File\workspace\python\sklearns\data\titanic\train.csv")
    # 显示所有的行列
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    #选取特征
    data=train.loc[:,["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
    # print(data.describe())
    sex=data.loc[:,"Sex"].values.reshape(-1,1)
    onehot = OneHotEncoder(categories='auto')
    oh_sex=onehot.fit_transform(sex).toarray()
    data=pd.concat([data,pd.DataFrame(oh_sex)],axis=1)
    data.columns=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked",'x0_female','x0_male']
    data=data.drop("Sex",axis=1)

    mean_data=data.copy()

    mean_data.loc[mean_data.loc[:,"Embarked"].isnull(),"Embarked"]="S"
    mean_data.loc[:,"Age"]=mean_data.loc[:,"Age"].fillna(mean_data.loc[:,"Age"].mean())

    print(mean_data)

    # print(data)

    #先填充缺失少的列
    y=data.loc[:,"Embarked"]
    x=data.loc[:,data.columns!="Embarked"]
    #先用0填充其他的缺失值
    x=x.fillna(0)

    y_test=y.loc[y.isnull()]
    y_train=y.loc[y.notnull()]
    x_test=x.loc[y_test.index,:]
    x_train=x.loc[y_train.index,:]

    onehote = OneHotEncoder(categories='auto')
    y_train = onehote.fit_transform(y_train.values.reshape(-1,1)).toarray()

    rfc=RandomForestClassifier(n_estimators=100)
    rfc.fit(x_train,y_train)
    x_predict=rfc.predict(x_test)

    #将onehot转换为分类
    xp=onehote.inverse_transform(x_predict)
    #用预测值替换缺失值
    data.loc[data.loc[:,"Embarked"].isnull(),"Embarked"]=xp

    e=data.loc[:,"Embarked"].values.reshape(-1,1)
    ohe=onehote.transform(e).toarray()

    data=pd.concat([data,pd.DataFrame(ohe)],axis=1)
    data.columns=['Pclass','Age','SibSp',  'Parch', 'Fare', 'Embarked',  'x0_female',  'x0_male','x0_C', 'x0_Q', 'x0_S']
    data = data.drop("Embarked", axis=1)
    # print(data)

    #开始填充第二个缺失值
    y = data.loc[:, "Age"]
    x = data.loc[:, data.columns != "Age"]

    y_test = y.loc[y.isnull()]
    y_train = y.loc[y.notnull()]
    x_test = x.loc[y_test.index, :]
    x_train = x.loc[y_train.index, :]

    rfr=RandomForestRegressor(n_estimators=100)
    rfr.fit(x_train,y_train)
    rfr_x_predict=rfr.predict(x_test)

    data.loc[data.loc[:, "Age"].isnull(), "Age"] = rfr_x_predict
    # print(data.describe())


    model=RandomForestClassifier(n_estimators=100,random_state=0)
    model_y=train.loc[:,"Survived"]
    model_x=data

    cvs=cross_val_score(model,model_x,model_y,scoring="accuracy",cv=5)

    print(cvs.mean())



    #均值填充效果对比
    me =  mean_data.loc[:, "Embarked"].values.reshape(-1, 1)
    ohme = onehote.transform(me).toarray()

    mean_data = pd.concat([mean_data, pd.DataFrame(ohme)], axis=1)
    mean_data.columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'x0_female', 'x0_male', 'x0_C', 'x0_Q',
                    'x0_S']
    mean_data = mean_data.drop("Embarked", axis=1)

    mean_model = RandomForestClassifier(n_estimators=100, random_state=0)
    mean_model_y = train.loc[:, "Survived"]
    mean_model_x = mean_data

    mean_cvs=cross_val_score(mean_model,mean_model_x,mean_model_y,scoring="accuracy",cv=5)
    print(mean_cvs.mean())




def test5():
    '''
        类别数据处理（onehot）
            将类别信息编码成由0和1组成的数组。
                假设由三个类别：a、b、c。onehot会使用[1,0,0] 来代表类别a；[0,1,0]代表b；[0,0,1]代表c。
    :return:
    '''
    train=pd.read_csv(r"C:\File\workspace\python\sklearns\data\titanic\train.csv")
    #显示所有的行列
    # pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # print(train.head())


    #这里的使用的编码不是使用的onehot，他是直接将类别转换成数字，每一个数字对应一个类别。
    #onehot编码方式与这种方式可以互相转换
    embarked=train.loc[:,"Embarked"]
    le=LabelEncoder()
    le.fit(embarked)
    eml=le.transform(embarked)
    print("-------------------")
    print(eml)

    # 这个方法与上面的相识，一般上面的方法用于处理标签，这里的方法用于处理特征
    ordata=train.loc[:,["Sex","Embarked"]]
    # 填充空值
    ordata.loc[:,"Embarked"]=ordata.loc[:,"Embarked"].fillna("S")
    ordinal=OrdinalEncoder()
    ordinal.fit(ordata)
    tso=ordinal.transform(ordata)
    print("-------------------")
    print(tso)


    #这里是onehot方式的处理
    ordata = train.loc[:, ["Sex", "Embarked"]]
    # 填充空值
    ordata.loc[:, "Embarked"] = ordata.loc[:, "Embarked"].fillna("S")
    onehot=OneHotEncoder(categories= 'auto')
    onehot.fit(ordata)
    print(onehot.get_feature_names())
    oso=onehot.transform(ordata).toarray()

    print("-------------------")
    # print(type(oso))
    print(oso)

    new=pd.concat([train,pd.DataFrame(oso)],axis=1)
    new.columns=['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch',
             'Ticket','Fare','Cabin','Embarked','x0_female', 'x0_male', 'x1_C', 'x1_Q', 'x1_S']
    new.drop(["Sex", "Embarked"],axis=1,inplace=True)
    print(new)



def test6():

    '''
        连续值
            一般可以使用二值化或者分段
    :return:
    '''

    train = pd.read_csv(r"C:\File\workspace\python\sklearns\data\titanic\train.csv")
    # 显示所有的行列
    # pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)


    age = train.loc[:, "Age"]
    age=age.dropna()
    age=age.reset_index(drop=True).values.reshape(-1,1)


    #二值化，将低于threshold的值设为0，高于的设为1
    ft_age=Binarizer(threshold=30).fit_transform(age)
    print(ft_age)

    #分段，n_bins设置分段数，encode设置编码后数据的格式（onehot或ordinal），
    #strategy设置分段的策略：uniform：按数值分段
    kbd=KBinsDiscretizer(n_bins=5,encode="ordinal",strategy="uniform")
    kbd_age=kbd.fit_transform(age)

    print(kbd_age)











if __name__ == '__main__':
    test4()


