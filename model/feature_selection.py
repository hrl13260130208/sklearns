

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd



"""
    特征选择
        主要有过滤法、嵌入法、包装法、降维
"""


# def get_data():
#     train = pd.read_csv(r"C:\File\workspace\python\sklearns\data\titanic\train.csv")
#     # 显示所有的行列
#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_columns', None)
#
#     # 选取特征
#     data = train.loc[:, ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
#     # print(data.describe())
#     sex = data.loc[:, "Sex"].values.reshape(-1, 1)
#     onehot = OneHotEncoder(categories='auto')
#     oh_sex = onehot.fit_transform(sex).toarray()
#     data = pd.concat([data, pd.DataFrame(oh_sex)], axis=1)
#     data.columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", 'x0_female', 'x0_male']
#     data = data.drop("Sex", axis=1)
#
#     mean_data = data.copy()
#
#     mean_data.loc[mean_data.loc[:, "Embarked"].isnull(), "Embarked"] = "S"
#     mean_data.loc[:, "Age"] = mean_data.loc[:, "Age"].fillna(mean_data.loc[:, "Age"].mean())
#
#     # print(mean_data)
#
#     onehote = OneHotEncoder(categories='auto')
#
#     me = mean_data.loc[:, "Embarked"].values.reshape(-1, 1)
#     ohme = onehote.fit_transform(me).toarray()
#
#     mean_data = pd.concat([mean_data, pd.DataFrame(ohme)], axis=1)
#     mean_data.columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'x0_female', 'x0_male', 'x0_C', 'x0_Q',
#                          'x0_S']
#     mean_data = mean_data.drop("Embarked", axis=1)
#     #
#     # ss=StandardScaler()
#     #
#     # mean_data=ss.fit_transform(mean_data)
#
#     return mean_data


def test1():
    '''
        过滤法——方差过滤
            将方差小于指定值的列删除（默认删除方差为0的列。方差用于描述数据的偏离程度，方差为0，即列中的数据都相同。）
    :return:
    '''

    data=pd.read_csv(r"C:\File\workspace\python\sklearns\data\numbers\digit recognizor.csv")
    print(data)

    vt=VarianceThreshold()
    sdata=vt.fit_transform(data)
    sdata=pd.DataFrame(sdata)
    print(sdata)
    print(vt.variances_)



def test2():
    '''
        过滤法——卡方过滤
            卡方检验用于检查期望与输入的相关性(适用于输出为分类的情况)
    :return:
    '''

    data = pd.read_csv(r"C:\File\workspace\python\sklearns\data\numbers\digit recognizor.csv")

    y=data.loc[:,"label"]
    x=data.loc[:,data.columns!="label"]

    #使用卡方过滤去除数据
    sk=SelectKBest(chi2,k=300)
    sk_data=sk.fit_transform(x,y)
    # print(sk_data.shape)
    # print(sk.n_features_in_)

    c,p=chi2(x,y)
    print(c)
    print(p)



def test3():
    '''
        过滤法——F检验过滤
            F检验
    :return:
    '''
    data = pd.read_csv(r"C:\File\workspace\python\sklearns\data\numbers\digit recognizor.csv")

    y = data.loc[:, "label"]
    x = data.loc[:, data.columns != "label"]

    # 使用F检验过滤去除数据
    sk = SelectKBest(f_classif, k=300)
    fc=sk.fit_transform(x,y)

    print(fc)
    c,p=f_classif(x,y)
    print(c)
    print(p)



def test4():
    '''
        过滤法——互信息法

    :return:
    '''

    data = pd.read_csv(r"C:\File\workspace\python\sklearns\data\numbers\digit recognizor.csv")

    y = data.loc[:, "label"]
    x = data.loc[:, data.columns != "label"]

    # 使用互信息法过滤去除数据
    sk = SelectKBest(mutual_info_classif, k=300)
    mic=sk.fit_transform(x,y)

    print(mic)
    result=mutual_info_classif(x,y)
    print(result)


def test5():
    '''
        嵌入法
            嵌入法即让模型自己决定使用什么特征
    :return:
    '''

    data = pd.read_csv(r"C:\File\workspace\python\sklearns\data\numbers\digit recognizor.csv")

    y = data.loc[:, "label"]
    x = data.loc[:, data.columns != "label"]

    rfc= RandomForestClassifier(n_estimators=100,random_state=0)

    sfm=SelectFromModel(rfc,threshold=0.005)
    x_=sfm.fit_transform(x,y)

    print(x_.shape)
    print(x_)



def test6():
    '''
        包装法
    :return:
    '''

    data = pd.read_csv(r"C:\File\workspace\python\sklearns\data\numbers\digit recognizor.csv")

    y = data.loc[:, "label"]
    x = data.loc[:, data.columns != "label"]

    rfc = RandomForestClassifier(n_estimators=100, random_state=0)

    rfe=RFE(rfc,n_features_to_select=350,step=50)
    result=rfe.fit_transform(x,y)

    print(rfe.support_)
    print(rfe.ranking_)

    print(result)



if __name__ == '__main__':
    test6()
