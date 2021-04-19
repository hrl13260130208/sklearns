

from model import titanic_preprocess
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt




def rfc_find():
    '''
    使用for循环寻找最优参数
    :return:
    '''
    x,y,test= titanic_preprocess.get_xy()
    train_x,test_x,train_y,test_y=train_test_split(x,y)

    print(train_x.shape)

    ts = []
    cs = []

    for i in range(1, 100):
        print(i)
        rfc= RandomForestClassifier(n_estimators=i,random_state=0)
        rfc.fit(train_x, train_y)

        train_score = rfc.score(train_x, train_y)
        # s=clf.score(test_x,test_y)
        # 交叉验证
        cross_score = cross_val_score(rfc, x, y, cv=10).mean()

        ts.append(train_score)
        cs.append(cross_score)

    print(max(cs),cs.index(max(cs))+1)
    lindex = range(1, 100)
    plt.plot(lindex, ts, color="red", label="train")
    plt.plot(lindex, cs, color="blue", label="test")
    # plt.xticks(lindex,step=10)
    plt.legend()
    plt.show()


def rfc():
    x, y,test = titanic_preprocess.get_xy()
    train_x, test_x, train_y, test_y = train_test_split(x, y)
    rfc = RandomForestClassifier(n_estimators=89, random_state=0)
    rfc.fit(train_x, train_y)

    print(rfc.score(test_x,test_y))

    p=rfc.predict(test)
    for i in p:
        print(i)







if __name__ == '__main__':
    # rfc_find()
    rfc()


