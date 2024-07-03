from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def knn_iris():
    """
    用knn对鸢尾花进行分类
    :return:
    """
    # 1) 获取数据
    iris = load_iris()

    # 2) 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3) 特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4) KNN算法预估器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train, y_train)

    # 5) 模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict: \n", y_predict)
    print("真实值和预测值的比对结果：\n", y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    return None


def knn_iris_gscv():
    """
    用knn对鸢尾花进行分类，添加交叉验证集网格搜索
    :return:
    """
    # 1) 获取数据
    iris = load_iris()

    # 2) 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=6)

    # 3) 特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4) KNN算法预估器
    estimator = KNeighborsClassifier()

    # 加入网格搜索与交叉验证
    param_dict = {"n_neighbors": [1, 3, 5, 7, 9, 11, 13]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10)

    estimator.fit(x_train, y_train)

    # 5) 模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict: \n", y_predict)
    print("真实值和预测值的比对结果：\n", y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    print("最佳参数：\n", estimator.best_params_)
    print("最佳结果：\n", estimator.best_score_)
    print("最佳估计器：\n", estimator.best_estimator_)
    print("交叉验证结果：\n", estimator.cv_results_)

    return None


def nb_news():
    """
    用朴素贝叶斯算法对新闻进行分类
    :return:
    """
    # 获取数据
    news = fetch_20newsgroups(subset='all')
    # print(news.data)
    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target)
    # 特征工程：文本特征提取-tfidf
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 朴素贝叶斯算法预估器流程
    estimator = MultinomialNB(alpha=0.01)
    estimator.fit(x_train, y_train)
    # 模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict: \n", y_predict)
    print("真实值和预测值的比对结果：\n", y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    return None


if __name__ == '__main__':
    # case1
    # knn_iris()
    # case2
    # knn_iris_gscv()
    # case3
    nb_news()

