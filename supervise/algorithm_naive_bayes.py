"""
Native Bayes
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


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

    nb_news()
