"""
Decision Tree
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def decision_tree():
    """
    利用决策树对鸢尾花进行分类
    :return:
    """
    # 获取数据
    iris = load_iris()
    # 数据集划分, random_state=6和KNN进行对比
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
    # 特征工程
    # 预估器
    estimator = DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train, y_train)

    # 模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict: \n", y_predict)
    print("真实值和预测值的比对结果：\n", y_test == y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 决策树可视化
    export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)

    return None


if __name__ == '__main__':

    decision_tree()
