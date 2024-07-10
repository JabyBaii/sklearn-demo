"""
3.岭回归
"""
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def ridge():
    """
    使用岭回归对diabetes进行预测
    :return:
    """
    # 获取数据
    diabetes = load_diabetes()
    print("特征数量：\n", diabetes.data.shape)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state=22)

    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 预估器
    estimator = Ridge(alpha=0.2, max_iter=10000)    # 参数调整
    estimator.fit(x_train, y_train)

    # 模型
    print("Ridge-权重系数：\n", estimator.coef_)
    print("Ridge-偏置：\n", estimator.intercept_)

    # 模型评估
    y_predict = estimator.predict(x_test)
    print("预测diabetes：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("Ridge-均方误差：\n", error)


if __name__ == '__main__':

    ridge()

