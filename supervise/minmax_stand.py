"""
4. 特征预处理：
    归一化
    标准化
"""
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


def minmax_demo():
    """
    归一化
    :return:
    """
    # 获取数据
    data = pd.read_csv("../data_ml/dating.txt")
    data = data.iloc[:, :3]     # 所有样本，前3个特征
    print("前3个特征的所有样本：\n", data)
    # 实例化一个转换器类
    transfer = MinMaxScaler()
    # 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("归一化处理后：\n", data_new, data_new.shape)

    return None


def stand_demo():
    """
    标准化
    :return:
    """
    # 获取数据
    data = pd.read_csv("../data_ml/dating.txt")
    data = data.iloc[:, :3]     # 所有样本，前3个特征
    print("前3个特征的所有样本：\n", data)
    # 实例化一个转换器类
    transfer = StandardScaler()
    # 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("标准化处理后：\n", data_new, data_new.shape)

    return None


if __name__ == '__main__':

    # code9 归一化
    minmax_demo()

    # code10 标准化
    stand_demo()

