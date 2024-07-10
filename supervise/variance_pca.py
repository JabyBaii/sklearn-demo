"""
5. 特征降维:
    过滤低方差特征（特征选择）
    主成分分析：PCA特征降维
"""
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr


def variance_demo():
    """
    降维: 过滤低方差特征（特征选择）
    :return:
    """
    data = pd.read_csv("../data_ml/factor_returns.csv")
    data = data.iloc[:, 1:-2]   # 所有样本，从第二个特征开始取，到倒数第二个（开区间不取）
    print("data: \n", data)

    transfer = VarianceThreshold(threshold=10)  # 调整threshold过滤一些不太重要（低方差）的特征
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new, data_new.shape)

    # 计算两个特征之间的相关系数
    r1 = pearsonr(data["pe_ratio"], data["pb_ratio"])
    r2 = pearsonr(data["revenue"], data["total_expense"])
    print("相关系数r1：\n", r1)
    print("相关系数r2：\n", r2)  # 系数越接近1，相关性越大，越会影响训练结果

    return None


def pca_demo():
    """
    降维：主成分分析
    :return:
    """
    data = [[3, 4, 3, 5], [1, 5, 0, 8], [2, 9, 2, 0]]
    transfer = PCA(n_components=0.95)   # 保留95%的信息
    # transfer = PCA(n_components=3)   # 将特征数降为3
    data_new = transfer.fit_transform(data)
    print("PCA降维后的数据：\n", data_new)

    return None


if __name__ == '__main__':

    variance_demo()

    pca_demo()
