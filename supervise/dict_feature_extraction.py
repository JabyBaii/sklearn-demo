"""
2. 字典特征提取
"""
from sklearn.feature_extraction import DictVectorizer


def dict_demo():
    """
    字典特征提取:DictVectorizer
    :return:
    """
    data = [{'city': 'beijing', 'temperature': 100},
            {'city': 'shanghai', 'temperature': 100},
            {'city': 'shenzhen', 'temperature': 30}]
    transfer = DictVectorizer(sparse=False)
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new)
    print("feature name: \n", transfer.feature_names_)

    return None


if __name__ == '__main__':

    dict_demo()
