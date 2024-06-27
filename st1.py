from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import jieba
import pandas as pd


def datasets_iris_demo():
    """
    sklearn数据集的使用
    :return:
    """
    # 获取数据集
    iris = load_iris()
    print("鸢尾花数据集：\n", iris)
    print("数据集描述：\n", iris["DESCR"])
    print("查看特征值的名字：\n", iris.feature_names)
    print("查看特征值：\n", iris.data, iris.data.shape)

    # 数据集的划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值：\n", x_train, x_train.shape)


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


def text_count_demo():
    """
    文本特征抽取：CountVectorizer
    :return:
    """
    data = [
        "life is short, i like like python",
        "life is too long, i dislike python"
    ]
    transfer = CountVectorizer(stop_words=["is", "too"])
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new.toarray())   # 转成字典格式
    print("feature name: \n", transfer.get_feature_names_out())

    return None


def zh_text_count_demo():
    """
    中文文本抽取
    :return:
    """
    data = [
        "我爱北京天安门",
        "天安门上太阳升"
    ]
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new.toarray())  # 转成字典格式
    print("feature name: \n", transfer.get_feature_names_out())
    return None


def zh_text_count_demo2():
    """
    中文文本抽取，手动分词
    :return:
    """
    data = [
        "我 爱 北京 天安门",
        "天安门 上 太阳 升"
    ]
    transfer = CountVectorizer()
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new.toarray())  # 转成字典格式
    print("feature name: \n", transfer.get_feature_names_out())
    return None


def cut_words(text):
    """
    中文文本抽取，使用分词器
    :return:
    """
    text_new = " ".join(list(jieba.cut(text)))     # 返回一个生成器

    return text_new


def cut_words2():
    """
    中文文本抽取，使用分词器
    :return:
    """
    # 将中文文本进行分词
    data = [
        "one to one, 今天很残酷，明天更残酷，后天很美好，但是绝大部分人是死在明天晚上，所以每个人都不要放弃今天",
        "我们看到的从很远星系来的光是几百万年之前发出的，这样当我们看到宇宙时，我们是在看他的过去",
        "如果只用一种方式了解魔种事物，你就不会真正了解它，了解事物真正含义的奥秘就在于如何将其与我们所了解到的事物相联系"
    ]
    data_new = []
    for sentence in data:
        data_new.append(cut_words(sentence))
    # print(data_new)

    # 特征值提取
    transfer = CountVectorizer()
    data_new2 = transfer.fit_transform(data_new)
    print("data_new: \n", data_new2.toarray())  # 转成字典格式
    print("feature name: \n", transfer.get_feature_names_out())

    return None


def tfidf_demo():
    """
    使用TF-IDF进行文本特征抽取
    :return:
    """
    # 将中文文本进行分词
    data = [
        "one to one, 今天很残酷，明天更残酷，后天很美好，但是绝大部分人是死在明天晚上，所以每个人都不要放弃今天",
        "我们看到的从很远星系来的光是几百万年之前发出的，这样当我们看到宇宙时，我们是在看他的过去",
        "如果只用一种方式了解魔种事物，你就不会真正了解它，了解事物真正含义的奥秘就在于如何将其与我们所了解到的事物相联系"
    ]
    data_new = []
    for sentence in data:
        data_new.append(cut_words(sentence))
    # print(data_new)

    # 特征值提取
    transfer = TfidfVectorizer(stop_words=["一种", "所以"])
    data_new2 = transfer.fit_transform(data_new)
    print("data_new: \n", data_new2.toarray())  # 转成字典格式
    print("feature name: \n", transfer.get_feature_names_out())

    return None


def minmax_demo():
    """
    归一化
    :return:
    """
    # 获取数据
    data = pd.read_csv("data1/dating.txt")
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
    data = pd.read_csv("data1/dating.txt")
    data = data.iloc[:, :3]     # 所有样本，前3个特征
    print("前3个特征的所有样本：\n", data)
    # 实例化一个转换器类
    transfer = StandardScaler()
    # 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("标准化处理后：\n", data_new, data_new.shape)

    return None


if __name__ == '__main__':
    # code1 the using of sklearn dataset
    # datasets_iris_demo()

    # code2 dict feature_extraction
    # dict_demo()

    # code3 text feature_extraction
    # text_count_demo()

    # code4 zh text feature_extraction
    # zh_text_count_demo()

    # code5 zh text feature_extraction, cut words by person
    # zh_text_count_demo2()

    # code6 use lib to cut words
    # print(cut_words("我爱北京天安门"))

    # code7
    # cut_words2()

    # code8
    # tfidf_demo()

    # code9 归一化
    # minmax_demo()

    # code10 标准化
    stand_demo()

