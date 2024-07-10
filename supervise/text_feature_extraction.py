"""
3. 文本特征提取
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba


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


if __name__ == '__main__':

    # text_count_demo()
    # zh_text_count_demo()
    # zh_text_count_demo2()

    # ret = cut_words("我爱北京天安门")
    # print(ret)

    # cut_words2()

    tfidf_demo()


