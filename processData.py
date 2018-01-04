import pandas as pd
from os import walk
import fileinput as fi
import Rules
from matplotlib import pyplot as plt
import numpy as np

def getCategoryItems():
    cates = {}
    items = {}
    for line in fi.input('data/cate_item.csv'):
        cate = {}
        strs = line.split(',')
        cate[strs[0]] = strs[1:-1]
        #ignore first category and last \n
        for str in strs[1:-1]:
            items[str] = strs[0]
        cates[str] = cate
    print("finish read cates and items")
    return cates, items


def getUsers():
    users = []
    for line in fi.input('data/users.csv'):
        users.append(line.replace("\n", ""))
    print("finish read users")
    return users


def getfilelist(thres, type):
    fs = []
    count = 0
    for (dirpath, dirnames, filenames) in walk('data/showData'):
        for file in filenames:
            if type == "train" and count <= thres:
                fs.append(dirpath + "/" + file)
                count += 1

            elif type == "test" and count > thres:
                fs.append(dirpath + "/" + file)
                count += 1
    fs.sort()
    return fs


def readData(day):
    """
    read data from files
    :return df_data:
    """
    d_fs = getfilelist(day, "train")
    t_fs = getfilelist(day, "test")
    df_data = pd.DataFrame(columns=['userID', 'itemID','operType', 'categoryID', 'time'])
    tf_data = pd.DataFrame(columns=['userID', 'itemID','operType', 'categoryID', 'time'])

    #################  training data  #######################
    for i in range(len(d_fs)):
        df_tmp = pd.read_csv(d_fs[i], usecols=[0, 1, 2, 4, 5])
        df_tmp.columns = ['userID', 'itemID', 'operType', 'categoryID', 'time']
        print(d_fs[i], ": training")
        df_data = df_data.append(df_tmp, ignore_index=True)

    #################  test data  ##########################
    for i in range(len(t_fs)):
        tf_tmp = pd.read_csv(t_fs[i], usecols=[0, 1, 2, 4, 5])
        tf_tmp.columns = ['userID', 'itemID', 'operType', 'categoryID', 'time']
        print(t_fs[i], ": test")
        tf_data = tf_data.append(df_tmp, ignore_index=True)

    df_data[['userID', 'itemID', 'operType', 'categoryID', 'time']] = df_data[['userID', 'itemID', 'operType', 'categoryID']].astype(int)
    tf_data[['userID', 'itemID', 'operType', 'categoryID', 'time']] = tf_data[['userID', 'itemID', 'operType', 'categoryID']].astype(int)
    tf_data = tf_data.to_sparse()
    df_data = df_data.to_sparse()
    return df_data, tf_data


def getUserTrain(df_data, type):
    """
    user-based RS
    :param df_data:
    :return usertrainDict: return map, let user as key, item#-list as value
    """
    df_show = df_data[df_data.operType == type]
    usertrainDict = {}
    for row in df_show.iterrows():
        if row[1].values[0] not in usertrainDict.keys():
            usertrainDict[row[1].values[0]] = list()
        #add item# in user-map
        usertrainDict[row[1].values[0]].append(row[1].values[1])
    print("finish build user train", type)
    return usertrainDict


def getUserTest(tf_data):
    """
    user-based RS
    :param df_data:
    :return usertrainDict: return map, let user as key, item#-list as value
    """
    df_show = tf_data[tf_data.operType == 4]
    usertrainDict = {}
    for row in df_show.iterrows():
        if row[1].values[0] not in usertrainDict.keys():
            usertrainDict[row[1].values[0]] = set()
        #add item# in user-map
        usertrainDict[row[1].values[0]].add(row[1].values[1])
    for user, row in usertrainDict.items():
        usertrainDict[user] = list(row)
    print("finish build test")
    return usertrainDict


def getItemTrain(df_data, type):
    """
    item-based RS
    :param df_data:
    :return: return map, let item as key, user#-list as value
    """
    df_show = df_data[df_data.operType == type]
    itemtrainDict = {}
    for row in df_show.iterrows():
        if row[1].values[1] not in itemtrainDict.keys():
            itemtrainDict[row[1].values[1]] = set()
        itemtrainDict[row[1].values[1]].add(row[1].values[0])
    return itemtrainDict


def geAllItems(df_data):
    it = list(set(df_data['itemID']))
    return it


def splitData(df_data):
    """
    let 30days data as training set, 31 day data as test set
    :param df_data:
    :return:
    """
    def split(x):
        return '2014-11-18' <= x[:9] <= '2014-12-17'
    df_train = df_data[df_data['time'].apply(split)]
    df_label = df_data[~df_data['time'].apply(split)]
    return df_train, df_label


def intergrationData(df_data, columns):
    df_xdata = pd.DataFrame(columns=columns)
    tmpID = set(df_data[columns[0]])
    for tmpid in tmpID:
        tmp = df_data[df_data[columns[0]] == tmpid]
        row = df_data[tmpid, tmp[columns[1]].values]
        df_xdata.append(pd.Series(row, columns), ignore_index=True)
    return df_xdata


def countType(user_1_dict, user_2_dict, user_3_dict, user_4_dict):
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for user, item in user_1_dict.items():
        item = list(map(lambda x: int(x.split("$")[1]), item))
        count1 += sum(item)
    for user, item in user_2_dict.items():
        item = list(map(lambda x: int(x.split("$")[1]), item))
        count2 += sum(item)
    for user, item in user_3_dict.items():
        item = list(map(lambda x: int(x.split("$")[1]), item))
        count3 += sum(item)
    for user, item in user_4_dict.items():
        item = list(map(lambda x: int(x.split("$")[1]), item))
        count4 += sum(item)
    count_sum = count1 + count2 + count3 + count4
    print("viewing percent: ", round(count1 / count_sum * 100, 2), "%")
    print("collect percent: ", round(count2 / count_sum * 100, 2), "%")
    print("add to cart percent: ", round(count3 / count_sum * 100, 2), "%")
    print("payment percent: ", round(count4 / count_sum * 100, 2), "%")
    label = ('view', 'collect', 'add to cart', 'payment')
    size = [round(count1 / count_sum * 100, 2), round(count2 / count_sum * 100, 2), round(count3 / count_sum * 100, 2), round(count4 / count_sum * 100, 2)]
    explode = (0.2, 0.2, 0.2, 0.2)
    pie = plt.pie(size, labels=label, autopct="%1.2f%%", explode=explode, shadow=False)
    plt.title("Percentage in Each Action Type")
    ll = [r'view: ' + str(count1), r'collect: ' + str(count2), r'add to cart: ' + str(count3), r'payment: ' + str(count4)]
    plt.legend(pie[0], ll, loc="best")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    '''
        1 as click, 2 as collect, 3 as add to cart, 4 as payment
    '''
    '''
    date = 4
    df_data, tf_data = readData(date)

    user_1_dict = getUserTrain(df_data, 1)
    user_2_dict = getUserTrain(df_data, 2)
    user_3_dict = getUserTrain(df_data, 3)
    user_4_dict = getUserTrain(df_data, 4)
    '''
    topk1 = Rules.load_dict("test/topk1.txt")
    topk2 = Rules.load_dict("test/topk2.txt")
    topk3 = Rules.load_dict("test/topk3.txt")
    topk4 = Rules.load_dict("test/topk4.txt")

    countType(topk1, topk2, topk3, topk4)

    #cates, items = getCategoryItems()

    #users = getUsers()
