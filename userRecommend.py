import processData
import pandas as pd
import numpy as np
import math
from collections import OrderedDict
import operator
import json
from functools import reduce



def topk(user_N_dict):
    """
    topK click items for everyone
    :param user_1_dict:
    :return:
    """
    users_topk = {}
    for user, list in user_N_dict.items():
        list = sorted(list, reverse=True)
        items = {}
        count = 0
        current = list[0]
        for strs in list:
            if current != strs:
                items[current] = count
                count = 1
                current = strs
            else:
                count += 1
        items[current] = count
        items = sorted(items.items(), key=operator.itemgetter(1), reverse=True)
        user_topk = []
        for i in range(len(items)):
            tuple = items[i]
            user_topk.append(str(tuple[0]) + "$" + str(tuple[1]))
        users_topk[user] = user_topk
    return users_topk


def hottestItem(user_4_dict, k=30):
    items = []
    for user, list in user_4_dict.items():
        for strs in list:
            items.append(strs)
    items.sort()
    count = 0
    current = items[0]
    dict = {}
    for strs in items:
        if current != strs:
            dict[current] = count
            count = 1
            current = strs
        else:
            count += 1
    dict[current] = count
    dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    count_k = k
    item_topk = []
    for i in range(len(dict)):
        tuple = dict[i]
        if count_k > 0:
            item_topk.append(str(tuple[0]) + "$" + str(tuple[1]))
            count_k -= 1
        else:
            break
    return item_topk


def UserAction(topk1, topk2, topk3, topk4):
    """
    let all related item add into user dict
    :param topk1:
    :param topk2:
    :param topk3:
    :param topk4:
    :return:
    """
    user_item = OrderedDict()
    for user, k in topk1.items():
        k = list(map(lambda x: x + '$1', k))
        user_item[user] = k
    for user, k in topk2.items():
        k = list(map(lambda x: x + '$3', k))
        if user not in user_item.keys():
            user_item[user] = k
            continue
        l = user_item[user]
        l.extend(k)
        user_item[user] = l
    for user, k in topk3.items():
        k = list(map(lambda x: x + '$6', k))
        if user not in user_item.keys():
            user_item[user] = k
            continue
        l = user_item[user]
        l.extend(k)
        user_item[user] = l
    for user, k in topk4.items():
        k = list(map(lambda x: x + '$9', k))
        if user not in user_item.keys():
            user_item[user] = k
            continue
        l = user_item[user]
        l.extend(k)
        user_item[user] = l
    print("finish build user action table")
    return user_item


def Similarity(user_item):
    for user, k in user_item.items():
        dictk = {}
        #compute score for each item in k
        for ki in k:
            kis = ki.split('$')
            if kis[0] not in dictk.keys():
                dictk[kis[0]] = int(kis[2]) * int(kis[1])
            else:
                dictk[kis[0]] = dictk[kis[0]] + int(kis[2]) * int(kis[1])
        ld = []
        for key, value in dictk.items():
            ld.append(str(key) + "$" + str(value))
        user_item[user] = ld
    print("finish build user-item table")
    return user_item

def user_matrix(user_item, path, k=5):
    """
    :param path:
    :param user_item:
    :param k:
    :return:
    """
    user_num = len(user_item)
    users = list(user_item.keys())
    user_user = {}
    for i in range(user_num):
        user_user[users[i]] = {}
    cur_num = 0
    for i in range(user_num):
        cur_num += 1
        if cur_num % 100 == 0:
            print("building user similarity process", round((cur_num / user_num) * 100, 2), "%")
        for j in range(user_num):
            if i >= j:
                continue
            v1 = user_item[users[i]]
            v2 = user_item[users[j]]
            v1_item = list(map(lambda x: x.split("$")[0], v1))
            v2_item = list(map(lambda x: x.split("$")[0], v2))
            common_list = list(set(v1_item) & set(v2_item))
            if common_list:
                v1_sco = list(map(lambda x: int(x.split("$")[1]), v1))
                v2_sco = list(map(lambda x: int(x.split("$")[1]), v2))
                v1_dict = dict(zip(v1_item, v1_sco))
                v2_dict = dict(zip(v2_item, v2_sco))
                v1_add = reduce(lambda a, b: a + b, v1_sco)
                v2_add = reduce(lambda a, b: a + b, v2_sco)
                score = calCosine(v1_dict, v1_add, v2_dict, v2_add, common_list)
                user_user[users[i]][users[j]] = score
                user_user[users[j]][users[i]] = score
        current_user = user_user[users[i]]
        luser = []
        if current_user.keys():
            user_user[users[i]] = OrderedDict()
            sorted_tuple = sorted(current_user.items(), key=operator.itemgetter(1), reverse=True)
            stlen = len(sorted_tuple)
            if stlen < k:
                for st in range(stlen):
                    user_user[users[i]][sorted_tuple[st][0]] = sorted_tuple[st][1]
            else:
                for o in range(k):
                    user_user[users[i]][sorted_tuple[o][0]] = sorted_tuple[o][1]
            for us, sco in user_user[users[i]].items():
                luser.append(str(us) + "$" + str(sco))
        suser = {users[i]: luser}
        save_dict_line(suser, path)
        user_user[users[i]] = {}
    return user_user



def calCosine(v1_dict, v1_add, v2_dict, v2_add, common_list):
    """
    v1 and v2 are list, combined item-score data
    :param v1:
    :param v2:
    :return: cosine_similarity number
    """
    #numerator
    nu_sum = 0
    for lis in common_list:
        nu_sum += v1_dict[lis] * v2_dict[lis]
    return nu_sum / (v1_add * v2_add)


def knn(topk, k=10):
    for user, top in topk.items():
        if len(top) < k:
            topk[user] = top
        else:
            topk[user] = top[0:k]
    return topk


def notbrought(top2, top3, top4):
    users2 = set(top2.keys())
    users3 = set(top3.keys())
    users4 = set(top4.keys())
    common_users1 = users2 & users4
    common_users2 = users3 & users4
    not_buy_dict = {}
    for user in list(common_users2):
        cart = list(map(lambda x: x.split('$')[0], top3[user]))
        buy = list(map(lambda x: x.split('$')[0], top4[user]))
        not_buy = [l for l in cart if l not in buy]
        not_buy_dict[user] = not_buy
    for user in list(common_users1):
        collect = list(map(lambda x: x.split('$')[0], top2[user]))
        buy = list(map(lambda x: x.split('$')[0], top4[user]))
        not_buy = [l for l in collect if l not in buy]
        if user in not_buy_dict.keys():
            not_buy_dict[user].extend(not_buy)
        else:
            not_buy_dict[user] = not_buy
    return not_buy_dict


def save_dict(dict1, filename):
    f = open(filename, 'w')
    f.write(json.dumps(dict1))
    f.close()
    print("dict save success: " + filename)
    
    
def save_dict_line(dict1, filename):
    f = open(filename, 'a')
    f.write(json.dumps(dict1) + "\n")
    f.close()


def load_dict(filename):
    f = open(filename, 'r')
    line = f.readline()
    print("read dict success: " + filename)
    return json.loads(line)


if __name__ == "__main__":
    '''
    date = 8
    df_data, tf_data = processData.readData(date)
    #test_dict = processData.getUserTest(tf_data)
    #save_dict(test_dict, "data\\test_dict.txt")
    items = processData.geAllItems(df_data)
    #save_dict(items, "data\\all_items.txt")
    
    user_4_dict = processData.getUserTrain(df_data, 4)
    topk4 = topk(user_4_dict)
    #save_dict(topk4, "data\\topk4.txt")
    
    #rule4
    hottest = hottestItem(user_4_dict)
    save_dict(hottest, "data\\hottest.txt")
    
    
    user_2_dict = processData.getUserTrain(df_data, 2)
    topk2 = topk(user_2_dict)
    save_dict(topk2, "data\\topk2.txt")

    user_3_dict = processData.getUserTrain(df_data, 3)
    topk3 = topk(user_3_dict)
    #save_dict(topk3, "data\\topk3.txt")
    
    user_1_dict = processData.getUserTrain(df_data, 1)
    topk1 = topk(user_1_dict)
    save_dict(topk1, "data\\topk1.txt")
    
    #rule3
    top_view = knn(topk1)
    save_dict(top_view, "data\\topview.txt")
    '''
    #rule2
    # topk3 = load_dict("test/topk3.txt")
    # topk4 = load_dict("test/topk4.txt")
    # topk2 = load_dict("test/topk2.txt")
    # nb_dict = notbrought(topk2, topk3, topk4)
    # save_dict(nb_dict, "test/notbuy.txt")
    '''
    #rule1
    user_item = UserAction(topk1, topk2, topk3, topk4)
    save_dict(user_item, "data\\user_action.txt")
    user_item_combo = Similarity(user_item)
    user_user = user_matrix(user_item_combo)
    save_dict(user_user, "data\\similarity.txt")
    
    user_item = load_dict("data\\user_action.txt")
    user_item_combo = Similarity(user_item)
    user_user = user_matrix(user_item_combo)
    
    topk1 = load_dict("data\\topk1.txt")
    topk2 = load_dict("data\\topk2.txt")
    topk3 = load_dict("data\\topk3.txt")
    topk4 = load_dict("data\\topk4.txt")
    user_action = UserAction(topk1, topk2, topk3, topk4)
    #save_dict(user_action, "data\\user_action.txt")
    user_item_combo = Similarity(user_action)
    #save_dict(user_item_combo, "data\\user_item.txt")

    user_item = load_dict("data\\user_item.txt")
    user_matrix(user_item)
    '''

