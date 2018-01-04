import json
import operator
import numpy as np
import processData
import userRecommend

def load_dict(filename):
    f = open(filename, 'r')
    line = f.readline()
    print("read dict success: " + filename)
    return json.loads(line)


def transfer_simi(similarity):
    for user, d in similarity.items():
        l = []
        if d.keys():
            for ud, sc in d.items():
                l.append(ud + "$" + str(sc))
        similarity[user] = l


def load_similarity(filename):
    f = open(filename, 'r')
    user_similarity = {}
    while 1:
        line = f.readline()
        if not line:
            break
        user_dict = json.loads(line)
        for user, k in user_dict.items():
            user_similarity[user] = k
    return user_similarity


def fill_dict(users, dict):
    c_user = dict.keys()
    for user in users:
        if user in c_user:
            continue
        dict[user] = []


def Similarity(similarity, user_item, topk4, k=5):
    '''
    return user-based-user-dict
    '''
    users = similarity.keys()
    users_reco = {}
    for user in users:
        sim = similarity[user]
        # items1 = user_item[user]
        user_brought = []
        if user in topk4.keys():
            user_brought = list(map(lambda x: x.split("$")[0], topk4[user]))
        user_recommend = []
        for user_score in sim:
            u_s = user_score.split("$")
            s_user = u_s[0]
            if float(u_s[1]) > 0.1:
                items2 = user_item[s_user]
                user_recommend.extend(similarItem(items2, user_brought, k))
        users_reco[user] = user_recommend
    return users_reco


def getUserLabel(user, pool, user_label):
    right_count = 0
    brought = user_label[user]
    brought = list(map(lambda x: str(x), brought))
    label = []
    for po in pool:
        if po in brought:
            label.append(1)
            right_count += 1
        else:
            label.append(0)
    return label, brought, right_count

def getAction(user_action):
    item_name = list(map(lambda x: x.split('$')[0], user_action))
    item_time = list(map(lambda x: int(x.split('$')[1]), user_action))
    item_type = list(map(lambda x: int(x.split('$')[2]), user_action))
    item_dict1 = {}
    item_dict2 = {}
    item_dict3 = {}
    item_dict4 = {}
    for it in range(len(item_name)):
        if item_type[it] == 1:
            item_dict1[item_name[it]] = item_time[it]
            item_dict2[item_name[it]] = 0
            item_dict3[item_name[it]] = 0
            item_dict4[item_name[it]] = 0
        elif item_type[it] == 3:
            item_dict1[item_name[it]] = 0
            item_dict2[item_name[it]] = item_time[it]
            item_dict3[item_name[it]] = 0
            item_dict4[item_name[it]] = 0
        elif item_type[it] == 6:
            item_dict1[item_name[it]] = 0
            item_dict2[item_name[it]] = 0
            item_dict3[item_name[it]] = item_time[it]
            item_dict4[item_name[it]] = 0
        elif item_type[it] == 9:
            item_dict1[item_name[it]] = 0
            item_dict2[item_name[it]] = 0
            item_dict3[item_name[it]] = 0
            item_dict4[item_name[it]] = item_time[it]
    return item_dict1, item_dict2, item_dict3, item_dict4


def classification(notbuy, hottest, topview, user_reco, item_reco, users, user_label, user_item, item_action):
    hottest_item = list(map(lambda x: x.split('$')[0], hottest))
    hottest_score = list(map(lambda x: x.split('$')[1], hottest))
    hottest_dict = dict(zip(hottest_item, hottest_score))
    user_LR_matrix = {}
    user_pool_dict = {}
    user_right_dict = {}
    len_users = len(users)
    count = 0
    for user in users:
        count += 1
        if count % 1000 == 0:
            print("user_reco_matrix process: ", round((count / len_users) * 100, 2), "%")
        user_item_a = user_item[user]
        user_item_name = list(map(lambda x: x.split("$")[0], user_item_a))
        user_item_score = list(map(lambda x: x.split("$")[1], user_item_a))
        user_item_dict = dict(zip(user_item_name, user_item_score))
        user_action = item_action[user]
        item_dict1, item_dict2, item_dict3, item_dict4 = getAction(user_action)
        user_pool = []
        nb = notbuy[user]
        user_pool.extend(nb)
        user_pool.extend(hottest_item)
        ur = user_reco[user]
        ur_dict = {}
        if ur:
            ur_item = list(map(lambda x: x.split('$')[0], ur))
            user_pool.extend(ur_item)
            ur_score = list(map(lambda x: x.split('$')[1], ur))
            ur_dict = dict(zip(ur_item, ur_score))
        tv = topview[user]
        tv_dict = {}
        if tv:
            tv_item = list(map(lambda x: x.split('$')[0], tv))
            user_pool.extend(tv_item)
            tv_score = list(map(lambda x: x.split('$')[1], tv))
            tv_dict = dict(zip(tv_item, tv_score))
        ir = item_reco[user]
        if ir:
            user_pool.extend(ir)
        first = True
        user_pool = list(set(user_pool))
        user_pool_dict[user] = user_pool
        label, item_brought, right_count = getUserLabel(user, user_pool, user_label)
        user_right_dict[user] = right_count
        for up in user_pool:
            item_feature = []
            if up in notbuy:
                item_feature.append(1)
            else:
                item_feature.append(0)
            if up in hottest_dict.keys():
                item_feature.append(float(hottest_dict[up]))
            else:
                item_feature.append(0)
            if up in tv_dict.keys():
                item_feature.append(float(tv_dict[up]))
            else:
                item_feature.append(0)
            if up in ur_dict.keys():
                item_feature.append(float(ur_dict[up]))
            else:
                item_feature.append(0)
            if up in ir:
                item_feature.append(1)
            else:
                item_feature.append(0)
            if up in user_item_dict.keys():
                item_feature.append(float(user_item_dict[up]))
            else:
                item_feature.append(0)
            if up in item_dict1.keys():
                item_feature.append(item_dict1[up])
                item_feature.append(item_dict2[up])
                item_feature.append(item_dict3[up])
                item_feature.append(item_dict4[up])
            else:
                item_feature.append(0)
                item_feature.append(0)
                item_feature.append(0)
                item_feature.append(0)
            if first:
                poolMatrix = np.array(item_feature)
                first = False
            else:
                poolMatrix = np.vstack((poolMatrix, np.array(item_feature)))
        poolMatrix = np.hstack((np.transpose(np.matrix(label)), poolMatrix))
        poolMatrix = np.hstack((np.transpose(np.matrix(list(map(float, user_pool)))), poolMatrix))
        user_LR_matrix[user] = poolMatrix
    return user_LR_matrix, user_pool_dict, user_right_dict


def LR_classification(user_matrix):
    print(1)


def similarItem(items2, user_brought, k):
    """
    item2 is the recommended
    """
    count = k
    user_brought = list(map(lambda x: x.split('$')[0], user_brought))
    item2 = list(map(lambda x: x.split('$')[0], items2))
    score2 = list(map(lambda x: x.split('$')[1], items2))
    d_item2 = {}
    for i in range(len(item2)):
        d_item2[item2[i]] = int(score2[i])
    items_tuple = sorted(d_item2.items(), key=operator.itemgetter(1), reverse=True)
    tuple_l = []
    for l in items_tuple:
        tuple_l.append(l[0])
    items_or = [l for l in tuple_l if l not in user_brought]
    reco_item = []
    for io in items_or:
        if count > 0 and d_item2[io] > 10:
            count -= 1
            reco_item.append(io + "$" + str(d_item2[io]))
    return reco_item


if __name__ == "__main__":
    users = processData.getUsers()
    similarity = load_similarity("data\\similarity2.txt")
    fill_dict(users, similarity)
    hottest = load_dict("data\\hottest.txt")
    notbuy = load_dict("data\\notbuy.txt")
    fill_dict(users, notbuy)
    topview = load_dict("data\\topview.txt")
    fill_dict(users, topview)
    user_item = load_dict("data\\user_item.txt")
    topk4 = load_dict("data\\topk4.txt")
    user_label = load_dict("data\\test_dict.txt")
    fill_dict(users, user_label)
    item_reco = load_dict("data\\item_based_user_dict.txt")
    # transfer_simi(similarity)
    user_reco = Similarity(similarity, user_item, topk4)
    user_matrix, pool_dict, right_dict = classification(notbuy, hottest, topview, user_reco, item_reco, users, user_label)
    sum = len(users)
    right = 0
    for user, ri in right_dict.items():
        if ri != 0:
            right += 1
    print("pool accuracy: ", round(right / sum * 100, 2), "%")