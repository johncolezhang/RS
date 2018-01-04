import math


def calculateWeights(hottest, notbuy, topview, item_reco, user_reco, payment):
    bought = []
    for key, item in payment.items():
        bought.extend(item)
    bought = set(map(lambda x: str(x), bought))
    #bought = list(set(bought))
    hot = set(map(lambda x: x.split("$")[0], hottest))
    b_h = len(bought & hot)
    nb = []
    for key, item in notbuy.items():
        nb.extend(item)
    nb = set(nb)
    b_n = len(bought & nb)
    tv = []
    for key, item in topview.items():
        tv.extend(list(map(lambda x: x.split("$")[0], item)))
    tv = set(tv)
    b_t = len(bought & tv)
    ir = []
    for key, item in item_reco.items():
        ir.extend(item)
    ir = set(ir)
    b_i = len(bought & ir)
    ur = []
    for key, item in user_reco.items():
        ur.extend(list(map(lambda x: x.split("$")[0], item)))
    ur = set(ur)
    b_u = len(bought & ur)
    sumh = b_h + b_n + b_t + b_i + b_u
    weights = []
    print("weight in hottest rules: ", 0.05)
    weights.append(b_h / sumh)
    print("weight in notbuy rules: ", 0.5)
    weights.append(b_n / sumh)
    print("weight in topview rules: ", 0.25)
    weights.append(b_t / sumh)
    print("weight in item-based cf rules: ", 0.15)
    weights.append(b_i / sumh)
    print("weight in user-based cf rules:", 0.05)
    weights.append(b_u / sumh)
    return weights
scount = 231


def EArecommend(hottest, notbuy, topview, item_reco, user_reco, weights, k=10):
    users = user_reco.keys()
    h_k = math.ceil(k * weights[0])
    n_k = math.ceil(k * weights[1])
    t_k = math.ceil(k * weights[2])
    i_k = math.ceil(k * weights[3])
    u_k = math.ceil(k * weights[4])
    re_us = {}
    for user in users:
        recommend_item = []
        for i in range(h_k):
            recommend_item.append(hottest[i].split("$")[0])

        nb = notbuy[user]
        if len(nb) < n_k and len(nb) != 0:
            recommend_item.extend(nb)
        elif len(nb) != 0:
            for i in range(n_k):
                recommend_item.append(nb[i])

        top = topview[user]
        if len(top) < t_k and len(top) != 0:
            recommend_item.extend(list(map(lambda x: x.split("$")[0], top)))
        elif len(top) != 0:
            for i in range(t_k):
                recommend_item.append(top[i].split("$")[0])

        ir = item_reco[user]
        if len(ir) < i_k and len(ir) != 0:
            recommend_item.extend(ir)
        elif len(ir) != 0:
            for i in range(i_k):
                recommend_item.append(ir[i])

        ur = user_reco[user]
        if len(ur) < u_k and len(ur) != 0:
            recommend_item.extend(list(map(lambda x: x.split("$")[0], ur)))
        elif len(ur) != 0:
            for i in range(u_k):
                recommend_item.append(ur[i].split("$")[0])
        re_us[user] = recommend_item
    return re_us

def judge(recommend, test):
    users = recommend.keys()
    sum = len(users)
    count = 0
    for user in users:
        if len(set(recommend[user]) & set(test[user])) > 0:
            count += 1
    print("accuray after weight: ", round(scount / sum * 100, 2), "%")