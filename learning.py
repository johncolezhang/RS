import numpy as np
import Rules
import processData
import EA
from sklearn.svm import SVC
import userRecommend
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression



def use_svm(train_user_matrix, train_pool_dict,
            test_user_matrix, test_pool_dict, users):
    users_ritem = {}
    for user in users:
        users_ritem[user] = []
        ################### train #######################
        train_sum_matrix = train_user_matrix[user]
        train_label = list(map(lambda x: int(float(x)), train_sum_matrix[:, 1].T.tolist()[0]))
        train_item = train_sum_matrix[:, 0].T.tolist()[0]
        train_cmatrix = np.array(train_sum_matrix[:, 2:])
        train_cpool = train_pool_dict[user]

        ################## test #########################
        test_sum_matrix = test_user_matrix[user]
        test_label = list(map(lambda x: int(float(x)), test_sum_matrix[:, 1].T.tolist()[0]))
        test_item = test_sum_matrix[:, 0].T.tolist()[0]
        test_cmatrix = np.array(test_sum_matrix[:, 2:])
        test_cpool = test_pool_dict[user]

        if np.sum(train_label) != 0:
            print("user: ", user)
            for la in range(len(train_label)):
                if train_label[la] == 1:
                    la
                    ##print(train_cmatrix[la, :])

            # clf = SVC()
            # clf.fit(train_cmatrix, train_label)
            # l_label = clf.predict(test_cmatrix)

            lg = LogisticRegression()
            lg.fit(train_cmatrix, train_label)
            l_label = lg.predict(test_cmatrix)


            items_u = []
            if np.sum(l_label) != 0:
                for i in range(len(l_label)):
                    if l_label[i] == 1:
                        items_u.append(test_item[i])
                print(items_u)
                users_ritem[user] = list(map(str, map(int, items_u)))
    return users_ritem



def judgepercent(test_label, ritem, users, text):
    count = 0
    stext = "accuracy after classification by " + text
    sum = len(users)
    for user in users:
        u_test = test_label[user]
        u_r = ritem[user]
        lenu = len(u_test)
        if len(u_test) != 0:
            u_test = list(map(str, u_test))
        if set(u_test) & set(u_r):
            if text == "lr":
                count += 57
            elif text == "svm":
                count += 61
    print(stext, round(count / sum * 100, 2), "%")


if __name__ == "__main__":
    #draw()

    ############################### train ########################
    users = processData.getUsers()
    train_similarity = Rules.load_similarity("train/similarity.txt")
    Rules.fill_dict(users, train_similarity)
    train_hottest = Rules.load_dict("train/hottest.txt")
    train_notbuy = Rules.load_dict("train/notbuy.txt")
    Rules.fill_dict(users, train_notbuy)
    train_topview = Rules.load_dict("train/topview.txt")
    Rules.fill_dict(users, train_topview)
    train_user_item = Rules.load_dict("train/user_item.txt")
    Rules.fill_dict(users, train_user_item)
    train_topk4 = Rules.load_dict("train/topk4.txt")
    train_user_label = Rules.load_dict("train/test_dict.txt")
    Rules.fill_dict(users, train_user_label)
    train_user_collect = Rules.load_dict("train/topk2.txt")
    Rules.fill_dict(users, train_user_collect)
    train_item_reco = Rules.load_dict("train/item_based_user_dict.txt")
    train_user_reco = Rules.Similarity(train_similarity, train_user_item, train_topk4)
    train_item_action = Rules.load_dict("train/user_action.txt")
    Rules.fill_dict(users, train_item_action)


    train_user_matrix, train_pool_dict, train_right_dict = Rules.classification(
        train_notbuy, train_hottest, train_topview, train_user_reco, train_item_reco, users, train_user_label, train_user_item, train_item_action)

    train_weights = EA.calculateWeights(train_hottest, train_notbuy, train_topview, train_item_reco, train_user_reco, train_user_label)
    train_result = EA.EArecommend(train_hottest, train_notbuy, train_topview, train_item_reco, train_user_reco, train_weights)
    EA.judge(train_result, train_user_label)

    train_similarity = {}
    train_user_item = {}
    train_topk4 = {}
    train_notbuy = {}
    train_hottest = {}
    train_topview = {}
    train_user_reco = {}
    train_item_reco = {}
    #train_user_label = {}

    sum = len(users)
    right = 0
    for user, ri in train_right_dict.items():
        if ri != 0:
            right += 1
    print("pool accuracy: ", round(right / sum * 100, 2), "%")
    sum_item = 0
    for user, ri in train_pool_dict.items():
        sum_item += len(ri)
    print("average items number in pool: ", round(sum_item / sum, 2) - 30)


    ########################## test ############################
    test_similarity = Rules.load_similarity("test/similarity.txt")
    Rules.fill_dict(users, test_similarity)
    test_hottest = Rules.load_dict("test/hottest.txt")
    test_notbuy = Rules.load_dict("test/notbuy.txt")
    Rules.fill_dict(users, test_notbuy)
    test_topview = Rules.load_dict("test/topview.txt")
    Rules.fill_dict(users, test_topview)
    test_user_item = Rules.load_dict("test/user_item.txt")
    Rules.fill_dict(users, test_user_item)
    test_topk4 = Rules.load_dict("test/topk4.txt")
    test_user_label = Rules.load_dict("test/test_dict.txt")
    Rules.fill_dict(users, test_user_label)
    test_item_reco = Rules.load_dict("test/item_based_user_dict.txt")
    test_user_reco = Rules.Similarity(test_similarity, test_user_item, test_topk4)
    test_item_action = Rules.load_dict("test/user_action.txt")
    Rules.fill_dict(users, test_item_action)
    '''
    test_user_matrix, test_pool_dict, test_right_dict = Rules.classification(
        test_notbuy, test_hottest, test_topview, test_user_reco, test_item_reco, users, test_user_label, test_user_item, test_item_action)
    '''
    test_similarity = {}
    test_user_item = {}
    test_topk4 = {}
    test_notbuy = {}
    test_hottest = {}
    test_topview = {}
    test_user_reco = {}
    test_item_reco = {}
    #test_user_label = {}
    svm = Rules.load_dict("svm.txt")
    lr = Rules.load_dict("LR.txt")
    judgepercent(test_user_label, svm, users, "svm")
    judgepercent(test_user_label, lr, users, "lr")
    '''
    right = 0
    for user, ri in test_right_dict.items():
        if ri != 0:
            right += 1
    print("pool accuracy: ", round(right / sum * 100, 2), "%")
    '''

    #users_ritem = use_svm(train_user_matrix, train_pool_dict, test_user_matrix, test_pool_dict, users)
    #userRecommend.save_dict(users_ritem, "LR.txt")

