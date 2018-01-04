import processData
import userRecommend
import itemRecommend

if __name__ == "__main__":
    cates, items = processData.getCategoryItems()
    cates = {}
    items = list(items.keys())
    #userRecommend.save_dict(items, "train\\all_items.txt")

    date = 30
    #df_data, tf_data = processData.readData(date)

    #test_dict = processData.getUserTest(tf_data)
    #userRecommend.save_dict(test_dict, "train\\test_dict.txt")
    tf_data = {}
    test_dict = {}

    #user_4_dict = processData.getUserTrain(df_data, 4)
    #topk4 = userRecommend.topk(user_4_dict)
    #userRecommend.save_dict(topk4, "train\\topk4.txt")

    #hottest = userRecommend.hottestItem(user_4_dict)

    #userRecommend.save_dict(hottest, "train\\hottest.txt")

    #user_2_dict = processData.getUserTrain(df_data, 2)
    #topk2 = userRecommend.topk(user_2_dict)
    #userRecommend.save_dict(topk2, "train\\topk2.txt")

    #user_3_dict = processData.getUserTrain(df_data, 3)
    #topk3 = userRecommend.topk(user_3_dict)
    #userRecommend.save_dict(topk3, "train\\topk3.txt")

    #user_1_dict = processData.getUserTrain(df_data, 1)

    df_data = {}

    #topk1 = userRecommend.topk(user_1_dict)
    #userRecommend.save_dict(topk1, "train\\topk1.txt")

    #top_view = userRecommend.knn(topk1)
    #userRecommend.save_dict(top_view, "train\\topview.txt")

    #nb_dict = userRecommend.notbrought(topk3, topk4)
    #userRecommend.save_dict(nb_dict, "train\\notbuy.txt")

    #user_action = userRecommend.UserAction(topk1, topk2, topk3, topk4)
    #userRecommend.save_dict(user_action, "train\\user_action.txt")

    topk1 = {}
    user_1_dict = {}
    topk2 = {}
    user_2_dict = {}
    topk3 = {}
    user_3_dict = {}
    topk4 = {}
    user_4_dict = {}

    #user_item_combo = userRecommend.Similarity(user_action)
    #userRecommend.save_dict(user_item_combo, "train\\user_item.txt")
    user_item_combo = userRecommend.load_dict("train\\user_item.txt")

    user_action = {}

    #userRecommend.user_matrix(user_item_combo, "train\\similarity.txt")

    users = processData.getUsers()

    item_dict, user_sorted_action = itemRecommend.item_based(items, user_item_combo)

    user_dict = itemRecommend.Similarity(item_dict, user_sorted_action, users)

    #userRecommend.save_dict(user_dict, "train\\item_based_user_dict.txt")