import userRecommend
import operator
import processData

def item_based(all_items, user_item):
    """
    user{item$score} -> item{user$score}
    """
    items_dict = {}
    user_sorted_action = {}
    all_items = list(map(lambda x: str(x), all_items))
    for item in all_items:
        items_dict[item] = []
    for user, action in user_item.items():
        c_item = list(map(lambda x: x.split("$")[0], action))
        c_score = list(map(lambda x: int(x.split("$")[1]), action))
        c_dict = dict(zip(c_item, c_score))
        c_sorted = sorted(c_dict.items(), key=operator.itemgetter(1), reverse=True)
        user_sorted_action[user] = c_sorted
        for i in range(len(c_item)):
            if int(c_score[i]) > 10:
                items_dict[c_item[i]].append(user + "$" + str(c_score[i]))
    print("finish build item_based_dict")
    return items_dict, user_sorted_action


def Similarity(item_dict, user_action, users, k=10):
    user_dict = {}
    for user in users:
        user_dict[user] = []
    category, item_cate = processData.getCategoryItems()
    item_dict_len = len(item_dict)
    count_item = 0
    for item, record in item_dict.items():
        count_item += 1
        if count_item % 500000 == 0:
            print("item-based simi process: ", round((count_item / item_dict_len) * 100, 2), "%")
        if len(record) < 2:
            continue
        current_cate = item_cate[item]
        relevant_user = list(map(lambda x: x.split("$")[0], record))
        ru_item = {}
        for us in relevant_user:
            us_reco = [l for l in relevant_user if l is not us]
            ur_items = []
            for ur in us_reco:
                ur_items_score = user_action[ur]
                count_k = k
                for uis in ur_items_score:
                    if count_k > 0:
                        if item_cate[uis[0]] is current_cate:
                            ur_items.append(uis[0])
                        count_k -= 1
            ur_items = [l for l in ur_items if l is not item]
            user_dict[us].extend(ur_items)
    for user, di in user_dict.items():
        user_dict[user] = list(set(di))
    return user_dict




if __name__ == "__main__":
    all_items = userRecommend.load_dict("data\\all_items.txt")
    user_item = userRecommend.load_dict("data\\user_item.txt")
    users = processData.getUsers()
    item_dict, user_sorted_action = item_based(all_items, user_item)
    user_dict = Similarity(item_dict, user_sorted_action, users)
    #userRecommend.save_dict(user_dict, "data\\item_based_user_dict.txt")