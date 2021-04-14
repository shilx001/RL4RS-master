import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# data = pd.read_csv('ml-latest-small/ratings.csv')
# data = pd.read_table('ratings.dat', sep='::', names=['userId', 'movieId', 'rating', 'timestep'])
data = pd.read_table('ml-1m/ml-1m/ratings.dat', sep='::', names=['userId', 'movieId', 'rating', 'timestep'])

np.random.seed(1)
user_idx = data['userId'].unique()  # id for all the user
np.random.shuffle(user_idx)
train_id = user_idx[:int(len(user_idx) * 0.8)]
test_id = user_idx[int(len(user_idx) * 0.8):]

# 找出集合中有多少个电影。
movie_id = []
for idx1 in train_id:  # 针对train_id中的每个用户
    user_record = data[data['userId'] == idx1]
    for idx2, row in user_record.iterrows():
        # 检查list中是否有该电影的id
        if row['movieId'] in movie_id:
            idx = movie_id.index(row['movieId'])  # 找到该位置
        else:
            # 否则新加入movie_id
            movie_id.append(row['movieId'])

# 针对训练集建立user-rating matrix
rating_mat = np.zeros([len(train_id), len(movie_id)])
movie_id = np.array(movie_id)
for idx in train_id:  # 针对每个train数据
    record = data[data['userId'] == idx]  # record有多个数据，所以row_index也有多个
    for _, row in record.iterrows():  # 针对每个用户的每条评分
        r = np.where(train_id == idx)
        c = np.where(row['movieId'] == movie_id)
        rating_mat[r, c] = row['rating']

# 对rating_mat进行SVD
reduction_dim = 2
u, s, vt = np.linalg.svd(rating_mat)
S = np.zeros([len(train_id), len(movie_id)])  # 需要把输出特征值矩阵变换一下
temp_length = len(train_id) if len(train_id) < len(movie_id) else len(movie_id)
S[:temp_length, :temp_length] = np.diag(s)
S = S[:, :reduction_dim]
vt = vt[:reduction_dim, :]
svd_rating_mat = u.dot(S.dot(vt))

# 根据SVD评分最高的几个电影进行推荐
average_rating = np.mean(svd_rating_mat, axis=0)
recommend_index = np.argsort(-average_rating)

# 对测试数据进行评价
# 按照popularity进行推荐，并对测试集的reward进行评测。
result = []  # 评测结果以list进行存储，存储内容为
k = 30  # 用于评估的top-k参数
alpha = 0


# 评分是正则化到[-1,1]间
def normalize(rating):
    max_rating = 5
    min_rating = 0.5
    return -1 + 2 * (rating - min_rating) / (max_rating - min_rating)

def evaluate(recommend_id, item_id, rating, top_N):
    '''
    evalute the recommend result for each user.
    :param recommend_id: the recommend_result for each item, a list that contains the results for each item.
    :param item_id: item id.
    :param rating: user's rating on item.
    :param top_N: N, a real number of N for evaluation.
    :return: reward@N, recall@N, MRR@N
    '''
    session_length = len(recommend_id)
    relevant = 0
    recommend_relevant = 0
    selected = 0
    output_reward = 0
    mrr = 0
    for ti in range(session_length):
        current_recommend_id = list(recommend_id[ti])[:top_N]
        current_item = item_id[ti]
        current_rating = rating[ti]
        if current_rating > 3.5:
            relevant += 1
            if current_item in current_recommend_id:
                recommend_relevant += 1
        if current_item in current_recommend_id:
            selected += 1
            output_reward += normalize(current_rating)
            rank = current_recommend_id.index(current_item)
            mrr += 1.0 / (rank + 1)
    recall = recommend_relevant / relevant if relevant is not 0 else 0
    precision = recommend_relevant / selected if selected is not 0 else 0
    return output_reward / session_length, precision, recall, mrr / session_length

test_count = 0
for idx1 in test_id:  # 针对test_id中的每个用户
    user_record = data[data['userId'] == idx1]  # 找出每个用户的index
    user_watched_list = []  # 用户已观看过的电影list
    user_rating_list = []
    cp = []  # consecutive positive count
    cn = []  # consecutive negative count
    all_recommend = []
    all_item = []
    all_rating = []
    r = 0  # user rating
    for idx2, row in user_record.iterrows():  # 针对每个测试集的用户
        # 推荐电影为其没观看过的且排行最早的电影
        user_rating_list.append(row['rating'])  # 记录user_rating_list
        if idx2 is not 1:
            cp.append(np.sum(np.array(user_rating_list)[:-1] > 3))
            cn.append(np.sum(np.array(user_rating_list)[:-1] <= 3))
        else:
            cp.append(0)
            cn.append(0)
        rec_count = 0
        current_recommend = list(recommend_index[:k])
        # 针对recommend_index里面所有的电影,找出用户没有看过的k个电影推荐给用户
        for movie_idx in recommend_index:
            if movie_idx in user_watched_list:  # 如果看过，则continue
                continue
            else:
                # 如果没看过则加入推荐名单,每次推荐k个给用户
                rec_count += 1
                current_recommend.append(movie_idx)
                if rec_count > k:
                    break
        # 对推荐电影进行评估
        all_recommend.append(current_recommend)
        all_item.append(row['movieId'])
        all_rating.append(row['rating'])
    reward_10, precision_10, recall_10, mkk_10 = evaluate(all_recommend, all_item, all_rating, 10)
    reward_30, precision_30, recall_30, mkk_30 = evaluate(all_recommend, all_item, all_rating, 30)
    test_count += 1
    print('Test user #', test_count, '/', len(test_id))
    print('Reward@10: %.4f, Precision@10: %.4f, Recall@10: %.4f, MRR@10: %4f'
          % (reward_10, precision_10, recall_10, mkk_10))
    print('Reward@30: %.4f, Precision@30: %.4f, Recall@30: %.4f, MRR@30: %4f'
          % (reward_30, precision_30, recall_30, mkk_30))
    result.append([reward_10, precision_10, recall_10, mkk_10, reward_30, precision_30, recall_30, mkk_30])

pickle.dump(result, open('greedy-svd', mode='wb'))
print('Result:')
display = np.mean(np.array(result).reshape([-1, 8]), axis=0)
for num in display:
    print('%.5f' % num)
