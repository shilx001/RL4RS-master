import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from ddpg import *
import math
import pickle
from funk_svd import SVD
import itertools
import pyflann

# feature改为index

# DDPG-KNN
np.random.seed(1)
#data = pd.read_csv('ml-latest-small/ratings.csv', header=0, names=['u_id', 'i_id', 'rating', 'timestep'])
data = pd.read_table('ml-1m/ml-1m/ratings.dat', sep='::',  names=['u_id', 'i_id', 'rating', 'timestep'])
# data = data[:100]
user_idx = data['u_id'].unique()  # id for all the user
np.random.shuffle(user_idx)
train_id = user_idx[:int(len(user_idx) * 0.8)]
test_id = user_idx[int(len(user_idx) * 0.8):]

# 找出集合中总共有多少电影
movie_id = list(data['i_id'].unique())

# 针对训练集建立user-rating matrix,构建item的feature matrix
rating_mat = np.zeros([len(train_id), len(movie_id)])
movie_id = np.array(movie_id)
for idx in train_id:  # 针对每个train数据
    record = data[data['u_id'] == idx]  # record有多个数据，所以row_index也有多个
    for _, row in record.iterrows():  # 针对每个用户的每条评分
        r = np.where(train_id == idx)
        c = np.where(row['i_id'] == movie_id)
        rating_mat[r, c] = row['rating']

# Funk SVD for item representation
train = data[data['u_id'].isin(train_id)]
test = data[data['u_id'].isin(test_id)]
svd = SVD(learning_rate=1e-3, regularization=0.005, n_epochs=200, n_factors=128, min_rating=0, max_rating=5)
svd.fit(X=data, X_val=test, early_stopping=True, shuffle=False)
item_matrix = svd.qi


def get_feature(input_id):
    # 根据输入的movie_id得出相应的feature, feature为index
    item_index = np.where(movie_id == input_id)
    return item_matrix[item_index]


def get_movie_idx(input_id):
    # 根据输入的movie_id得到存储的index
    idx = np.where(movie_id == input_id)
    return int(idx[0])


# 根据item_id找出相应的动作
dim = 3  # DDPG-KNN映射的空间维度
axis = []
points_in_each_axis = np.ceil(len(movie_id) ** (1.0 / dim)) + 1
for i in range(dim):
    axis.append(list(np.linspace(0, 1, points_in_each_axis)))

space = []  # action space for each movie.
for i in itertools.product(*axis):
    space.append(list(i))
space = np.float32(np.reshape(space, [-1, dim]))
flann = pyflann.FLANN()


def normalize(rating):
    max_rating = 5
    min_rating = 0
    return -1 + 2 * (rating - min_rating) / (max_rating - min_rating)


def one_hot_rating(input_rating):
    '''
    convert the rating to one-hot code.
    :param input_rating:
    :return: one-hot code for ratings
    '''
    output_rating = np.zeros(11)
    index = int(input_rating / 0.5)
    output_rating[index] = 1
    return output_rating


MAX_SEQ_LENGTH = 32
agent = DDPG(state_dim=128 + 11, action_dim=int(dim), action_bound=1,
             max_seq_length=MAX_SEQ_LENGTH, batch_size=64, discount_factor=1,actor_lr=1e-4,critic_lr=1e-4)

print('Start training.')
start = datetime.datetime.now()
# 根据训练数据对DDPG进行训练。
train_step = 0
user_count = 0
actor_loss_list = []
critic_loss_list = []
for id1 in train_id:
    user_record = data[data['u_id'] == id1]  # 找到该用户的所有
    state = []
    reward = []
    action = []
    for _, row in record.iterrows():  # 针对每个用户的评分数据，对state进行录入
        movie_feature = get_feature(row['i_id'])  # 用户的movie feature
        current_state = np.hstack((movie_feature.flatten(), one_hot_rating(row['rating'])))
        state.append(current_state)
        reward.append(row['rating'])
        current_movie_idx = get_movie_idx(row['i_id'])
        action.append(space[current_movie_idx])
    # 针对每个state,把reward
    for i in range(2, len(state)):
        current_state = state[:i - 1]  # 到目前为止所有的state
        current_state_length = i - 1
        next_state = state[:i]
        next_state_length = i
        current_reward = reward[i]
        current_action = action[i]
        if current_state_length > MAX_SEQ_LENGTH:
            current_state = current_state[-MAX_SEQ_LENGTH:]
            current_state_length = MAX_SEQ_LENGTH
        if next_state_length > MAX_SEQ_LENGTH:
            next_state = next_state[-MAX_SEQ_LENGTH:]
            next_state_length = MAX_SEQ_LENGTH
        done = 0
        if i % 32 == 0 or i == len(state) - 1:
            done = 1
        agent.store(current_state, current_state_length, current_action, current_reward, next_state,
                    next_state_length, done)
        a_loss, c_loss = [], []
        memory_length = agent.replay_buffer.get_size()
        if memory_length > 100:
            a, c = agent.train(1)
            a_loss.append(a)
            c_loss.append(c)
    print('Learning step ', user_count)
    print('User #:', user_count, '/', len(train_id))
    print('Actor loss: ', np.mean(a_loss))
    print('Critic loss: ', np.mean(c_loss))
    # actor_loss_list.append(a_loss)
    # critic_loss_list.append(c_loss)
    train_step += 1
    user_count+=1
    agent.replay_buffer.clear()

print('Training finished.')
end = datetime.datetime.now()
training_time = (end - start).seconds


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
    recall = recommend_relevant / relevant if relevant != 0 else 0
    precision = recommend_relevant / selected if selected != 0 else 0
    return output_reward / session_length, precision, recall, mrr / session_length


print('Begin test.')

start = datetime.datetime.now()
# TEST阶段
total_testing_steps = 0
result = []
ratio = 0.1
#K = int(np.ceil(ratio * len(movie_id)))
K = 1
N = 30  # top-N evaluation
test_count = 0
for idx1 in test_id:  # 针对test_id中的每个用户
    user_record = data[data['u_id'] == idx1]
    user_watched_list = []
    user_rating_list = []
    relevant = 0
    recommend_relevant = 0
    selected = 0
    r = 0
    all_state = []
    all_recommend = []
    all_item = []
    all_rating = []
    for idx2, row in user_record.iterrows():  # 针对每个电影记录
        user_rating_list.append(row['rating'])
        current_movie = row['i_id']
        current_state = np.hstack(
            (get_feature(current_movie).flatten(), one_hot_rating(row['rating'])))  # current state的维度: movie_length+1
        all_state.append(current_state)  # all_state: 所有的state集合的list
        if len(all_state) > 1:  # 针对第二个电影开始推荐
            temp_state = all_state[:-1]  # 当前的特征list,不包含倒数第一个
            if len(temp_state) > MAX_SEQ_LENGTH:
                temp_state = temp_state[-MAX_SEQ_LENGTH:]
            proto_action = np.float32(agent.get_action(temp_state, len(temp_state)))  # DDPG-knn输出的Proto action
            total_testing_steps += 1
            # 根据proto_action找K个最近的动作
            search_idx, _ = flann.nn(space, np.reshape(proto_action, [dim, ]), K, algorithm='kmeans')  # [1, K]
            # 评估nearest_index的value
            search_idx = list(search_idx.flatten())
            search_action = [space[i] for i in search_idx]
            eval_state = []
            eval_length = []
            eval_action = []
            # 对temp_state进行补0
            temp_length = len(temp_state)
            if len(temp_state) < MAX_SEQ_LENGTH:  # 如果当前的小于max_length，就补0
                padding_mat = np.zeros([MAX_SEQ_LENGTH - len(temp_state), 128 + 11])
                temp_state = np.vstack((np.array(temp_state), padding_mat))
            eval_state = np.tile(temp_state, [K, 1])
            eval_length = np.tile(temp_length, [K, 1])
            critic_value = agent.eval_critic(eval_state, eval_length, search_action)  # 评估所有动作里的
            # 推荐Q值最高的N个
            critic_value = critic_value.flatten()
            temp_idx = np.argsort(-critic_value)[:N]  # 找距离最近的N个
            recommend_idx = []
            for _ in temp_idx:
                if _ < len(movie_id):
                    recommend_idx.append(int(_))
            recommend_movie = [movie_id[int(_)] for _ in recommend_idx]  # 转为list
            # 针对每个推荐item评估下
            all_recommend.append(recommend_movie)
            all_item.append(row['i_id'])
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
end = datetime.datetime.now()
testing_time = (end - start).seconds

print('###############')
print('Learning finished')
print('Total training steps: {}'.format(train_step))
print('Total learning time: {}'.format(training_time))
print('Average learning time for each step: {:.5f}'.format(training_time / train_step))
print('Total testing steps: {}'.format(total_testing_steps))
print('Total testing time: {}'.format(testing_time))
print('Average time per decision: {:.5f}'.format(testing_time / total_testing_steps))

pickle.dump(result, open('ddpg_knn', mode='wb'))
print('Result:')
display = np.mean(np.array(result).reshape([-1, 8]), axis=0)
for num in display:
    print('%.5f' % num)
