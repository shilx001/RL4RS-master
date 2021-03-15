import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from ddpg import *
import math
import pickle
from funk_svd import SVD

# feature改为index

# DDPG-KNN
np.random.seed(1)
data = pd.read_csv('ml-latest-small/ratings.csv', header=0, names=['u_id', 'i_id', 'rating', 'timestep'])
# data = pd.read_table('ml-1m/ml-1m/ratings.dat', sep='::',  names=['u_id', 'i_id', 'rating', 'timestep'])
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


# 根据item_id找出相应的动作
BASE = 5  # 输出动作的进制
output_action_dim = int(np.ceil(math.log(len(movie_id), BASE)))  # DDPG输出动作的维度
output_action_bound = 1.0 / BASE


def action_mapping(item_id):
    # 根据movie的id返回其转换的连续型变量
    output_action = []
    item_id = np.where(movie_id == item_id)
    item_id = item_id[0]
    while item_id / BASE > 0:
        output_action.append(item_id % BASE)
        item_id = item_id // BASE
    output_action = np.hstack(
        (np.array(output_action).flatten(), np.zeros([int(output_action_dim) - len(output_action)])))
    return output_action  # 针对不满的要补0


def get_movie(movie_mask):
    # 根据电影编码得到电影的index
    # movie_mask: [N, output_action_dim]
    d = BASE ** np.arange(output_action_dim)
    return np.dot(movie_mask, d)


def normalize(rating):
    max_rating = 5
    min_rating = 0
    return -1 + 2 * (rating - min_rating) / (max_rating - min_rating)


action_mask_set = []
# 针对每个movie构建action mask集合
for idx in movie_id:
    action_mask_set.append(action_mapping(idx))

MAX_SEQ_LENGTH = 32
agent = DDPG(state_dim=128 + 1, action_dim=int(output_action_dim), action_bound=output_action_bound,
             max_seq_length=MAX_SEQ_LENGTH, batch_size=128, discount_factor=1)


print('Start training.')
start_time = datetime.datetime.now()
# 根据训练数据对DDPG进行训练。
global_step = 0
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
        current_state = np.hstack((movie_feature.flatten(), row['rating']))
        state.append(current_state)
        reward.append(row['rating'])
        action.append(action_mapping(row['i_id']))
    # 针对每个state,把reward
    for i in range(2, len(state)):
        current_state = state[:i - 1]  # 到目前为止所有的state
        current_state_length = i - 1
        next_state = state[:i]
        next_state_length = i
        current_reward = (reward[i])
        current_action = action[i]
        if current_state_length > MAX_SEQ_LENGTH:
            current_state = current_state[-MAX_SEQ_LENGTH:]
            current_state_length = MAX_SEQ_LENGTH
        if next_state_length > MAX_SEQ_LENGTH:
            next_state = next_state[-MAX_SEQ_LENGTH:]
            next_state_length = MAX_SEQ_LENGTH
        done = 0
        if i % 32 is 0:
            done = 1
        agent.store(current_state, current_state_length, current_action, current_reward, next_state,
                    next_state_length, done)
    memory_length = agent.replay_buffer.get_size()
    user_count += 1
    if memory_length > 100:
        a_loss, c_loss = agent.train(int(memory_length / 32))
        print('Learning step ', global_step)
        print('User #:', user_count, '/', len(train_id))
        print('Actor loss: ', a_loss)
        print('Critic loss: ', c_loss)
        actor_loss_list.append(a_loss)
        critic_loss_list.append(c_loss)
        global_step += 1
        agent.replay_buffer.clear()

print('Training finished.')
end_time = datetime.datetime.now()
print('Training time(seconds):', (end_time - start_time).seconds)
pickle.dump(actor_loss_list, open('actor_loss_v2', mode='wb'))
pickle.dump(critic_loss_list, open('critic_loss_v2', mode='wb'))




print('Begin test.')

start_time = datetime.datetime.now()
# TEST阶段
result = []
K = 100
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
    for idx2, row in user_record.iterrows():  # 针对每个电影记录
        user_rating_list.append(row['rating'])
        current_movie = row['i_id']
        current_state = np.hstack(
            (get_feature(current_movie).flatten(), row['rating']))  # current state的维度: movie_length+1
        all_state.append(current_state)  # all_state: 所有的state集合的list
        if len(all_state) > 1:  # 针对第二个电影开始推荐
            temp_state = all_state[:-1]  # 当前的特征list,不包含倒数第一个
            if len(temp_state) > MAX_SEQ_LENGTH:
                temp_state = temp_state[-MAX_SEQ_LENGTH:]
            proto_action = agent.get_action(temp_state, len(temp_state))  # DDPG-knn输出的Proto action
            # 根据proto_action找K个最近的动作
            dist = np.sqrt(np.sum(
                (np.array(action_mask_set).reshape([-1, int(output_action_dim)]) - proto_action.flatten()) ** 2,
                axis=1))
            sorted_index = np.argsort(dist)  # 距离从小到大排列
            nearest_index = sorted_index[:K]  # 找k个距离最近的动作，这个index是动作里面的index
            # 评估nearest_index的value
            eval_state = []
            eval_length = []
            eval_action = []
            # 对temp_state进行补0
            temp_length = len(temp_state)
            if len(temp_state) < MAX_SEQ_LENGTH:  # 如果当前的小于max_length，就补0
                padding_mat = np.zeros([MAX_SEQ_LENGTH - len(temp_state), 128 + 1])
                temp_state = np.vstack((np.array(temp_state), padding_mat))
            for idx3 in nearest_index:
                eval_state.append(temp_state)
                eval_action.append(np.array(action_mask_set[idx3]))
                eval_length.append(temp_length)
            critic_value = agent.eval_critic(eval_state, eval_length, eval_action)  # 评估所有动作里的
            # 推荐Q值最高的N个
            critic_value = critic_value.flatten()
            temp_idx = np.argsort(-critic_value)[:N]  # 找距离最近的N个
            recommend_mask = [action_mask_set[_] for _ in nearest_index[temp_idx]]  # 最近index中Q值最大的几个
            recommend_index = get_movie(np.reshape(recommend_mask, [-1, output_action_dim]))
            recommend_movie = [movie_id[int(_)] for _ in recommend_index]  # 转为list
            # 针对每个推荐item评估下
            if row['rating'] > 3.5:
                relevant += 1
                if row['i_id'] in recommend_movie:
                    recommend_relevant += 1
            if row['i_id'] in recommend_movie:
                selected += 1
                r += normalize(row['rating'])
    test_count += 1
    precision = recommend_relevant / selected if selected is not 0 else 0
    recall = recommend_relevant / relevant if relevant is not 0 else 0
    print('Test user #: ', test_count, '/', len(test_id))
    print('Precision: %.5f Recall: %.5f' % (precision, recall))
    result.append([r, precision, recall])

pickle.dump(result, open('ddpg_knn', mode='wb'))
print('Result:')
print(np.mean(np.array(result).reshape([-1, 3]), axis=0))
end_time = datetime.datetime.now()
print('Testing time(seconds):', (end_time - start_time).seconds)
