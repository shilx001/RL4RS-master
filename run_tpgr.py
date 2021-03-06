import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from TreePolicy import *
import datetime
import matplotlib.pyplot as plt

np.random.seed(1)
#data = pd.read_csv('ml-latest-small/ratings.csv')
# data = pd.read_table('ratings.dat',sep='::',names=['userId','movieId','rating','timestep'])
data = pd.read_table('ml-1m/ml-1m/ratings.dat', sep='::', names=['userId', 'movieId', 'rating', 'timestep'])
user_idx = data['userId'].unique()  # id for all the user
np.random.shuffle(user_idx)
train_id = user_idx[:int(len(user_idx) * 0.8)]
test_id = user_idx[int(len(user_idx) * 0.8):]

# Count the movies
movie_id = []
for idx1 in user_idx:  # 针对train_id中的每个用户
    user_record = data[data['userId'] == idx1]
    for idx2, row in user_record.iterrows():
        if row['movieId'] in movie_id:
            idx = movie_id.index(row['movieId'])  # 找到该位置
        else:
            # 否则新加入movie_id
            movie_id.append(row['movieId'])

# Build user rating matrix
rating_mat = np.zeros([len(train_id), len(movie_id)])
movie_id = np.array(movie_id)
for idx in train_id:  # 针对每个train数据
    record = data[data['userId'] == idx]  # record有多个数据，所以row_index也有多个
    for _, row in record.iterrows():  # 针对每个用户的每条评分
        r = np.where(train_id == idx)
        c = np.where(row['movieId'] == movie_id)
        rating_mat[r, c] = row['rating']


def get_feature_v2(input_id):
    # 根据输入的movie_id得出相应的feature, feature为index
    movie_index = np.where(movie_id == input_id)
    feature = np.zeros(len(movie_id))
    feature[movie_index] = 1
    return feature


def get_feature(input_id):
    # 根据输入的movie_id得出相应的feature
    movie_index = np.where(movie_id == input_id)
    return rating_mat[:, movie_index]


def action_mapping(input_id):
    '''
    convert input movie id to index
    :param input_id: movie id
    :return: index of movie.
    '''
    return np.where(movie_id == input_id)


max_seq_length = 32
state_dim = len(train_id) + 1
hidden_size = 64

feature_extractor = FeatureExtractor(state_dim=state_dim, max_seq_length=max_seq_length, hidden_size=hidden_size,
                                     learning_rate=1e-3)
# Pre-training process to learn state features.
print('Begin pre-train.')
pre_train_step = 0
loss_list = []
for id1 in train_id[:100]:
    user_record = data[data['userId'] == id1]
    state = []
    rating = []
    for _, row in user_record.iterrows():
        movie_feature = get_feature(row['movieId'])
        current_state = np.hstack((movie_feature.flatten(), row['rating']))
        rating.append(row['rating'])
        state.append(current_state)
    state_list = []
    length_list = []
    rating_list = []
    for i in range(1, len(state)):
        current_state = state[:i]
        current_state_length = i
        if current_state_length > max_seq_length:
            current_state = current_state[-max_seq_length:]
            current_state_length = max_seq_length
        current_state = np.array(current_state).reshape([current_state_length, state_dim])
        if current_state_length < max_seq_length:
            # padding the zero matrix.
            padding_mat = np.zeros([max_seq_length - current_state_length, state_dim])
            temp_state = np.vstack((current_state, padding_mat))
        length_list.append(current_state_length)
        state_list.append(temp_state)
        rating_list.append(rating[i])
    loss = feature_extractor.train(state_list, length_list, rating_list)
    loss_list.append(loss)
    pre_train_step += 1
    print('Pretrain step: ', pre_train_step, ' MSE:', loss)

plt.figure()
plt.plot(loss_list)
plt.title('Training loss')
plt.savefig('Train loss')

'''
print('Begin test for feature extraction.')
loss_list = []
for id1 in test_id:
    user_record = data[data['userId'] == id1]
    state = []
    rating = []
    for _, row in user_record.iterrows():
        movie_feature = get_feature(row['movieId'])
        current_state = np.hstack((movie_feature.flatten(), row['rating']))
        rating.append(row['rating'])
        state.append(current_state)
    state_list = []
    length_list = []
    rating_list = []
    for i in range(1, len(state)):
        current_state = state[:i]
        current_state_length = i
        if current_state_length > max_seq_length:
            current_state = current_state[-max_seq_length:]
            current_state_length = max_seq_length
        current_state = np.array(current_state).reshape([current_state_length, state_dim])
        if current_state_length < max_seq_length:
            # padding the zero matrix.
            padding_mat = np.zeros([max_seq_length - current_state_length, state_dim])
            temp_state = np.vstack((current_state, padding_mat))
        length_list.append(current_state_length)
        state_list.append(temp_state)
        rating_list.append(rating[i])
    loss = feature_extractor.get_loss(state_list, length_list, rating_list)
    loss_list.append(loss)
    pre_train_step += 1
    print('test loss: ', ' MSE:', loss)
'''

plt.figure()
plt.plot(loss_list)
plt.title('Test loss')
plt.savefig('Test loss')

agent = TreePolicy(state_dim=hidden_size, layer=3, branch=16, learning_rate=1e-4)


def normalize(rating):
    max_rating = 5
    min_rating = 0
    return -1 + 2 * (rating - min_rating) / (max_rating - min_rating)


print('Begin training the tree policy.')
discount_factor = 0.6
train_step = 0
loss_list = []
for id1 in train_id:
    user_record = data[data['userId'] == id1]
    state = []
    rating = []
    action = []
    for _, row in user_record.iterrows():
        movie_feature = get_feature(row['movieId'])
        current_state = np.hstack((movie_feature.flatten(), row['rating']))
        rating.append(row['rating'])
        state.append(current_state)
        action.append(action_mapping(row['movieId']))
    state_list = []
    action_list = []
    reward_list = []
    for i in range(2, len(state)):
        current_state = state[:i - 1]
        current_state_length = i - 1
        if current_state_length > max_seq_length:
            current_state = current_state[-max_seq_length:]
            current_state_length = max_seq_length
        current_state = np.array(current_state).reshape([current_state_length, state_dim])
        if current_state_length < max_seq_length:
            # padding the zero matrix.
            padding_mat = np.zeros([max_seq_length - current_state_length, state_dim])
            temp_state = np.vstack((current_state, padding_mat))
        state_list.append(feature_extractor.get_feature(temp_state, [current_state_length]).flatten())
        action_list.append(action[i])
        reward_list.append(normalize(rating[i]))  # normalization of the ratings to 0,1
    discount = discount_factor ** np.arange(len(reward_list))
    Q_value = np.cumsum(reward_list[::-1])
    Q_value = Q_value[::-1] * discount
    loss = agent.train(state_list, action_list, Q_value)
    loss_list.append(loss)
    train_step += 1
    print('Step ', train_step, 'Loss: ', loss)

plt.figure()
plt.plot(loss_list)
plt.title('Learning loss')
plt.savefig('Tree learning loss')

print('Begin Test')
N = 32
test_count = 0
result = []
for id1 in test_id:
    user_record = data[data['userId'] == id1]
    all_state = []
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    r = 0
    for _, row in user_record.iterrows():
        movie_feature = get_feature(row['movieId'])
        current_state = np.hstack((movie_feature.flatten(), row['rating']))
        all_state.append(current_state)
        if len(all_state) > 1:
            temp_state = all_state[:-1]
            temp_state_length = len(temp_state)
            if len(temp_state) > max_seq_length:
                temp_state = temp_state[-max_seq_length:]
                temp_state_length = max_seq_length
            # 对state进行padding
            temp_state = np.array(temp_state).reshape([temp_state_length, state_dim])
            if len(temp_state) < max_seq_length:
                padding_mat = np.zeros([max_seq_length - temp_state_length, state_dim])
                temp_state = np.vstack((temp_state, padding_mat))
            state_feature = feature_extractor.get_feature(temp_state, [temp_state_length])
            output_action = agent.get_action_prob(state_feature).flatten()
            output_action = output_action[:len(movie_id)]
            recommend_idx = np.argsort(output_action)[:N]
            recommend_movie = list(movie_id[recommend_idx])
            if row['movieId'] in recommend_movie:
                if row['rating'] > 3:
                    tp += 1
                else:
                    fp += 1
                r = normalize(row['rating'])
            else:
                if row['rating'] > 3:
                    fn += 1
                else:
                    tn += 1
    test_count += 1
    print('Test user #', test_count, '/',len(test_id))
    print('Precision: ', tp / (tp + fp + 1e-12), ' Recall: ', tp / (tp + fn + 1e-12))
    result.append([r, tp / (tp + fp + 1e-12), tp / (tp + fn + 1e-12)])

pickle.dump(result, open('tpgr', mode='wb'))
print('Result:')
print(np.mean(np.array(result).reshape([-1, 3]), axis=0))