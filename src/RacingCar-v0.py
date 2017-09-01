import gym

'''
Created on 26 Aug 2017

@author: adrian.mitevski
'''
import gym, random
from numpy import mean, median
import numpy as np
from collections import Counter
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn
import tensorflow

LR = 1e-3
env = gym.make('CarRacing-v0')
env.reset()
goal_steps = 1000
score_requirement = 1
initial_games = 100
DRIVE_COMMAND_TURN_LEFT = [-1,0,0]
DRIVE_COMMAND_TURN_RIGHT = [1,0,0]
DRIVE_COMMAND_THROTTLE = [0,1,0]
DRIVE_COMMAND_BRAKE = [0,0,-1]


def some_random_games_first():
    for episode in range(initial_games):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = generateRandomAction()
            observation, reward, done, info = env.step(action)
            print(observation,reward,info)
            if done:
                break

def generateRandomAction():
    left = random.uniform(0.0, -1.0)
    right = random.uniform(0.0, 1.0)
    throttle = random.uniform(0.0, 1.0)
    brake = random.uniform(0.0, 1.0)
    action = [left+right,throttle,brake]

    return action


def initial_population():
    maxScore = 0
    training_data = []
    scores = []
    accepted_scores = []

    # Iterating through games
    for _ in range(initial_games):
        score = 0
        actualMaxScore = 0
        game_memory = []
        previous_observation = []

        # Play a game till the end
        for _ in range(goal_steps):
            #env.render()
            action = generateRandomAction()
            observation, reward, done, info = env.step(action)

            previous_observation = observation
            score += reward

            #If car is improving save how it drove
            if score > actualMaxScore:
                actualMaxScore = score
            game_memory.append([previous_observation, action])
            print('Score: ',score,'actualMaxScore:',actualMaxScore,' maxScore:',maxScore)

            #Stop if no further progress is made
            if done or score < -5:
                break

        # If the gamescore was high enough save how you played
        if abs(actualMaxScore - maxScore) < 3 or maxScore == 0:
            accepted_scores.append(actualMaxScore)
            maxScore = actualMaxScore
            for data in game_memory:
                training_data.append([data[0], data[1]])

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score: ', mean(accepted_scores))
    print('Median accepted score: ', median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data


def neural_network_model(input_size):
    #InputLayer
    network = input_data(shape=[None, 96, 96, 3], name='input')

    # HiddenLayers
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    # Output Layer
    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data])
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]*X[1]))

    #X[1:313][1:96][1:96]
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True,
              run_id='openaistuff')
    return model



#some_random_games_first()




training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

with tensorflow.Session() as sess:

    for each_game in range(10):
        score = 0
        game_memory = []
        prev_observations = []
        env.reset()

        for _ in range(goal_steps):
            env.render()
            if len(prev_observations) == 0:
                action = generateRandomAction()
            else:
                action = model.predict(prev_observations)[0]
            choices.append(action)

            new_observation, reward, done, info = env.step(action)
            prev_observations = np.array([i for i in new_observation]).reshape(1,96,96,3)
            game_memory.append([new_observation, action])
            score += reward
            if done:
                break
        scores.append(score)
    print('Average accepted score: ', mean(scores))
    print('Choice 1: {}, Choice 2: {}'.format(choices.count(1) / len(choices),
                                          choices.count(0) / len(choices)))

    model.save('NN_1.model')
#
# # initial_population()
# # some_random_games_first()
#
#
#
# env = gym.make('CarRacing-v0')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action