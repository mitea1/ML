import gym
import numpy

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
initial_games = 5
numberOfGames = 3
NEGATIVE_REWARDS_THRESHOLD = 1
DRIVE_COMMAND_TURN_LEFT = [-1,0,0]
DRIVE_COMMAND_TURN_RIGHT = [1,0,0]
DRIVE_COMMAND_THROTTLE = [0,1,0]
DRIVE_COMMAND_BRAKE = [0,0,1]

#NEURAL NETWORK
x = tensorflow.placeholder(tensorflow.float64, shape=[None, 32,32,3])
y = tensorflow.placeholder(tensorflow.float64, shape=[None, 3])
numberOfNodesHiddenLayer_1 = 500
numberOfNodesHiddenLayer_2 = 500
numberOfNodesHiddenLayer_3 = 500

numberOfOutputs = 3
batchSize = 1

def generateRandomAction():
    left = random.uniform(0, -1)
    right = random.uniform(0, 1)
    throttle = random.randint(0, 1)
    brake = random.uniform(0, -1)*0.1

    action = [left + right, throttle, brake]

    return action

def transformActionToTarget(action):
    action[0] = (action[0] / 2) + 0.5
    action[2] = (action[2] / 2) + 0.5

    return action

def transformPredictionToAction(prediction):
    prediction[0] = (prediction[0] - 0.5) * 2
    prediction[2] = (prediction[2] - 0.5) * 2
    return prediction

def initial_population():
    maxScore = 0
    training_data = []
    scores = []
    accepted_scores = []

    # Iterating through games
    for _ in range(initial_games):
        actualScore = 0
        actualMaxScore = 0
        game_memory = []
        previous_observation = []
        env.reset()
        negativCounter = 0

        print('New Game')
        # Play a game till the end
        for _ in range(goal_steps):
            #env.render()
            action = generateRandomAction()
            #action = env.action_space.sample()
            print(action)
            observation, reward, done, info = env.step(action)

            previous_observation = numpy.array(observation,dtype=float)[51:83,32:64]
            action = numpy.array(action,dtype=float)[0:3]
            actualScore += reward
            if reward < 0 and actualScore > 10 :
                negativCounter += 1
            else:
                negativCounter = 0

            #If car is improving save how it drove
            if actualScore > actualMaxScore:
                actualMaxScore = actualScore
            game_memory.append([previous_observation, action])
            print('ActualScore: ',actualScore,'actualMaxScore:',actualMaxScore,' maxScore:',maxScore)

            #Stop if no further progress is made
            if done or negativCounter > NEGATIVE_REWARDS_THRESHOLD:
                break

        # If the gamescore was high enough save how you played
        if actualMaxScore > maxScore or maxScore == 0:
            accepted_scores.append(actualMaxScore)
            maxScore = actualMaxScore
            for data in game_memory:
                training_data.append([data[0], data[1]])
            print('\n************\nData added\n*************\n')

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score: ', mean(accepted_scores))
    print('Median accepted score: ', median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data


def neural_tensorflow_model(data):

    #Define Layers
    hiddenLayer_1 = {'weights': tensorflow.Variable(tensorflow.random_normal([32,32,3, numberOfNodesHiddenLayer_1])),
                    'biases': tensorflow.Variable(tensorflow.random_normal([numberOfNodesHiddenLayer_1]))}
    hiddenLayer_2 = {'weights': tensorflow.Variable(tensorflow.random_normal([numberOfNodesHiddenLayer_1, numberOfNodesHiddenLayer_2])),
                     'biases': tensorflow.Variable(tensorflow.random_normal([numberOfNodesHiddenLayer_2]))}
    hiddenLayer_3 = {'weights': tensorflow.Variable(tensorflow.random_normal([numberOfNodesHiddenLayer_2, numberOfNodesHiddenLayer_3])),
                     'biases': tensorflow.Variable(tensorflow.random_normal([numberOfNodesHiddenLayer_3]))}
    outputLayer = {'weights': tensorflow.Variable(tensorflow.random_normal([numberOfNodesHiddenLayer_3, numberOfOutputs])),
                     'biases': tensorflow.Variable(tensorflow.random_normal([numberOfOutputs]))}

    #Connect Layers
    data = tensorflow.cast(data, tensorflow.float32)
    tensorflow.expand_dims(data,-1)
    layer1 = tensorflow.add(tensorflow.matmul(data,hiddenLayer_1['weights']),hiddenLayer_1['biases'])
    layer1 = tensorflow.nn.relu(layer1)
    layer2 = tensorflow.add(tensorflow.matmul(layer1, hiddenLayer_2['weights']), hiddenLayer_2['biases'])
    layer2 = tensorflow.nn.relu(layer2)
    layer3 = tensorflow.add(tensorflow.matmul(layer2, hiddenLayer_3['weights']), hiddenLayer_3['biases'])
    layer3 = tensorflow.nn.relu(layer3)
    output = tensorflow.matmul(layer3,outputLayer['weights'] + outputLayer['biases'])

    return output


def train_neural_tensorflow_network(x,y):
    x_float = x[0]

    prediction = neural_tensorflow_model(x_float)
    cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(prediction,tensorflow.cast(y,tensorflow.float32)))
    optimizer = tensorflow.train.AdamOptimizer().minimize(cost)

    numberOfEpochs = 5
    numberOfExamples = 10

    with tensorflow.Session as sess:
        sess.run(tensorflow.initialize_all_variables())

        for epoch in numberOfEpochs:
            epochLoss = 0
            for _ in range(int(numberOfExamples/batchSize)):
                epoch_x,epoch_y = 1,1 # training data
                _,c = sess.run([optimizer,cost], feed_dict={x: epoch_x, y: epoch_y})
                epochLoss += c
            print('Epoch', epoch, 'completed out of', numberOfEpochs, 'loss:', epochLoss)

        correct = tensorflow.equal(tensorflow.argmax(prediction,1), tensorflow.argmax(y,1))
        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:x,y:y}))

def transform_to_array(data):
    X = []
    for i in range(0,len(data)-1):
        row = data[i]
        x_entry = []
        for entry in row:
            for color in entry:
                x_entry.append(float(color[0]))
                x_entry.append(float(color[1]))
                x_entry.append(float(color[2]))
        X.append(x_entry)

    return X[0]#uint8 needs to be casted to float etc. TypeError: Value passed to parameter 'a' has DataType uint8 not in list of allowed values: float16, float32, float64, int32, complex64, complex128

def transform_to_float_array(data):
    for i in range(0,32):
        for j in range(0,32):
            for k in range(0,3):
                data[i][j][k] = float(data[i][j][k])


    return data

initial_training_data = initial_population()
train_neural_tensorflow_network(initial_training_data[0],initial_training_data[1])

