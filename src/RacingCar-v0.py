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
x = tensorflow.placeholder(tensorflow.int8, shape=[None, 32, 32, 3])
y = tensorflow.placeholder(tensorflow.float32, shape=[None, 3])
numberOfNodesHiddenLayer_1 = 500
numberOfNodesHiddenLayer_2 = 500
numberOfNodesHiddenLayer_3 = 500

numberOfOutputs = 3
batchSize = 1



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
            env.render()
            action = generateRandomAction()
            #action = env.action_space.sample()
            print(action)
            observation, reward, done, info = env.step(action)

            previous_observation = numpy.array(observation)[51:83,32:64]
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


def neural_network_model():
    #InputLayer
    network = input_data(shape=[None, 32,32,3], name='input',dtype='float32')

    # HiddenLayers
    activation = 'tanh'
    network = fully_connected(network, 3072, activation=activation)
    network = dropout(network, 0.8)
    network = fully_connected(network, 100, activation=activation)
    network = dropout(network, 0.8)
    network = fully_connected(network, 50, activation=activation)
    network = dropout(network, 0.8)
    network = fully_connected(network, 25, activation=activation)
    network = dropout(network, 0.8)
    network = fully_connected(network, 12, activation=activation)
    network = dropout(network, 0.8)



    #network = dropout(network, 0.8)


    # Output Layer
    network = fully_connected(network, 3, activation='softmax')
    network = tflearn.regression(network, optimizer='adam',learning_rate=0.1,loss='categorical_crossentropy',name='targets',dtype='float32')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def neural_tensorflow_model(data):

    #Define Layers
    hiddenLayer_1 = {'weights': tensorflow.Variable(tensorflow.random_normal([32, 32, 3, numberOfNodesHiddenLayer_1])),
                    'biases': tensorflow.Variable(tensorflow.random_normal(numberOfNodesHiddenLayer_1))}
    hiddenLayer_2 = {'weights': tensorflow.Variable(tensorflow.random_normal([numberOfNodesHiddenLayer_1, numberOfNodesHiddenLayer_2])),
                     'biases': tensorflow.Variable(tensorflow.random_normal(numberOfNodesHiddenLayer_2))}
    hiddenLayer_3 = {'weights': tensorflow.Variable(tensorflow.random_normal([numberOfNodesHiddenLayer_2, numberOfNodesHiddenLayer_3])),
                     'biases': tensorflow.Variable(tensorflow.random_normal(numberOfNodesHiddenLayer_3))}
    outputLayer = {'weights': tensorflow.Variable(tensorflow.random_normal([numberOfNodesHiddenLayer_3, numberOfOutputs])),
                     'biases': tensorflow.Variable(tensorflow.random_normal(numberOfOutputs))}

    #Connect Layers
    layer1 = tensorflow.add(tensorflow.matmul(data,hiddenLayer_1['weights']),hiddenLayer_1['biases'])
    layer1 = tensorflow.nn.relu(layer1)
    layer2 = tensorflow.add(tensorflow.matmul(layer1, hiddenLayer_2['weights']), hiddenLayer_2['biases'])
    layer2 = tensorflow.nn.relu(layer2)
    layer3 = tensorflow.add(tensorflow.matmul(layer2, hiddenLayer_3['weights']), hiddenLayer_3['biases'])
    layer3 = tensorflow.nn.relu(layer3)
    output = tensorflow.matmul(layer3,outputLayer['weights'] + outputLayer['biases'])

    return output


def train_neural_tensorflow_network(x):
    prediction = neural_tensorflow_model(x)
    cost = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tensorflow.train.AdamOptimizer().minimize(cost)

    numberOfEpochs = 5
    numberOfExamples = 10

    with tensorflow.Session as sess:
        sess.run(tensorflow.initialize_all_variables())

        for epoch in numberOfEpochs:
            epochLoss = 0
            for _ in range(int(numberOfExamples/batchSize)):
                x,y = 1,1 # training data
                _,c = sess.run([optimizer,cost], feed_dict={x: x, y: y})
                epochLoss += c
            print('Epoch', epoch, 'completed out of', numberOfEpochs, 'loss:', epochLoss)

        correct = tensorflow.equal(tensorflow.argmax(prediction,1), tensorflow.argmax(y,1))
        accuracy = tensorflow.reduce_mean(tensorflow.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:training_x,y:training_y}))









def transform_to_array(data):
    X = []
    for row in data:
        x_entry = []
        for entry in row[0]:
            for color in entry:
                x_entry.append(color[0])
                x_entry.append(color[1])
                x_entry.append(color[2])
        X.append(x_entry)

    return X

def transform_to_array_1(data):
    X = []
    for row in data:
        for column in row:
            for entry in column:
                for color in entry:
                    X.append(color)


    return X

def train_model(training_data, model=False):
    #X = transform_to_array(training_data)
    X = np.array([i[0] for i in training_data ])
    y = [transformActionToTarget(i[1]) for i in training_data]

    if not model:
        model = neural_network_model()


    model.fit({'input': X}, {'targets': y}, n_epoch=3, snapshot_step=500, show_metric=True,
              run_id='openaistuff')
    return model



#some_random_games_first()


#####################################
###START
##########




initial_training_data = initial_population()
model = train_model(initial_training_data)

scores = []
choices = []

with tensorflow.Session() as sess:
    actualMaxScore = 0
    maxScore = 0
    for each_game in range(numberOfGames):
        score = 0
        negativCounter = 0
        game_memory = []
        training_data = []
        prev_observations = []
        env.reset()

        print('New Game')
        for _ in range(goal_steps):
            env.render()
            #if (each_game%5)==0:
                #env.render()
            if len(prev_observations) == 0:
                action = generateRandomAction()
            else:
                #prev_observations = transform_to_array_1(prev_observations)
                #prev_observations = np.reshape(prev_observations,(1,3072))
                action = model.predict(prev_observations)[0]
                print(action)
            choices.append(action)

            new_observation, reward, done, info = env.step(transformPredictionToAction(action))
            new_observation = numpy.array(new_observation)[51:83,32:64]
            prev_observations = np.array([i for i in new_observation]).reshape(1,32,32,3)
            game_memory.append([new_observation, action])
            score += reward
            if reward < 0:
                negativCounter += 1
            else:
                negativCounter = 0


            print('Score: ', score, 'actualMaxScore:', actualMaxScore, ' maxScore:', maxScore)

            # If car is improving save how it drove
            if score > actualMaxScore:
                actualMaxScore = score


            if done or negativCounter > 50*NEGATIVE_REWARDS_THRESHOLD:
                break

        # If the gamescore was high enough save how you played
        #if abs(actualMaxScore - maxScore) < 3 or maxScore == 0:
        maxScore = actualMaxScore
        for data in game_memory:
            training_data.append([np.array(data[0]), np.array(data[1])])
        #Retrain model with new data
        #model = train_model(training_data,model)

        scores.append(score)

    #Show Performance of Model
    print('Average accepted score: ', mean(scores))
    #print('Choice 1: {}, Choice 2: {}'.format(choices.count(1) / len(choices),
    #                                      choices.count(0) / len(choices)))

    model.save('NN_1.model')
