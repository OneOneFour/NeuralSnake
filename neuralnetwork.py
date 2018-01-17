import numpy as np
import names
import time
import snake
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layer_setup, name, family, bias=None, weights=None):
        '''
        Create neural network with blank parameters for snake
        '''
        self.first_name = name
        self.family = family
        self.layer_count = len(layer_setup)
        self.layer_setup = layer_setup
        if weights is None:
            self.weights = [np.random.randn(y, x) for x, y in zip(layer_setup[:-1], layer_setup[
                                                                                    1:])]  # i think you can use zip() here but this is easier to understand
        else:
            self.weights = weights
        if bias is None:
            self.bias = [np.random.randn(y) for y in layer_setup[1:]]
        else:
            self.bias = bias

    def get_weight_size(self):
        return np.sum([np.size(x) for x in self.weights])

    def get_bias_size(self):
        return np.sum([np.size(x) for x in self.weights])

    def out(self, input):
        '''
        Get output result from neural network
        :param input:Vector of
        input neurons
        :return: Vector of output neurons
        '''
        if len(input) != self.layer_setup[0]:
            raise Exception("Input dimens much match network input dimens")
        current_layer = input
        for i in range(self.layer_count - 1):
            current_layer = self.sigmoid(np.dot(self.weights[i], current_layer) + self.bias[i])

        return current_layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# class CalclusClass:
#     def __init__(self):
#         self.bot = NeuralNetwork([20**2, 16, 16, 10], names.get_first_name(),names.get_last_name())
#         self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#
#     def test_set(self, start, size):
#         average_cost = 0
#
#         score = 0
#         for i in range(size):
#             result = self.bot.out(self.mnist.train.images[i])
#             true = self.mnist.train.labels[i]
#             if np.argmax(result) == np.argmax(true):
#                 score += 1
#             cost = np.sum((result - true) ** 2)
#             average_cost += cost
#             # Calculate gradient
#

class Classroom:
    def __init__(self):
        # Load training images
        self.mixing = 0.5
        self.mutation = 0.05
        self.population = 20
        self.gen = 0
        #self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        self.brains = [NeuralNetwork([20 ** 2, 6, 3], names.get_first_name(), names.get_last_name()) for i in
                       range(self.population)]
        plt.ion()
        plt.show()
        self.fig = plt.figure()
        self.bestAI = self.brains[0], 0
        self.best = []
        self.avg = []
        self.ax1 = plt.gca()
        self.ax1.set_ylim(0, 100)
        self.line, = self.ax1.plot([], self.best, 'b-')
        self.line2, = self.ax1.plot([], self.avg, 'r-')
        time.sleep(0.5)
        self.trial_all()

    def trial_all(self):
        while self.bestAI[1] < 50:
            scores = []
            for network in self.brains:
                scores.append((network, self.trial_snake(network)))
            scores.sort(key=lambda x: x[1], reverse=True)
            self.print_scoreboard(scores)
            if scores[0][1] > self.bestAI[1]:
                self.bestAI = scores[0]

            self.brains = []
            while len(self.brains) < self.population:
                a = self.tournament_selection(scores,2)
                b = self.tournament_selection(scores,2)
                if a is b: continue

                ab_weights, ab_bias, ba_weights, ba_bias = self.swap_genome(a[0], b[0])
                family_name = a[0].family if a[1] > b[1] else b[0].family
                self.brains.append(
                    NeuralNetwork(a[0].layer_setup, names.get_first_name(), family_name, ab_bias, ab_weights))
                self.brains.append(
                    NeuralNetwork(b[0].layer_setup, names.get_first_name(), family_name, ba_bias, ba_weights))

            self.gen += 1

    def swap_genome(self, a, b):
        # unpack
        a_chromosone = np.concatenate(np.append([x.flatten() for x in a.weights], [x.flatten() for x in a.bias]))
        b_chromosone = np.concatenate(np.append([x.flatten() for x in b.weights], [x.flatten() for x in b.bias]))

        ab_chromosone, ba_chromosone = [], []
        for i in range(len(a_chromosone)):
            if np.random.uniform() > self.mixing:
                ba_chromosone.append(a_chromosone[i])
                ab_chromosone.append(b_chromosone[i])
            else:
                ba_chromosone.append(b_chromosone[i])
                ab_chromosone.append(a_chromosone[i])

        total = [a_chromosone, ab_chromosone, ba_chromosone, b_chromosone]
        for chromo in total:
            for i in range(len(chromo)):
                if np.random.uniform() < self.mutation:
                    chromo[i] = np.random.randn()

        aa_weights, aa_bias = a_chromosone[:a.get_weight_size()], a_chromosone[a.get_weight_size():]
        ab_weights, ab_bias = ab_chromosone[:a.get_weight_size()], ab_chromosone[a.get_weight_size():]
        ba_weights, ba_bias = ba_chromosone[:b.get_weight_size()], ba_chromosone[b.get_weight_size():]
        bb_weights, bb_bias = b_chromosone[:b.get_weight_size()], b_chromosone[b.get_weight_size():]

        aa_weights = self.reform_chromosone_weight(aa_weights, a.layer_setup)
        bb_weights = self.reform_chromosone_weight(bb_weights, b.layer_setup)
        ab_weights = self.reform_chromosone_weight(ab_weights, a.layer_setup)
        ba_weights = self.reform_chromosone_weight(ba_weights, b.layer_setup)
        aa_bias = self.reform_chromosone_bias(aa_bias, a.layer_setup)
        ab_bias = self.reform_chromosone_bias(ab_bias, a.layer_setup)
        ba_bias = self.reform_chromosone_bias(ba_bias, b.layer_setup)
        bb_bias = self.reform_chromosone_bias(bb_bias, b.layer_setup)

        return ab_weights, ab_bias, ba_weights, ba_bias

    def reform_chromosone_bias(self, bias, network_setup):
        values = network_setup[1:]
        end_bias = []
        for v in range(len(values)):
            end_bias.append(np.reshape(bias[int(np.sum(values[:v])):int(np.sum(values[:v + 1]))], (values[v],)))

        return end_bias

    def reform_chromosone_weight(self, weight, network_setup):
        values = [a * b for a, b in zip(network_setup, network_setup[1:])]
        end_weight = []
        for v in range(len(values)):
            end_weight.append(np.reshape(weight[int(np.sum(values[:v])):int(np.sum(values[:v + 1]))],
                                         list(zip(network_setup[1:], network_setup))[v]))

        return end_weight

    def print_scoreboard(self, scores):
        names = ["{0} {1}".format(k[0].first_name,k[0].family) for k in scores]
        scores = [s[1] for s in scores]
        self.best.append(scores[0])
        self.avg.append(np.mean(scores))

        print("------------------GENERATION {0} results---------------".format(self.gen))
        for i in range(len(scores)):
            print("{0}             Score: {1} %".format(names[i], scores[i]))
        print("------------------------------------------------------\n"
              "Average Score: {0} % \n"
              "------------------------------------------------------".format(np.mean(scores)))
        self.line.set_ydata(self.best)
        self.line.set_xdata(np.arange(0, self.gen + 1, 1))
        self.line2.set_ydata(self.avg)
        self.line2.set_xdata(np.arange(0, self.gen + 1, 1))
        self.ax1.set_xlim(0, self.gen + 1)
        self.ax1.set_ylim(0,np.max(scores) + 10)

        plt.draw()
        plt.pause(1e-17)
        time.sleep(0.001)

    # def trial_images(self, network):
    #     score = 0
    #     for i in range(self.num):
    #         result = network.out(self.mnist.test.images[i].reshape((28 ** 2,)))
    #         true = self.mnist.test.labels[i]
    #         if np.argmax(result) == np.argmax(true):
    #             score += 1
    #     return score

    def trial_snake(self,network):
        score = []
        snake.Game(network,score)
        return score[0]



    def tournament_selection(self, scores, k):
        current_winner = None
        for i in range(k):
            challenger = scores[np.random.randint(0, len(scores))]
            if current_winner is None or challenger[1] > current_winner[1]:
                current_winner = challenger
        return current_winner

c = Classroom()


# plt.ioff()
# while True:
#     i = np.random.randint(0, 10000)
#     result = c.bestAI[0].out(c.mnist.test.images[i].reshape((28 ** 2,)))
#     true = c.mnist.test.labels[i]
#     fig = plt.figure()
#     print(result)
#     plt.title("Value is {0}, my guess was {1}".format(np.argmax(true), np.argmax(result)))
#     plt.imshow(c.mnist.test.images[i].reshape((28, 28)), cmap='gray')
#     plt.show()
