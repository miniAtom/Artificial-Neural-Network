# -*- coding: utf-8 -*-
# @Author: GWL
# @Date:   2022-07-19 17:47:19
# @Last Modified by:   GWL
# @Last Modified time: 2022-07-19 18:24:54

import imageio
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import special


class neuralNetwork(object):

	def __init__(self,
					in_nodes=784,
					hide_nodes=520,
					out_nodes=10,
					learning_rate=0.1,
					epoch=5):
		self.in_nodes = in_nodes
		self.hide_nodes = hide_nodes
		self.out_nodes = out_nodes

		# 输入层到隐藏层的权重
		self.wih = np.random.normal(0.0, pow(self.in_nodes, -0.5),
									(self.hide_nodes, self.in_nodes))
		# 隐藏层到输出层的权重
		self.who = np.random.normal(0.0, pow(self.hide_nodes, -0.5),
									(self.out_nodes, self.hide_nodes))

		# 学习率
		self.learning_rate = learning_rate

		# 激活函数sigmoid
		self.activation_function = lambda x: special.expit(x)
		self.inverse_activation_function = lambda x: special.logit(x)

		# 训练代数
		self.epoch = epoch

		# 训练数据
		self.train_data_list = []

		# 测试数据
		self.test_data_list = []

		# 当前的上一级路径
		self.path = os.path.abspath(os.path.dirname(os.getcwd()))

	def load_data(self, train_set_path, test_set_path):
		# train_set_path = "/MNIST-csv-mini/train.csv"
		# train_set_path = "MNIST-csv-all/mnist_train.csv"
		with open(train_set_path, 'r') as f:
			train_data_list = f.readlines()

		# test_set_path = "/MNIST-csv-mini/test.csv"
		# test_set_path = "MNIST-csv-all/mnist_test.csv"
		with open(test_set_path, 'r') as f:
			test_data_list = f.readlines()

		self.train_data_list = train_data_list
		self.test_data_list = test_data_list
		return len(train_data_list), len(test_data_list)

	# 训练数据
	def train(self, input_list, target_list):
		inputs = np.array(input_list, ndmin=2).T
		targets = np.array(target_list, ndmin=2).T

		hide_inputs = np.dot(self.wih, inputs)
		hide_outputs = self.activation_function(hide_inputs)

		final_inputs = np.dot(self.who, hide_outputs)
		final_outputs = self.activation_function(final_inputs)

		output_errors = targets - final_outputs
		hide_errors = np.dot(self.who.T, output_errors)

		# 更新权重
		self.who += self.learning_rate * np.dot(
			(output_errors * final_outputs *
				(1.0 - final_outputs)), np.transpose(hide_outputs))
		self.wih += self.learning_rate * np.dot(
			(hide_errors * hide_outputs *
				(1.0 - hide_outputs)), np.transpose(inputs))

	# 预测数据
	def predict(self, input_list):
		inputs = np.array(input_list, ndmin=2).T
		hide_inputs = np.dot(self.wih, inputs)
		hide_outputs = self.activation_function(hide_inputs)
		final_inputs = np.dot(self.who, hide_outputs)
		final_outputs = self.activation_function(final_inputs)
		return final_outputs

	# 神经网络反向传播
	def backpredict(self, targets_list):
		final_outputs = np.array(targets_list, ndmin=2).T
		final_inputs = self.inverse_activation_function(final_outputs)

		# 归一化处理到区间[0.01, 0.99]
		hidden_outputs = np.dot(self.who.T, final_inputs)
		hidden_outputs -= np.min(hidden_outputs)
		hidden_outputs /= np.max(hidden_outputs)
		hidden_outputs *= 0.98
		hidden_outputs += 0.01

		hidden_inputs = self.inverse_activation_function(hidden_outputs)
		inputs = np.dot(self.wih.T, hidden_inputs)
		inputs -= np.min(inputs)
		inputs /= np.max(inputs)
		inputs *= 0.98
		inputs += 0.01

		return inputs

	def train_data(self):
		# 训练数据
		for e in range(self.epoch):
			for record in self.train_data_list:
				all_values = record.split(',')
				inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
				target = np.zeros(self.out_nodes) + 0.01
				target[int(all_values[0])] = 0.99
				self.train(inputs, target)
				print("\r", np.random.normal(), end="\r")
		print("                                  \r训练完成！")

	def test_data(self):
		scorecard = []
		right = 0
		total = 0
		for record in self.test_data_list:
			all_values = record.split(',')
			correct_label = int(all_values[0])
			inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
			outputs = self.predict(inputs)
			label = np.argmax(outputs)
			right += correct_label == label
			total += 1

		performance = right / total
		print("\rPerformance =", performance)
		return performance

	def train_and_test(self):
		self.train_data()
		performance = self.test_data()
		return performance

	def numbers_in_network(self):
		for i in range(self.out_nodes):
			targets = np.zeros(self.out_nodes) + 0.01
			targets[i] = 0.99
			# print(targets)
			image_data = self.backpredict(targets)
			plt.imshow(image_data.reshape(28, 28),
						cmap="Greys",
						interpolation="None")
			plt.savefig(self.path + "/figures/{}_in_network.png".format(i))

	def identify_numbers(self, cnt=10):
		right = 0
		for i in range(cnt):
			img_array = imageio.imread(self.path +
										"/figures/手写-{}.png".format(i),
										as_gray=True)
			img_data = 255.0 - img_array.reshape(28 * 28)
			img_data = (img_data / 255.0 * 0.99) + 0.01

			plt.imshow(img_data.reshape(28, 28),
						cmap="Greys",
						interpolation="None")
			outputs = self.predict(img_data)

			label = np.argmax(outputs)
			print("network says: {}, actually is {}".format(label, i))
			right += label == i

		print("Accuracy:", right / cnt)

	def draw_picture(self,
						x=[],
						y=[],
						x_lable='x',
						y_lable='y',
						title='title'):
		plt.scatter(x, y)
		plt.plot(x, y)
		plt.xlabel(x_lable)
		plt.ylabel(y_lable)
		plt.title(title)
		plt.grid()
		plt.savefig("./figures/" + title + ".png")
		plt.show()
		plt.close()

	def different_learning_rate(self):
		x = []
		y = []
		origin_learning_rate = self.learning_rate
		origin_wih = self.wih
		origin_who = self.who

		self.wih = np.random.normal(0.0, pow(self.in_nodes, -0.5),
									(self.hide_nodes, self.in_nodes))
		self.who = np.random.normal(0.0, pow(self.hide_nodes, -0.5),
									(self.out_nodes, self.hide_nodes))

		self.learning_rate = 0.05
		while self.learning_rate < 1:
			x.append(self.learning_rate)
			y.append(self.train_and_test())
			self.learning_rate += 0.05
		print(x, "\n", y)
		self.draw_picture(x, y, "learning rate", "lerformance",
							"Performance varies with learning rate")
		self.learning_rate = origin_learning_rate
		self.wih = origin_wih
		self.who = origin_who

	def different_epoch(self):
		x = []
		y = []
		origin_epoch = self.epoch
		origin_wih = self.wih
		origin_who = self.who

		self.wih = np.random.normal(0.0, pow(self.in_nodes, -0.5),
									(self.hide_nodes, self.in_nodes))
		self.who = np.random.normal(0.0, pow(self.hide_nodes, -0.5),
									(self.out_nodes, self.hide_nodes))
		self.epoch = 1
		while self.epoch < 21:
			x.append(self.epoch)
			y.append(self.train_and_test())
			self.epoch += 1
		print(x, "\n", y)
		self.draw_picture(x, y, "epoch", "performance",
							"Performance varies with epoch")
		self.epoch = origin_epoch
		self.wih = origin_wih
		self.who = origin_who

	def different_hide_nodes(self):
		x = []
		y = []
		origin_hide_nodes = self.hide_nodes
		origin_wih = self.wih
		origin_who = self.who

		self.wih = np.random.normal(0.0, pow(self.in_nodes, -0.5),
									(self.hide_nodes, self.in_nodes))
		self.who = np.random.normal(0.0, pow(self.hide_nodes, -0.5),
									(self.out_nodes, self.hide_nodes))

		self.hide_nodes = 20
		while self.hide_nodes < 520:
			x.append(self.hide_nodes)
			y.append(self.train_and_test())
			if self.hide_nodes <= 200:
				self.hide_nodes += 20
			else:
				self.hide_nodes += 50
		print(x, "\n", y)
		self.draw_picture(x, y, "number of hidden nodes", "performance",
							"Performance varies with hidden nodes")
		self.hide_nodes = origin_hide_nodes
		self.wih = origin_wih
		self.who = origin_who


def main():
	n = neuralNetwork()

	# 获取当前的上一级路径
	path = os.path.abspath(os.path.dirname(os.getcwd()))
	train_set_path = path + "/MNIST-csv-mini/train.csv"
	# train_set_path = path + "/MNIST-csv-all/mnist_train.csv"

	test_set_path = path + "/MNIST-csv-mini/test.csv"
	# test_set_path = path + "/MNIST-csv-all/mnist_test.csv"

	n.load_data(train_set_path, test_set_path)

	n.train_and_test()

	n.numbers_in_network()

	n.identify_numbers()

	n.different_epoch()

	n.different_hide_nodes()

	n.different_learning_rate()


if __name__ == "__main__":
	main()
