import numpy
import matplotlib

import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torchvision import datasets, models, transforms

######################################################################
#*************************** init weight ****************************#
def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1:
		init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
		init.constant_(m.bias.data, 0.0)
	elif classname.find('BatchNorm1d') != -1:
		init.normal(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)
	elif classname.find('BatchNorm2d') != -1:
		init.constant_(m.weight.data, 1)
		init.constant_(m.bias.data, 0)


def weights_init_classifier(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		init.normal_(m.weight.data, std=0.001)
		init.constant_(m.bias.data, 0.0)
#*************************** init weight ****************************#
######################################################################
'''
PCB Process：
	Step 1： 修改ResNet Backbone最后两层的全局Pooling层变成 局部的Pooling层，以此讲图像的Feature Map水平分成 N=6 块
	Step 2： 对每一块进行 FC module的叠加进行预测结果
	Step 3： 对 N=6 个结果进行叠加求取Loss
	Step 4： RPP

'''
ininputs = 1280
######################################################################
#*************************** RPP layers *****************************#
class RPP(nn.Module):
	def __init__(self):
		super(RPP, self).__init__()
		self.part = 6
		
		self.add_block = nn.Sequential(
			nn.Conv2d(ininputs, 6, kernel_size=1, bias=False),
		)
		self.add_block.apply(weights_init_kaiming)

		self.norm_block = nn.Sequential(
			nn.BatchNorm2d(ininputs),
			nn.ReLU(inplace=True),
		)
		self.norm_block.apply(weights_init_kaiming)

		self.softmax = nn.Softmax(dim=1)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


	def forward(self, x):
		w = self.add_block(x)
		p = self.softmax(w)
		y = []
		for i in range(self.part):
			p_i = p[:, i, :, :]
			p_i = torch.unsqueeze(p_i, 1)
			y_i = torch.mul(x, p_i)
			y_i = self.norm_block(y_i)
			y_i = self.avgpool(y_i)
			y.append(y_i)

		f = torch.cat(y, 2)
		return f
#*************************** RPP layers *****************************#
######################################################################


######################################################################
#************************ PCB baseline  *****************************#
class ClassBlock(nn.Module):
	def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
		super(ClassBlock, self).__init__()

		self.add_block = nn.Sequential(
			nn.Conv2d(input_dim, num_bottleneck, kernel_size=1, bias=False), 
			nn.BatchNorm2d(num_bottleneck),
			nn.ReLU(inplace=True),
		)
		self.add_block.apply(weights_init_kaiming)

		self.classifier = nn.Sequential(
			nn.Linear(num_bottleneck, class_num)
		)
		self.classifier.apply(weights_init_classifier)

	def forward(self, x):
		x = self.add_block(x)
		x = torch.squeeze(x)
		x = self.classifier(x)
		return x

class PCB(nn.Module):
	def __init__(self, class_num):
		super(PCB, self).__init__()
		# attributes
		self.class_num = class_num
		self.part = 6
		# pre-definition of network function
		# pre-define Step 1
		mobile = models.mobilenet_v2(pretrained=True)
		# resnet.layer4[0].downsample[0].stride = (1,1)
		# resnet.layer4[0].conv1.stride = (1,1)
		modules = list(mobile.children())[:-1]
		self.backbone = nn.Sequential(*modules)
		self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
		self.dropout = nn.Dropout(p=0.5)
		# define six classifiers
		# pre-define Step 2
		self.classifiers = nn.ModuleList()
		for i in range(self.part):
			self.classifiers.append(ClassBlock(ininputs, self.class_num, True, 256))

	def forward(self, x):
		x = self.backbone(x)
		# [BatchSize, 512, 14, 14]
		x = self.avgpool(x)
		x = self.dropout(x)
		# Step 3
		predict = []
		for i in range(self.part):
			tmp = x[:, :, i, :]
			tmp = torch.unsqueeze(tmp, 3)
			predict.append(
				self.classifiers[i](tmp)
			)
		# return predict
		return predict	

	def convert_to_rpp(self):
		self.avgpool = RPP()
		return self
#************************ PCB baseline  *****************************#
######################################################################

######################################################################
#********************** PCB forward inferece ************************#
class PCB_test(nn.Module):
	def __init__(self, model, featrue_H=False):
		super(PCB_test, self).__init__()
		self.part = 6
		self.featrue_H = featrue_H
		self.backbone = model.backbone
		self.avgpool = model.avgpool
		self.classifiers = nn.ModuleList()
		for i in range(self.part):
			self.classifiers.append(model.classifiers[i].add_block)

	def forward(self, x):
		x = self.backbone(x)
		x = self.avgpool(x)

		if self.featrue_H:
			predict = []
			
			for i in range(self.part):
				tmp = x[:, :, i, :]
				tmp = torch.unsqueeze(tmp, 3)
				predict.append(
					self.classifiers[i](tmp)
				)

			x = torch.cat(predict, 2)
		f = x.view(x.size(0), x.size(1), x.size(2))
		return f
#********************** PCB forward inferece ************************#
######################################################################

# 25557032 resnet
#  3504872 mobilenet_v2
#  5351066 mobilenet_v2 + PCB without RPP
#  5361306 mobilenet_v2 + PCB RPP
