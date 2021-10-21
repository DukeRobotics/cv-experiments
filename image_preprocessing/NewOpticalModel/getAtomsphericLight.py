import numpy as np

# 用于排序时存储原来像素点位置的数据结构
class Node(object):
	def __init__(self,x,y,value):
		self.x = x
		self.y = y
		self.value = value
	def printInfo(self):
		print(self.x,self.y,self.value)

def getAtomsphericLight(darkChannel, img):
    # img = np.float16(img)
    height = darkChannel.shape[0]
    width = darkChannel.shape[1]
    nodes = []
    # 用一个链表结构(list)存储数据
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, darkChannel[i, j])
            nodes.append(oneNode)
    # 排序
    nodes = sorted(nodes, key=lambda node: node.value, reverse=False)
    atomsphericLight  = img[nodes[0].x, nodes[0].y, :]
    # atomsphericLight  =  [img[nodes[0].x, nodes[0].y, 0],img[nodes[0].x, nodes[0].y, 1]]
    return atomsphericLight
