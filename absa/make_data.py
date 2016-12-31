import re
from copy import deepcopy
import json

a = open('ABSA16_Restaurants_Train_SB1_v2.xml','r').read()
a = a.replace('\n',' ').replace('\r',' ')

p = re.findall(r'<sentence(.*?)</sentence>',a)
data = []
edict = {}
for i in p:
	edict['sentence'] = re.findall(r'<text>(.*?)</text>', i)[0]
	catagory = re.findall(r'category="(.*?)"', i)
	polarity = re.findall(r'polarity="(.*?)"', i)
	for j in xrange(len(catagory)):
		edict['attribute'] = catagory[j].split('#')[0]
		edict['polarity'] = polarity[0]
		ddict = deepcopy(edict)
		data.append(ddict)

json.dump(data, open('restraunt_data.json', 'w'))