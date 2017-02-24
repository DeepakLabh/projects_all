'''Problem 2: Write a program that takes one or more filenames as arguments and prints all the lines which are longer than 40 characters.'''
def read_print(filenames):
    for f in filenames:
	for line in open(f):
	    if len(line)>40:
		yield line

for i in read_print(['problem1_extended.py', 'problem1.py']):
    print i

#class read_file:
#    def __init__(self, filenames):
#	self.filenames = filenames
#	self.i = 0
#	self.l = len(filenames)
#
#    def __iter__(self):
#	return self
#
#    def next(self):
#	filenames = self.filenames
#	i = self.i
#	l = self.l
#	#print (i,l,filenames[i], self.i,self.l)
#	if i<l:
#	    self.i += 1
#	    return filenames[i]
#	else:
#	    raise StopIteration()
#
#class read_lines:
#    def __init__(self, filename):
#	self.filename = filename
#	self.i = 0
#	#self.l = len()
#
#    def __iter__(self):
#	return self
#
#    def next(self):
#	filename = self.filename
#	with open(filename) ad f:   
# 
