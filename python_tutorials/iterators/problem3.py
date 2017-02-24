'''Problem 3: Write a function "findfiles" that recursively descends the directory tree for the specified directory and generates paths of all the files in the tree.'''

import os

def findfiles(path):
    for i in os.listdir(path):
	#print i
	f_path = os.path.join(path, i)
	if not os.path.isdir(f_path):
	    yield f_path
	else:
	    #print 'here'
	    for j in findfiles(f_path):
		yield j
	    #print list(a),11111111

if __name__ == '__main__':
    a = findfiles('../')
    print list(a)
