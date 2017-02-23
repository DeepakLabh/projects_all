'''Problem 2: Write a program that takes one or more filenames as arguments and prints all the lines which are longer than 40 characters.'''

class read_file:
    def __init__(self, filenames):
	self.filenames = filenames
	self.i = 0
	self.l = len(filenames)

    def __iter__(self):
	return self

    def next(self):
	filenames = self.filenames
	i = self.i
	l = self.l
	if i<l:
	    self.i += 1
	    return filenames[i]
	else:
	    StopIteration()

