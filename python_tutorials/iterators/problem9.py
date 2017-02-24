'''The built-in function enumerate takes an iteratable and returns an iterator over pairs (index, value) for each value in the source.'''

class my_enumerate:
    def __init__(self, arg):
	self.arg = arg
	self.i = 0

    def __iter__(self):
	return self

    def next(self):
	i = self.i
	if i<len(self.arg):
	    #print i,self.arg,1111111111
	    self.i += 1
	    return (i, self.arg[i])
	else:
	    raise StopIteration()


if __name__ == '__main__':
    a = my_enumerate([1,3,5,6])
    print a,list(a)
