'''Implement a function izip that works like itertools.izip.'''

class my_izip:
    def __init__(self,arg1,arg2):
	self.arg1 = arg1
	self.arg2 = arg2
	self.i = 0
	self.l = min(len(self.arg1), len(self.arg2))

    def __iter__(self):
	return self

    def next(self):
	i = self.i
        l = self.l
	if i<l:	
	    self.i +=1
	    #print i,l
	    return (self.arg1[i], self.arg2[i])
	else:
	    raise StopIteration()


if __name__ == '__main__':
    a = my_izip([1,3,23,43,2], [23,45,34])
    print a,list(a)
