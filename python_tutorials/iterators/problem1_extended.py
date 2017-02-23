'''Problem 1: Write an iterator class reverse_iter, that takes a list and iterates it from the reverse direction. ::
   The items does not vanish after first call
'''

class rev_range:
    def __init__(self, n):
	self.n = n
	#self.i = n 
    
    def __iter__(self):
	return rev_iter(self.n)



class rev_iter:
    def __init__(self, n):
	self.n = n
	self.i = n

    def __iter__(self):
	return self

    def next(self):
	i = self.i
	if self.i > 0 :
	    self.i -= 1
	    return i 
	else:
	    raise StopIteration() 
