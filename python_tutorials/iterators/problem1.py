class rev_range:
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
