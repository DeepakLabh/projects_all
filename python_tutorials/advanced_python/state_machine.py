'''http://www.python-course.eu/finite_state_machine.php
Implement state machine
'''

class state_machine:
    def __init__(self):
	self.handlers = {}
	self.start_state = None
	self.end_state = []

    
