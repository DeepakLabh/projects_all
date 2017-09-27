from multiprocessing import Pool, Process
import os
'''
Pool: Distribute input data across multiple processes
Process: running on different parallel threads across all processes. (It follows the API of threading.Thread)
Queues: Sharing data or objects between two processes, Queues are threads and process safe

'''
def f(x):
    return x*x


def info(title):
    print title
    print 'module name:', __name__
    if hasattr(os, 'getppid'):  # only available on Unix
        print 'parent process:', os.getppid()
    print 'process id:', os.getpid()

def f1(name):
    info('function f')
    print 'hello', name

if __name__ == '__main__':
    ################# To check Pool #############
    #p = Pool(1) # args as no of processes
    #print(p.map(f, xrange(10)))
    ################# To check Pool #############
    ################### To check Process ##############
    p = Process(target = f1, args = ('bob',))
    p.start()
    p.join()
    ################### To check Process ##############
