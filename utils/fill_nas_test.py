import pandas as pd
import numpy as np

n = float('nan')
a = {'a':[1,2,3,n,5,8,22,n], 'b':[9,12,14,16,n,n,8,19]}
d = pd.DataFrame(a)

d_mean = d.mean()
d_std = d.std()
