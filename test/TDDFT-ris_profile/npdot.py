# -*- coding: utf-8 -*-

import numpy as np
import time

start = time.time()
a = np.random.rand(1000000,1000)
b = np.random.rand(1000,200)

c = np.dot(a, b)
end = time.time()

print('cost = ', end - start)
