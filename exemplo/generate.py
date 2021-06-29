import numpy as np

N = 1000
data = []
for i in range(N):
    x = np.random.random()*10 - 5
    y = x**2
    print(x,y)
    data.append((x,y))

with open('data.txt','w') as f:
    f.write('\n'.join(f'{x} {y}' for x, y in data))



