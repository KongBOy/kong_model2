import numpy  as np
import matlab
import matlab.engine

Nn = 30
x = 250 * np.ones((1, Nn))
y = 100 * np.ones((1, Nn))
zz = matlab.double([[32]])
xx = matlab.double(x.tolist())
yy = matlab.double(y.tolist())
eng = matlab.engine.start_matlab()

result = eng.kong_count(xx, yy, zz)
print("xx", xx)
print("yy", yy)
print("zz", zz)
print("result", result)

b = eng.sqrt(4.)
print(b)
