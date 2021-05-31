import matlab.engine

eng = matlab.engine.start_matlab()
a = eng.kong_count(1.0, 5.0)
print(a)

b = eng.sqrt(4.)
print(b)