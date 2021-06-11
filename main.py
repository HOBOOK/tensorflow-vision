import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = x * 2 + 1

print(x)

print(y)

W = tf.Variable(tf.random.uniform(shape=(3,3)), name="W")
b = tf.Variable(tf.random.uniform(shape=(3,1)), name="b")

@tf.function
def forward(x):
  return W * x + b

out_a = forward([[1,2,3],[4,5,6],[7,8,9]])

 
# 가충치값 출력
print("============== W ==============")
tf.print(W)
 
# 편향값 출력
print("============== b ==============")
tf.print(b)
 
# 최종결과
print("============== hypothesis ==============")
tf.print(out_a)


model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.compile('SGD', 'mse')

model.fit(x, y, epochs=100, verbose=1)

print('y:', y, ',predict:', model.predict(x).flatten())