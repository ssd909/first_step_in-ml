import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input,Embedding

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')


text = open(path_to_file, 'rb').read().decode(encoding='utf-8')


sorted_text=sorted(list(set(text)))
char_to_int={ch:i for i,ch in enumerate(sorted_text)}
int_to_char={i:ch for i,ch in enumerate(sorted_text)}
cont_sequence=[]
target_ch=[]

for i in range(0, len(text) - 40,3):
   cont_sequence.append(text[i : i + 40])
   target_ch.append(text[i + 40])

x=np.zeros((len(cont_sequence),40),dtype=int)
y=np.zeros((len(cont_sequence),len(sorted_text)),dtype=bool)
for i, seq in enumerate(cont_sequence):
    for t, char in enumerate(seq):
        x[i, t]= char_to_int[char]
        y[i, char_to_int[target_ch[i]]] = 1

model = Sequential([

    Embedding(65, 32, input_length=40),

    LSTM(128, return_sequences=False),


    Dense(65, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x,y,epochs=1)
start=text[100:140]
generate=''
for j in range(100):
  x_input=np.zeros((1,40))
  for t, char in enumerate(start[-40:]):
      x_input[0, t] = char_to_int[char]
  y_pred = model.predict(x_input)
  max_index = np.argmax(y_pred)
  start+=int_to_char[max_index]
  generate+=int_to_char[max_index]
print(generate)
