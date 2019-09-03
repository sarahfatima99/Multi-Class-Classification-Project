#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


# In[2]:


seed=7
np.random.seed(seed)


# In[3]:


dataframe=pandas.read_csv("iris.data",header=None)
dataset=dataframe.values

x=dataset[:,0:4].astype(float)
y=dataset[:,4]
print(y.shape)


# In[4]:


encoder=LabelEncoder()
encoder.fit(y)
yenc=encoder.transform(y)
dummy_y=np_utils.to_categorical(yenc)
#print(yenc)
#print(dummy_y)
#print(x)


# In[5]:


def build_model():
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(4,)))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    return model
estimator=KerasClassifier(build_fn=build_model,epochs=200,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,x,dummy_y,cv=kfold)
print("Baseline:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))


# In[ ]:





# # SMALLER MODEL
# 

# In[6]:


def build_model():
    model = Sequential()
    model.add(Dense(1, activation='relu', input_shape=(4,)))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    return model
estimator=KerasClassifier(build_fn=build_model,epochs=200,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,x,dummy_y,cv=kfold)
print("Baseline:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))


# # LARGER MODEL

# In[7]:


def build_model():
    model = Sequential()
    model.add(Dense(90, activation='relu', input_shape=(4,)))
    model.add(Dense(87, activation='relu'))
    model.add(Dense(207, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    return model
estimator=KerasClassifier(build_fn=build_model,epochs=200,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,x,dummy_y,cv=kfold)
print("Baseline:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))


# # OVERFITTING MODEL
# 

# In[8]:


def build_model():
    model = Sequential()
    model.add(Dense(90, activation='relu', input_shape=(4,)))
    model.add(Dense(87, activation='relu'))
    model.add(Dense(207, activation='relu')) 
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
    return model
estimator=KerasClassifier(build_fn=build_model,epochs=700,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,x,dummy_y,cv=kfold)
print("Baseline:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))


# # TUNING THE MODEL

# In[9]:


def build_model():
    model = Sequential()
    model.add(Dense(90, activation='relu', input_shape=(4,)))
    model.add(Dense(87, activation='relu'))
    model.add(Dense(207, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['acc'])
    return model
estimator=KerasClassifier(build_fn=build_model,epochs=200,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,x,dummy_y,cv=kfold)
print("Baseline:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))
def build_model():
    model = Sequential()
    model.add(Dense(90, activation='relu', input_shape=(4,)))
    model.add(Dense(87, activation='relu'))
    model.add(Dense(207, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop',
    loss='mse',
    metrics=['acc'])
    return model
estimator=KerasClassifier(build_fn=build_model,epochs=200,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,x,dummy_y,cv=kfold)
print("Baseline:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))
def build_model():
    model = Sequential()
    model.add(Dense(90, activation='relu', input_shape=(4,)))
    model.add(Dense(87, activation='tanh'))
    model.add(Dense(207, activation='tanh'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['acc'])
    return model
estimator=KerasClassifier(build_fn=build_model,epochs=200,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,x,dummy_y,cv=kfold)
print("Baseline:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))


# # Functional API

# In[10]:


from keras.models import Model
from keras.layers import Input, Dense
def build_model():
    input_tensor = Input(shape=(4,))
    x = Dense(8, activation='relu')(input_tensor)
    x = Dense(3, activation='relu')(x)
    output_tensor = Dense(3, activation='softmax')(x)
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
estimator= KerasClassifier(build_fn=build_model,epochs=100,batch_size=5,verbose=0)
kfold=KFold(n_splits=10,shuffle=True,random_state=seed)
results=cross_val_score(estimator,x,dummy_y,cv=kfold)
print("Results:%.2f%%(%.2f%%)"%(results.mean()*100,results.std()*100))


# # WITH KERAS

# In[11]:



#x=dataset[:,0:4].astype('float32')
np.random.shuffle(dummy_y)
np.random.shuffle(x)
#flower = 'Iris-setosa','Iris-versicolor','Iris-virginica'
#char_to_int = dict((c, i) for i, c in enumerate(flower))
#z= [char_to_int[char] for char in data]
#print(z)
train_dataset = x[:100]
test_dataset = x[100:]
train_label = dummy_y[:100]
test_label = dummy_y[100:]
#print(dummy_y)


# In[12]:


from keras import optimizers
def build_model():
    model = Sequential()
    model.add(Dense(4, activation='relu', input_shape=(4,)))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='categorical_crossentropy' , metrics=["accuracy"])
    return model


# In[13]:




k=4
num_val_samples = len(train_dataset) // k
num_epochs = 200
#print(num_val_samples)

for i in range(k):
    print('processing fold #', i)
    val_data = train_dataset[i * (num_val_samples): (i + 1) * (num_val_samples)]
    val_targets = train_label[i * (num_val_samples): (i + 1) * (num_val_samples)]
    partial_train_data = np.concatenate(
    [train_dataset[:i * num_val_samples],
    train_dataset[(i + 1) * num_val_samples:]],
    axis=0)
    partial_train_targets = np.concatenate(
    [train_label[:i * num_val_samples],
    train_label[(i + 1) * num_val_samples:]],
    axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1, verbose=0)
    loss , acc = model.evaluate(val_data, val_targets)
    print(acc)


# In[14]:



def build_model():
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=(4,)))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='categorical_crossentropy' , metrics=["accuracy"])
    return model
model = build_model()
model.fit(train_dataset, train_label , epochs=250, batch_size=5 , verbose=1)


# # MODEL SUBCLASSING

# In[15]:


import keras
class build_model(keras.Model):
    def __init__(self):
        super(build_model,self).__init__()
        inputs = (4,)
        self.dense1=Dense(8,activation='relu')
        self.dense2=Dense(3,activation='softmax')
       
    def call(self,inputs):
        x=self.dense1(inputs)
        return self.dense2(x)
def finalModel():
    model=build_model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn = finalModel  , epochs = 200, batch_size = 5 , verbose = 0 )
kfold = KFold(n_splits = 10 , shuffle = True , random_state = seed)
results = cross_val_score(estimator , x , dummy_y , cv = kfold)
print("Accuracy: %2f%%(%2f%%)"%(results.mean()*100,results.std()*100))


# In[ ]:





# In[ ]:




