import urllib.request
import os
#download dataset
url="http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
filepath="titanic3.xls"
if not os.path.isfile(filepath):
    result=urllib.request.urlretrieve(url,filepath)
    print('downloaded:',result)
#資料預處理
import numpy as np
import pandas as pd
all_df = pd.read_excel(filepath)

    
#去除沒用特徵
cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp',
        'parch', 'fare', 'embarked']
all_df = all_df[cols]
#all_df[2:5]
def PreprocessData(raw_df):
    #nan改成平均 性別轉數字 港口轉數字 姓名訓練時要去掉
    df = all_df.drop(['name'], axis=1)
    
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    
    df['sex'] = df['sex'].map({'female':0, 'male':1}).astype(int)
    
    x_OneHot_df = pd.get_dummies(data = df, columns=['embarked'])
    
    ndarray = x_OneHot_df.values
    label=ndarray[:,0]
    feature=ndarray[:,1:]
    #normalize
    from sklearn import preprocessing
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
    scaledFeature = minmax_scale.fit_transform(feature)
    return scaledFeature, label
#spilte dataset
msk = np.random.rand(len(all_df)) <0.8
train_df = all_df[msk]
test_df = all_df[~msk]

print('total:',len(all_df),
      'train:',len(train_df),
      'test:',len(test_df))

train_Feature, train_label = PreprocessData(train_df)
test_Feature, test_label = PreprocessData(test_df)

#modeling
from keras.models import Sequential
from keras.layers import Dense,Dropout
model = Sequential()
model.add(Dense(units=40,input_dim=9,
                kernel_initializer='uniform',
                activation='relu'))
model.add(Dense(units=30,
                kernel_initializer='uniform',
                activation='relu'))
model.add(Dense(units=1,
                kernel_initializer='uniform',
                activation='sigmoid'))
model.compile(loss='binary_crossentropy',
             optimizer='adam',metrics=['accuracy'])

train_history =model.fit(x=train_Feature,
                         y=train_label,
                         validation_split=0.1,
                         epochs=80,
                         batch_size=30, verbose=2)
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()
    
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

#prdiction rate
scores = model.evaluate(x=test_Feature, y=test_label)
print('Test loss:', scores[0])
print('accuracy',scores[1])


#save model
model.save('tainic_prediction_use_MLP.h5')

#load model
model = tf.contrib.keras.models.load_model('tainic_prediction_use_MLP.h5')