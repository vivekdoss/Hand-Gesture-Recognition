import pandas as pd
import numpy as np
import os
import sys
import cv2


from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical



def create_csv():
    
    folder1 = 'F:/Hand_Gesture/testing/Image_color/'
    folder2 = 'F:/Hand_Gesture/testing/Image_gray/'
    folder3 = 'F:/Hand_Gesture/testing/Image_binary/'
    folder4 = 'F:/Hand_Gesture/testing/Image_without_bg/'
    
    lower_hand= np.array([0, 61, 0], dtype=np.uint8)
    upper_hand = np.array([53, 239, 255], dtype=np.uint8)

    
    number=['zero','one','two','three','four','five','six','seven','eight','nine','ten']
    
    index = 1
    label = 0
    
    column = ['Label']
    for i in range(28*28):
        column.append('pixel_'+str(i))
    
    data_set = pd.DataFrame(columns = column)
    
    for sign in number:
        
        img_cnt = 1
        
        for img_name in tqdm(os.listdir(folder1+sign)):
            
            #gray_img = cv2.imread(folder1+sign+'/'+img_name,0)
            img = cv2.imread(folder1+sign+'/'+img_name)
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            if hsv is not None:   
            
                resize_img = cv2.resize(hsv, (28,28))
                #ret,binary_img = cv2.threshold(resize_img,127,255,cv2.THRESH_BINARY)
                #cv2.imwrite(folder2+sign+'_gray/'+'sign_'+str(img_cnt)+'.jpg', resize_img)
                #cv2.imwrite(folder3+sign+'_binary/'+'sign_'+str(img_cnt)+'.jpg', binary_img)
                hand=cv2.inRange(resize_img,lower_hand,upper_hand)
                cv2.imwrite(folder4+sign+'/'+'sign_'+str(img_cnt)+'.jpg', hand)
            img_cnt +=1
            
            img_value = np.array(hand).reshape(1, 28*28)
            img_value = np.append(label,img_value)
            data_set.loc[index] = img_value
            index +=1
            
        label +=1            
   # data_set = data_set.drop['Unname: 0']    
    data_set.to_csv('F:/Hand_Gesture/Hand_gesture_testing_set_withoutbg.csv',index=False)
   


# https://medium.com/@pushkarmandot/build-your-first-deep-learning-neural-network-model-using-keras-in-python-a90b5864116d

def train():
    
    
    training = pd.read_csv('F:/Hand_Gesture/Hand_gesture_training_set_withoutbg.csv')
    
    
    #print(data.head())
    
    train_x = training.iloc[:,1:] 
    train_y = training.iloc[:,0] 
    
    #test_x = testing.iloc[:,1:]
   
    train_x = train_x/255.
    #test_x = test_x/255.
    
    train_x = train_x.values.reshape(-1, 28,28, 1)
    #test_x = test_x.values.reshape(-1, 28,28, 1)
    
    
    train_y_cate = to_categorical(train_y)
    #print(train_x)
    
    # Splitting the dataset into the Training set and Test set
      
    x_train, x_val, y_train, y_val = train_test_split(train_x, train_y_cate, test_size = 0.2)

    print(x_train.shape)
    print(x_val.shape)
    print(y_train.shape)
    print(y_val.shape)

    batch_size = 64
    epochs = 20
    num_classes = 11
     

    #Initializing Neural Network
    classifier = Sequential()
    
    classifier.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28,28,1), padding='same'))
    classifier.add(LeakyReLU(alpha=0.1))
    classifier.add(MaxPooling2D((2, 2),padding='same'))
    classifier.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    classifier.add(LeakyReLU(alpha=0.1))
    classifier.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    classifier.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    classifier.add(LeakyReLU(alpha=0.1))                  
    classifier.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    classifier.add(Flatten())
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(128, activation='linear'))
    classifier.add(LeakyReLU(alpha=0.1)) 
    classifier.add(Dense(128, activation='linear'))
    classifier.add(LeakyReLU(alpha=0.1)) 
    classifier.add(Dense(128, activation='linear'))
    classifier.add(LeakyReLU(alpha=0.1)) 
    classifier.add(Dense(128, activation='linear'))
    classifier.add(LeakyReLU(alpha=0.1)) 
    classifier.add(Dense(128, activation='linear'))
    classifier.add(LeakyReLU(alpha=0.1)) 
    classifier.add(Dense(128, activation='linear'))
    classifier.add(LeakyReLU(alpha=0.1))                  
    classifier.add(Dense(num_classes, activation='softmax'))
    
    # Compiling Neural Network
    classifier.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    #classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #print(classifier.summary())
    """
    datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
    
    datagen.fit(x_train)
    
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)
    """
    # Fitting our model 
    classifier.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
    #classifier.fit(X_train, y_train, batch_size = 128, epochs = 10)
    
    test_eval = classifier.evaluate(x_val, y_val, verbose=0)
    
    print('Trainnig data  loss:', test_eval[0])
    print('Trainig data accuracy:', test_eval[1]*100)

    classifier.save_weights("F:/Hand_Gesture/classifier_weights_without_bg.h5")
    print("Saved model to disk")


def test():
    
    column = ['Label']
    for i in range(28*28):
        column.append('pixel_'+str(i))
    data_set = pd.DataFrame(columns = column)
    
    #batch_size = 64
    #epochs = 5
    num_classes = 11

    #Initializing Neural Network
    classifier = Sequential()
    
    classifier.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(28,28,1), padding='same'))
    classifier.add(LeakyReLU(alpha=0.1))
    classifier.add(MaxPooling2D((2, 2),padding='same'))
    classifier.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    classifier.add(LeakyReLU(alpha=0.1))
    classifier.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    classifier.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    classifier.add(LeakyReLU(alpha=0.1))                  
    classifier.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    classifier.add(Flatten())
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(128, activation='linear'))
    classifier.add(LeakyReLU(alpha=0.1)) 
    classifier.add(Dense(128, activation='linear'))
    classifier.add(LeakyReLU(alpha=0.1)) 
    classifier.add(Dense(128, activation='linear'))
    classifier.add(LeakyReLU(alpha=0.1)) 
    classifier.add(Dense(128, activation='linear'))
    classifier.add(LeakyReLU(alpha=0.1)) 
    classifier.add(Dense(128, activation='linear'))
    classifier.add(LeakyReLU(alpha=0.1)) 
    classifier.add(Dense(128, activation='linear'))
    classifier.add(LeakyReLU(alpha=0.1))                  
    classifier.add(Dense(num_classes, activation='softmax'))
    
    classifier.load_weights('F:/Hand_Gesture/classifier_weights_without_bg.h5')
    
    #classifier.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    
    #classifier.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))

    if str(input('If you want to test your image with bg (Y/N)'))=='y':
                   
        name = input('Enter your name:')
        index = 1
        label = 0
        cap=cv2.VideoCapture(0)
      
        #f=0
        while True:
            
            #test_number=int(input('Enter the number if you want to test:'))
            
            ret,frame = cap.read()
            frame=cv2.flip(frame,1) 
            roi=frame[100:400,150:450]
            cv2.rectangle(frame,(145,95),(455,405),(0,0,255),3)
            print("Place your hand gesture into rectangle box correctly")
            resize_img = cv2.resize(roi, (28,28))
            gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
            ret,binary_img = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
            cv2.imshow("Hand",frame)
            
            cv2.waitKey(500)
            if str(input('Shall I take your gesture? (y/n): ')) == 'y':
                print('Have taken your image!!!')
                cv2.waitKey(1000)
                
                #cv2.imwrite("F:/Hand_Gesture/new_user/"+name+"_"+str(label)+".jpg",roi)
                #cv2.imwrite("F:/Hand_Gesture/new_user/"+name+"_"+str(label)+".jpg",gray_img)
                #cv2.imwrite("F:/Hand_Gesture/new_user/"+name+"_"+str(label)+".jpg",binary_img)
               
                img_value = np.array(binary_img).reshape(1, 28*28)
                img_value = np.append(label,img_value)
                data_set.loc[index] = img_value
                
                if str(input('If you want to continue ??? (y/n:)')) == 'y':
                    index += 1
                    label += 1
                    #f=1
                else:
                    break
    
        data_set.to_csv('F:/Hand_Gesture/new_user/'+name+'_withbg'+'.csv',index=False)
        cap.release()
        cv2.destroyAllWindows()
        testing  = pd.read_csv('F:/Hand_Gesture/new_user/'+name+'_withbg'+'.csv')
    
    elif str(input('If you want to test your image without bg (Y/N)'))=='y':

        lower_hand= np.array([0, 61, 0], dtype=np.uint8)
        upper_hand = np.array([53, 239, 255], dtype=np.uint8)
        index = 1
        label = 0
        
        name = input('Enter your name:')
        
        cap=cv2.VideoCapture(0)
      
        #f=0
        while True:
            
            #test_number=int(input('Enter the number if you want to test:'))
            
            ret,frame = cap.read()
            frame=cv2.flip(frame,1) 
            roi=frame[100:400,150:450]
            cv2.rectangle(frame,(145,95),(455,405),(0,0,255),3)
            print("Place your hand gesture into rectangle box correctly")
            resize_img = cv2.resize(roi, (28,28))
            hsv = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV)
            hand=cv2.inRange(hsv,lower_hand,upper_hand)
            #gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
            #ret,binary_img = cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY)
            cv2.imshow("Hand",frame)
 
            cv2.waitKey(500)
            if str(input('Shall I take your gesture? (y/n): ')) == 'y':
                print('Have taken your image!!!')
                cv2.waitKey(1000)
                
                #cv2.imwrite("F:/Hand_Gesture/new_user/"+name+"_"+str(label)+".jpg",roi)
                #cv2.imwrite("F:/Hand_Gesture/new_user/"+name+"_"+str(label)+".jpg",gray_img)
                #cv2.imwrite("F:/Hand_Gesture/new_user/"+name+"_"+str(label)+".jpg",binary_img)
               
                img_value = np.array(hand).reshape(1, 28*28)
                img_value = np.append(label,img_value)
                data_set.loc[index] = img_value
                
                if str(input('If you want to continue ??? (y/n:)')) == 'y':
                    index += 1
                    label += 1
                    #f=1
                else:
                    break
    
        data_set.to_csv('F:/Hand_Gesture/new_user/'+name+'_withoutbg'+'.csv',index=False)
        cap.release()
        cv2.destroyAllWindows()
        testing  = pd.read_csv('F:/Hand_Gesture/new_user/'+name+'_withoutbg'+'.csv')

    elif str(input('If you want to test all the images from folder? (y/n):')) == 'y':
        
        # This is the testing CSV file for already build data
        #testing = pd.read_csv('F:/Hand_Gesture/Hand_gesture_testing_set_binary.csv')
        testing = pd.read_csv('F:/Hand_Gesture/Hand_gesture_testing_set_withoutbg.csv')
        
    test_x = testing.iloc[:,1:]
    test_x = test_x/255. 
    test_x = test_x.values.reshape(-1, 28,28, 1)
    
    #print("%s: %.2f%%" % (classifier.metrics_names[1], predict*100))
    predict = classifier.predict_classes(test_x, verbose = 0)
    
    print("Answer", predict)
    #print("True Value", y_val)
    
    if str(input('Are you sure!!! you want to close (y/n)')) == 'n':
        test()  
    else:
        sys.exit()
    
if __name__ == "__main__":
    #create_csv()
    #train()  
    test() 
    
    
"""
   elif str(input('If you want to test the images by perticular: (y/n)')) == 'y':
        
        index = 1
        
        #sign=['zero','one','two','three','four','five','six','seven','eight','nine','ten']
        folder = 'F:/Hand_Gesture/testing/Image_binary/'
        
        number = input("Enter the number that you want to test (by spell):")
        #no = int(number)
        
        for img_name in tqdm(os.listdir(folder+str(number)+'_binary')):
            bin_img = cv2.imread(folder+str(number)+'_binary'+'/'+img_name)
        
            if bin_img is not None:
                
                resize_img = cv2.resize(bin_img, (28,28))
            img_value = np.array(resize_img).reshape(1, 28*28)
            img_value = np.append(number,img_value)
            data_set.loc[index] = img_value
            index +=1
                
       
        testing = data_set.to_csv('F:/Hand_Gesture/new_user/'+number+'.csv',index=False)  
"""        