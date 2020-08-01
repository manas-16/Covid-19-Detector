import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2,l1
from keras.initializers import Ones
import random
import os
from sklearn.metrics import confusion_matrix
import seaborn


class CovidDetector:

    def __init__(self):

        # Training model
        self.model = Sequential()

        #first convolutional layer

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3),kernel_initializer='he_uniform'))#,bias_regularizer=l2(0.25)))

        #First hidden layer set

        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.1))


        #Second hidden layer set

        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.1))


        #Output layer set

        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu',))
        self.model.add(Dropout(0.05))
        self.model.add(Dense(1, activation='sigmoid'))


        #joining model together

        self.model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])
        self.preprocess()




    def preprocess(self):

        # Moulding train images

        self.train_datagen = image.ImageDataGenerator(rotation_range=45,width_shift_range=0.3,
        height_shift_range=0.3,rescale=1. / 255, shear_range=0.3, zoom_range=0.3,
                                                 horizontal_flip=True,brightness_range=[0.1,1.0])


        self.test_dataset = image.ImageDataGenerator(rescale=1. / 255, shear_range=0.25, zoom_range=0.25,brightness_range=[0.1,1.0])

        # Reshaping test and train images

        print('Training set:-')
        self.train_generator = self.train_datagen.flow_from_directory(
            'CovidDataset/Train',
            target_size=(128, 128),
            batch_size=8,
            class_mode='binary')


        print('Test set:-')
        self.test_generator = self.test_dataset.flow_from_directory(
            'CovidDataset/Test',
            target_size=(128, 128),
            batch_size=8,
            class_mode='binary')

    def load(self):
        self.model.load_weights('Saved model\CoronaDetect 65.769% .h5')


    def train(self):

        #preprocess

        self.preprocess()

        # Training the model

        self.hist_new = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=256,
            epochs=8,
            validation_data=self.test_generator,
            validation_steps=64
        )

        # Getting summary

        summary = self.hist_new.history
        print(summary)
        x = str(round(self.model.evaluate_generator(self.test_generator)[1]*100,3))+'%'
        self.model.save("Saved model/CoronaDetect "+x+" .h5")


    def visualise(self):

         #print('accuracy of machine is '+str(round(self.model.evaluate_generator(self.test_generator)[1]*100,3))+'%')

        #visualise
        self.pred_list_Normal = []
        self.pred_list_Covid = []

        for i in range(1011,1271):
            file = './CovidDataset/Test/Normal/Normal image_'+str(i)+'.jpg'
            img = image.load_img(file, target_size=(128, 128))
            img = image.img_to_array(img.convert('RGB'))
            img = np.expand_dims(img, axis=0)
            pred = self.model.predict_classes(img)[0][0]
            self.pred_list_Normal.append(pred)

        for i in range(1011,1271):
            file = './CovidDataset/Test/Corona/Corona image_' + str(i) + '.jpg'
            img = image.load_img(file, target_size=(128, 128))
            img = image.img_to_array(img.convert('RGB'))
            img = np.expand_dims(img, axis=0)
            pred = self.model.predict_classes(img)[0][0]
            self.pred_list_Covid.append(pred)


        #visualise
        from matplotlib.patches import Rectangle
        #prediction histogram Test set
        # create legend
        labels = ["Normal","Corona"]
        handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['#F4B41A','#143D59']]
        plt.legend(handles, labels)
        #histogram
        ticks = [0.17,0.835]
        label1 = ['Corona','Normal']
        plt.hist([self.pred_list_Covid,self.pred_list_Normal],bins=3,color=['#143D59','#F4B41A'])
        plt.xticks(ticks,label1)
        plt.title('CNN prediction on test images')
        plt.xlabel('Actual Class')
        plt.ylabel('Frequency')
        plt.show()

        #Confusion Matrix
        import seaborn as sns
        self.confusion_matrix =[]
        x = [self.pred_list_Normal.count(1),self.pred_list_Normal.count(0)]
        y = [self.pred_list_Covid.count(1),self.pred_list_Covid.count(0)]
        # create legend
        labels = ["Wrong Prediction","Correct Prediction"]
        handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['#373543','#3CC194']]
        plt.legend(handles, labels)
        #heatmap
        ax = plt.axes()
        ax.set_title('Confusion Matrix')
        self.confusion_matrix.append(x)
        self.confusion_matrix.append(y)
        sns.set(font_scale=1.8)
        sns.heatmap(self.confusion_matrix,xticklabels=['Normal','Corona'] ,yticklabels=['Normal','Corona'],cmap=['#373543','#3CC194'],ax=ax,annot=True,fmt='d', cbar=False)
        ax.set_title('Confusion Matrix')
        plt.xlabel('Actual Class')
        plt.ylabel('Predicted Class')
        plt.show()
        print('Accuracy over Normal images = '+ str(round(x[0]/260,2)*100)+'%')
        print('Accuracy over Covid images = '+ str(round(y[1]/260,2)*100)+'%')



    def detect(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        img = image.load_img(file_path, target_size=(128, 128))
        img = image.img_to_array(img.convert('RGB'))
        img = np.expand_dims(img,axis=0)
        pred = self.model.predict_classes(img)
        if pred==0:
            print('Pneumonia caused by virus detected.\nYou may be affected by Covid 19. \nIt is suggested that you consult a doctor.')
        else:
            print('Normal X ray.\nNothing specific was detected.')
        print('This result was predicted by this machine which has an accuracy of '+str(round(self.model.evaluate_generator(self.test_generator)[1]*100,3))+'%')



if __name__ == '__main__':
    c = CovidDetector()
    c.load()
    c.visualise()


