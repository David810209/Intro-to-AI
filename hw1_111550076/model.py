import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class CarClassifier:
    def __init__(self, model_name, train_data, test_data):

        '''
        Convert the 'train_data' and 'test_data' into the format
        that can be used by scikit-learn models, and assign training images
        to self.x_train, training labels to self.y_train, testing images
        to self.x_test, and testing labels to self.y_test.These four 
        attributes will be used in 'train' method and 'eval' method.
        '''

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None

        # Begin your code (Part 2-1)
        self.x_train = [x[0] for x in train_data]
        self.y_train = [x[1] for x in train_data]
        self.x_test = [x[0] for x in test_data]
        self.y_test = [x[1] for x in test_data]
        # End your code (Part 2-1)
        
        self.model = self.build_model(model_name)
        
    
    def build_model(self, model_name):
        '''
        According to the 'model_name', you have to build and return the
        correct model.
        '''
        # Begin your code (Part 2-2)
        if model_name == 'KNN':
            return KNeighborsClassifier(n_neighbors=5)
        elif model_name == 'RF':
            return RandomForestClassifier(n_estimators=100)
        elif model_name == 'AB':
            return AdaBoostClassifier(n_estimators= 50)
        # End your code (Part 2-2)

    def train(self):
        '''
        Fit the model on training data (self.x_train and self.y_train).
        '''
        # Begin your code (Part 2-3)
        x_train_flattened = np.array([img.flatten() for img in self.x_train])
        self.x_test= np.array([img.flatten() for img in self.x_test])
        self.model.fit(x_train_flattened,self.y_train)
        # End your code (Part 2-3)
    
    def eval(self):
        y_pred = self.model.predict(self.x_test)
        print(f"Accuracy: {round(accuracy_score(y_pred, self.y_test), 4)}")
        print("Confusion Matrix: ")
        print(confusion_matrix(y_pred, self.y_test))
    
    def classify(self, input):
        return self.model.predict(input)[0]