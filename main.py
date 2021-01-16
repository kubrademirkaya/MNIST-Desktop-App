from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint

import numpy as np
import matplotlib.image as mpimg

import cv2
import pickle

import canvas
        
class Window(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        #Ana pencere
        self.width = 1080
        self.height = 640
        
        self.setWindowTitle("El Yazısı Tanıma Uygulaması")        
        self.setGeometry(50, 100, self.width, self.height)
        
        self.create_canvas = canvas.Canvas()
        
        self.tabWidget()
        self.widgets()
        self.layouts()
        self.show()

    def tabWidget(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        
        self.tabs.addTab(self.tab1, "Uygulama")
        self.tabs.addTab(self.tab2, "Modeller")

    def widgets(self):
        
        #tab1 Sol Bölüm
        self.drawCanvas = QPushButton("Çizim Yap")
        self.drawCanvas.clicked.connect(self.drawCanvasFunction)
        
        self.openCanvas = QPushButton("Çizimi Görüntüle")
        self.openCanvas.clicked.connect(self.openCanvasFunction)
        
        self.inputImage = QLabel(self)
        self.inputImage.setPixmap(QPixmap("inputimages\\input.png"))
        self.inputImage.setAlignment(Qt.AlignCenter)
        
        self.searchText = QLabel("Yazılan Sayı: ")
        
        self.searchEntry = QLineEdit()
        
        self.methodSelection = QComboBox(self)
        self.methodSelection.addItems(["Convolutional Neural Network 1", 
                                       "Convolutional Neural Network 2", 
                                       "Convolutional Neural Network 3",
                                       "Multiple-layer Perceptron Neural Network",
                                       "Naive Bayes",
                                       "Support Vector Machine",
                                       "K-Nearest Neighbors",
                                       "Decision Tree"])
        
        self.predict = QPushButton("Tahmin Yap")
        self.predict.clicked.connect(self.predictionFunction)
        
        
        #tab1 Sağ Bölüm
        
        self.outputImage = QLabel(self)
        self.outputImage.setPixmap(QPixmap("inputimages\\input.png",))
        self.outputImage.setAlignment(Qt.AlignCenter)
        
        self.outputLabel = QLabel("", self)
        self.outputLabel.setAlignment(Qt.AlignCenter)
        
        self.resultTable = QTableWidget()
        self.resultTable.setColumnCount(2)
        self.resultTable.setRowCount(10)
        self.resultTable.horizontalHeader().setStretchLastSection(True)
        self.resultTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.resultTable.setHorizontalHeaderItem(0, QTableWidgetItem("Sınıf"))
        self.resultTable.setHorizontalHeaderItem(1, QTableWidgetItem("Tahmin Değeri"))
        
        
        #tab2 Sol Bölüm
        #CNN yapıları
        
        self.parameter_list1 = QListWidget(self)
        self.parameter_list1.addItems(["Convolutional Neural Network 1",
                                       "batch_size = 128" " epochs = 30",
                                       "Filter = [16,32,64]",
                                       "Activation Function = Relu",
                                       "loss = categorical cross entropy",
                                       "optimizer = Adam",
                                       "metrics = accuracy"])
        
        
        self.parameter_list2  = QListWidget(self)
        self.parameter_list2.addItems(["Convolutional Neural Network 2",
                                       "batch_size = 128" " epochs = 30",
                                       "Filter = [32,64,128]",
                                       "Activation Function = Relu",
                                       "loss = categorical cross entropy",
                                       "optimizer = Adam",
                                       "metrics = accuracy"])
        
        self.parameter_list3  = QListWidget(self)
        self.parameter_list3.addItems(["Convolutional Neural Network 3",
                                       "batch_size = 128" " epochs = 30",
                                       "Filter = [64,128,256]",
                                       "Activation Function = Relu",
                                       "loss = categorical cross entropy",
                                       "optimizer = Adam",
                                       "metrics = accuracy"])
        
        self.parameter_list4  = QListWidget(self)
        self.parameter_list4.addItems([ "Multiple-layer Perceptron Neural Network",
                                       "batch_size = 128" " epochs = 30",
                                       "Activation Function = Relu",
                                       "loss = categorical cross entropy",
                                       "optimizer = RMSprop",
                                       "metrics = accuracy"])
        
        #tab2 sağ bölüm
        
        self.parameter_list5  = QListWidget(self)
        self.parameter_list5.addItems(["Naive Bayes",
                                       "model = MultinomialNB"])
        
        self.parameter_list6  = QListWidget(self)
        self.parameter_list6.addItems(["Support Vector Machine",
                                       "model = SupportVectorClassification",
                                       "kernel = poly"])
        
        self.parameter_list7  = QListWidget(self)
        self.parameter_list7.addItems(["K-Nearest Neighbors",
                                       "n_neighbors = 1", 
                                       "metric = minkowski"])
        
        self.parameter_list8  = QListWidget(self)
        self.parameter_list8.addItems(["Decision Tree",
                                       "criterion = gini, default",
                                       "min_impurity_decrease = 0.0, default",
                                       "min_samples_leaf = 1, default",
                                       "min_samples_split = 2, default",
                                       "min_weight_fraction_leaf = 0.0, default",
                                       "splitter = random"])
        
        
    def drawCanvasFunction(self):
        self.create_canvas.show()
        
    def predictionFunction(self):
        save_string = ""
        
        real_entry = self.searchEntry.text()
        save_string = save_string + "real entry: " + str(real_entry) + ", "
        
        # CNN model selection
        model_name = self.methodSelection.currentText()
        
        if model_name == "Convolutional Neural Network 1":
            model = load_model("models\\modelCNN1.h5")
        elif model_name == "Convolutional Neural Network 2":
            model = load_model("models\\modelCNN2.h5")
        elif model_name == "Convolutional Neural Network 3":
            model = load_model("models\\modelCNN3.h5")
        elif model_name == "Multiple-layer Perceptron Neural Network":
            model = load_model("models\\modelMLP.h5")
        elif model_name == "Naive Bayes":
            model = pickle.load(open("models\\modelNB.pkl", 'rb'))
        elif model_name == "Support Vector Machine":
            model = pickle.load(open("models\\modelSVM.pkl", 'rb'))
        elif model_name == "K-Nearest Neighbors":
            model = pickle.load(open("models\\modelKNN.pkl", 'rb'))
        elif model_name == "Decision Tree":
            model = pickle.load(open("models\\modelDT.pkl", 'rb'))
        else:
            print("Error")
            
            
            
        save_string = save_string + "model name: " + str(model_name) + ", "
        print(save_string)
        
        # load image as numpy
        img_array = mpimg.imread("inputimages\\input_img.png")[26:175,26:175,0]
        
        resized_img_array = cv2.resize(img_array, dsize=(28,28),interpolation = cv2.INTER_CUBIC)

        #plt.imshow(resized_img_array, cmap = "gray")    

        # predict
        if model_name == "Convolutional Neural Network 1":
            result = model.predict(resized_img_array.reshape(1,28,28,1))
            predicted_class = np.argmax(result)
        elif model_name == "Convolutional Neural Network 2":
            result = model.predict(resized_img_array.reshape(1,28,28,1))
            predicted_class = np.argmax(result)
        elif model_name == "Convolutional Neural Network 3":
            result = model.predict(resized_img_array.reshape(1,28,28,1))
            predicted_class = np.argmax(result)
        elif model_name == "Multiple-layer Perceptron Neural Network":
            result = model.predict(resized_img_array.reshape(1,784))
            predicted_class = np.argmax(result)
        elif model_name == "Naive Bayes":
            result = model.predict(resized_img_array.reshape(1,784))
            predicted_class = result
        elif model_name == "Support Vector Machine":
            result = model.predict(resized_img_array.reshape(1,784))
            predicted_class = result
        elif model_name == "K-Nearest Neighbors":
            result = model.predict(resized_img_array.reshape(1,784))
            predicted_class = result
        elif model_name == "Decision Tree":
            result = model.predict(resized_img_array.reshape(1,784))
            predicted_class = result
        else:
            print("Error")
            
        QMessageBox.information(self,"Bilgi","Sınıflandırma tamamlandı.")
        #predicted_class = np.argmax(result)
        print("Prediction: ",predicted_class)
        
        save_string = save_string + "Predicted class: "+str(predicted_class)
        
        self.outputImage.setPixmap(QPixmap("images\\"+str(predicted_class)+".png"))
        self.outputLabel.setText("Yazılan Sayı: "+str(real_entry)+ " \n Tahmin Edilen Sayı: "+str(predicted_class))
        
        # set result
        for row in range(10):
            self.resultTable.setItem(row,0,QTableWidgetItem(str(row)))
            self.resultTable.setItem(row,1,QTableWidgetItem(str(np.round(result[0][row],3))))
        
        
    def openCanvasFunction(self):
        self.inputImage.setPixmap(QPixmap("inputimages\\input_img.png"))
        
        

    def layouts(self):
        
        #tab1Layout
        self.mainLayout = QHBoxLayout()
        self.leftLayout = QFormLayout()
        self.rightLayout = QFormLayout()
        
        
        #leftlayout
        self.leftLayoutGroupBox = QGroupBox("Giriş")
        self.leftLayout.addRow(self.drawCanvas)
        self.leftLayout.addRow(self.inputImage)
        self.leftLayout.addRow(self.openCanvas)
        self.leftLayout.addRow(self.searchText)
        self.leftLayout.addRow(self.searchEntry)
        self.leftLayout.addRow(self.methodSelection)
        self.leftLayout.addRow(self.predict)
        self.leftLayoutGroupBox.setLayout(self.leftLayout)
        
        
        #RightLayout
        self.rightLayoutGroupBox = QGroupBox("Çıkış")
        self.rightLayout.addRow(self.outputImage)
        self.rightLayout.addRow(self.outputLabel)
        self.rightLayout.addRow(self.resultTable)
        self.rightLayoutGroupBox.setLayout(self.rightLayout)


        #MainLayout
        self.mainLayout.addWidget(self.leftLayoutGroupBox, 50)
        self.mainLayout.addWidget(self.rightLayoutGroupBox, 50)
        self.tab1.setLayout(self.mainLayout)


        
        #tab2Layout
        self.tab2Layout = QHBoxLayout()
        self.tab2MethodCNNLayout = QFormLayout()
        self.tab2Method2Layout = QFormLayout()
        
        
        #tab2MethodCNNLayout
        self.tab2MethodCNNLayoutGroupBox = QGroupBox("Sinir Ağları Yapıları")
        self.tab2MethodCNNLayout.addRow(self.parameter_list1)
        self.tab2MethodCNNLayout.addRow(self.parameter_list2)
        self.tab2MethodCNNLayout.addRow(self.parameter_list3)
        self.tab2MethodCNNLayout.addRow(self.parameter_list4)
        self.tab2MethodCNNLayoutGroupBox.setLayout(self.tab2MethodCNNLayout)
        
        
        self.tab2Method2LayoutGroupBox = QGroupBox("Sınıflandırma Algoritmaları")
        self.tab2Method2Layout.addRow(self.parameter_list5)
        self.tab2Method2Layout.addRow(self.parameter_list6)
        self.tab2Method2Layout.addRow(self.parameter_list7)
        self.tab2Method2Layout.addRow(self.parameter_list8)
        self.tab2Method2LayoutGroupBox.setLayout(self.tab2Method2Layout)


        #tab2 MainLayout
        self.tab2Layout.addWidget(self.tab2MethodCNNLayoutGroupBox, 50)
        self.tab2Layout.addWidget(self.tab2Method2LayoutGroupBox, 50)
        self.tab2.setLayout(self.tab2Layout)


 
window = Window()