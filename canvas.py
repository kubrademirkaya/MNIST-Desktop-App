from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Canvas(QMainWindow):
    
    def __init__(self):
        
        super().__init__()
        
        self.width = 400
        self.height = 400
        self.setWindowTitle("Çizim Alanı")
        self.setGeometry(50,100,self.width, self.height)
        
        self.image = QImage(self.size(),QImage.Format_RGB32)
        self.image.fill(Qt.black)
        
        self.lastPoint = QPoint()
        self.drawing = False
        
        # image array
        self.im_np = np.zeros([self.width, self.height])
        
        buttonSaveImage = QPushButton("Kaydet", self)
        buttonSaveImage.move(75,350)
        buttonSaveImage.clicked.connect(self.enterFunction)
        
        buttonCleanCanvas = QPushButton("Temizle", self)
        buttonCleanCanvas.move(225,350)
        buttonCleanCanvas.clicked.connect(self.cleanCanvasFunction)
        
        #self.show()
        
    def paintEvent(self,event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(),self.image, self.image.rect())
        
    def enterFunction(self):
        ptr = self.image.constBits()
        ptr.setsize(self.image.byteCount())
        
        self.im_np = np.array(ptr).reshape(self.width, self.height,4)
        self.im_np = self.im_np[:,:,0]
        self.im_np = self.im_np/255.0
        
        if np.sum(self.im_np) ==0:
            QMessageBox.information(self,"Hata","Lütfen bir rakam yazınız.")
        else:
            plt.figure(figsize = (1,1),dpi = 200)
            plt.imshow(self.im_np, cmap = "gray")
            plt.axis("off")
            plt.grid(False)
            plt.savefig("inputimages\\input_img.png")
            
            self.close()
            
    def cleanCanvasFunction(self):
        self.image = QImage(self.size(),QImage.Format_RGB32)
        self.image.fill(Qt.black)
            
        self.update()
            
    def mousePressEvent(self,event):
            
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.drawing = True
            print(self.lastPoint)
                
    def mouseMoveEvent(self, event):
        if (event.buttons() == Qt.LeftButton) & self.drawing:
           painter = QPainter(self.image)
           painter.setPen(QPen(Qt.white, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
           painter.drawLine(self.lastPoint, event.pos())
           self.lastPoint = event.pos()
           self.update()
           
    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False
           
