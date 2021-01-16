#kütüphaneler
print("kütüphaneler yükleniyor")
import pickle
from sklearn.svm import SVC
from keras.datasets import mnist

print("veri seti yükleniyor")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("veri ön işleme yapılıyor")
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

print("normalizasyon yapılıyor")
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print("model oluşturuluyor")
modelSVM = SVC(kernel='poly')

print("model eğitiliyor")
modelSVM.fit(x_train, y_train)

print("test işlemi yapılıp, tahminler kaydediliyor")
y_pred = modelSVM.predict(x_test)

#model kaydediliyor
model = 'modelSVM.pkl'
pickle.dump(modelSVM, open(model, 'wb'))

#model yükleniyor
model = pickle.load(open('modelSVM.pkl', 'rb'))
