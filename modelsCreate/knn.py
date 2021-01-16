#kütüphaneler
print("kütüphaneler yükleniyor")
from tensorflow.keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier
import pickle

print("veriler yükleniyor")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#veri önişleme
print("veri ön işleme yapılıyor")
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

#normalizasyon
print("normalizasyon işlemi yapılıyor")
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print("model oluşturuluyor")
modelKNN = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

print("model eğitiliyor")
modelKNN.fit(x_train,y_train)

print("tahmin işlemi yapılıp kaydediliyor")
y_pred = modelKNN.predict(x_test)

#model kaydediliyor
model = 'modelKNN.pkl'
pickle.dump(modelKNN, open(model, 'wb'))

#model yükleniyor
model = pickle.load(open('modelKNN.pkl', 'rb'))
