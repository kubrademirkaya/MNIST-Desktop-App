#kütüphaneler
print("kütüphaneler yükleniyor")
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import pickle

#veri setinin yüklenmesi
print("veri seti yükleniyor")
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#veri önişleme
print("veri ön işleme yapılıyor")
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

#normalizasyon işlemi
print("normalizasyon yapılıyor")
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#model oluşturuluyor
print("model oluşturuluyor")
modelNB = MultinomialNB()

#model eğitiliyor
print("model eğitiliyor")
modelNB.fit(x_train, y_train)

#test işlemi yapılıyor
print("test işlemi yapılıp, tahminler kaydediliyor")
y_pred = modelNB.predict(x_test)

#model kaydediliyor
model = 'modelNB.pkl'
pickle.dump(modelNB, open(model, 'wb'))

#model yükleniyor
model = pickle.load(open('modelNB.pkl', 'rb'))
y_pred = model.predict(x_test)

