#from __future__ import print_function
# kütüphaneler
print("kütüphaneler yükleniyor")
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.image as mpimg


# her bir döngüde "128" resim alınıyor
batch_size = 128
# ayırt etmek istediğimiz "10" rakam (0-9)
num_classes = 10 
# eğitim 12 döngü kadar sürüyor
epochs = 30

# veri seti yükleniyor
print("veri seti yükleniyor")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# veri önişleme
print("veri önişleme yapılıyor")
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# veriler hakkında bilgilendirme 
print('\neğitim verileri giriş şekli:', x_train.shape)
print(x_train.shape[0], 'eğitim verileri')
print(x_test.shape[0], 'test verileri')

# sınıf vektörleri binary formununa dönüştürülüyor
# "to_catogorical" fonksiyonu ile one-hot-encoding yapılıyor
print("\nverilere OHE uygulanıyor")
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print("YSA-MLP yapısı oluşturuluyor")
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax')) 

#oluşturulan sinir ağının özet tablosu
print("oluşturulan YSA-MLP modeli ayrıntıları")
model.summary()


print("oluşturulan YSA-MLP modeli derleniyor")
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


print("model eğitiliyor")
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))


# tahminleri y_pred değişkenine kaydededip one-hot-encoding yapılıyor
print("test işlemi yapılıp tahminler kaydediliyor")
y_pred=model.predict(x_test)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save("modelMLP.h5")
model1 = load_model("modelMLP.h5")


