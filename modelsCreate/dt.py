#kütüphaneler
print("kütüphaneler yükleniyor")
from keras.datasets import mnist
from sklearn.tree import DecisionTreeClassifier
import pickle

print("veri seti yükleniyor")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#veri ön işleme
print("veri ön işleme yapılıyor")
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

#model oluşturuluyor
print("model oluşturuluyor")
modelDT = DecisionTreeClassifier(criterion='gini', #default
                                 min_impurity_decrease=0.0, #default
                                 min_samples_leaf=1, #default
                                 min_samples_split=2, #default
                                 min_weight_fraction_leaf=0.0, #default
                                 splitter='random');
                            
print("model eğitiliyor")
modelDT.fit(x_train,y_train)

print("model test edilip tahminler kaydediliyor")
y_pred = modelDT.predict(x_test)


model = 'modelDT.pkl'
pickle.dump(modelDT, open(model, 'wb'))


model = pickle.load(open('modelDT.pkl', 'rb'))
