"""
@author: Viet
A pretty standard quick implementation of
Emb-> biLSTM -> NN
as suggested by: https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy
"""

from model.pre2 import retrieve_and_pre, dump_to_csv
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

def main():
    X, y, test, names = retrieve_and_pre(fromsave=True, tfidfpca=False) #False because we want to acquire actual text
    max_features = 50000
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X)
    list_tokenized_train = tokenizer.texts_to_sequences(X)
    maxlen = 200
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

    X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=0.2, shuffle=True)

    embed_size = 256
    model = Sequential()
    model.add(Embedding(max_features, embed_size))
    model.add(Bidirectional(LSTM(8, return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.20))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    batch_size = 100
    epochs = 3
    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=epochs)

    model.save('lstm.h5')

    list_tokenized_lol = tokenizer.texts_to_sequences(lol)
    X_lol = pad_sequences(list_tokenized_lol, maxlen=maxlen)
    prediction = model.predict(X_lol)
    y_pred = (prediction > 0.5)

    scr = accuracy_score(y, y_pred)
    f1_scr = f1_score(y, y_pred)

    print("accuracy: %f; f1-score: %f" %(scr, f1_scr))

    #model = load_model('lstm.h5')
    list_tokenized_test = tokenizer.texts_to_sequences(test)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
    prediction = model.predict(X_te)
    y_pred = (prediction > 0.5)
    np.save('prediction', y_pred)
    lol = []
    print(y_pred.shape)
    for i in range(len(y_pred)):
        if y_pred[i]:
            lol.append(1)
        else:
            lol.append(0)
    print(lol[:10])
    names = np.load('names.npy')
    print(names[2], lol[2])
    dump = []
    for i in range(len(lol)):
        dump.append([names[i], lol[i]])
    print(dump[0])
    pd.DataFrame(np.asarray(dump)).to_csv('emb-lstm2' + '.csv')

if __name__ == '__main__':
    main()
