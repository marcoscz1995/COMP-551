"""
@author: Viet
Note that we commented out the autoencoder part because its performance was so bad that
we reconsidered our registration in this course

This is a pretty standard neural network implementation in pytorch, and so minimal commenting is used.
"""
import torch
from torch import nn, optim
from torch.autograd import Variable as var
from torch.utils.data import TensorDataset, DataLoader
from model.pre2 import retrieve_and_pre, dump_to_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
import pandas as pd


class FC(nn.Module):

    def forward(self, x):
        x = self.dropout(self.sigmoid(self.bn1(self.linear1(x))))
        x = self.dropout(self.relu(self.bn2(self.linear2(x))))
        return self.sigmoid(self.linear3(x))

    def __init__(self):
        super(FC, self).__init__()
        self.linear1 = nn.Linear(in_features=5000, out_features=750, bias=True)
        torch.nn.init.xavier_normal_(self.linear1.weight)
        self.bn1 = nn.BatchNorm1d(750)
        self.linear2 = nn.Linear(in_features=750, out_features=400, bias=True)
        self.bn2 = nn.BatchNorm1d(400)
        self.linear3 = nn.Linear(in_features=400, out_features=1, bias=True)


        self.dropout = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

def evaluate(model, test_loader, batch_size):
    correct = 0
    for test_imgs, test_labels in test_loader:
        test_imgs = var(test_imgs).float()
        var_y_batch = var(test_labels).float()
        output = model(test_imgs)
        pred = [1 if i > 0.5 else 0 for i in output]
        for i in range(len(pred)):
            if int(pred[i]) == int(var_y_batch[i]): correct += 1
    print("Test accuracy:{:.3f}% ".format(float(correct)*100 / 5000))

def evaluate_on_real(model, test_loader, batch_size):
    correct = 0
    for test_imgs, test_labels in test_loader:
        test_imgs = var(test_imgs).float()
        var_y_batch = var(test_labels).float()
        output = model(test_imgs)
        pred = [1 if i > 0.5 else 0 for i in output]
        for i in range(len(pred)):
            if int(pred[i]) == int(var_y_batch[i]): correct += 1
    print("Test accuracy:{:.3f}% ".format(float(correct)*100 / 10000))
    return pred


def fit(model, train_loader, test_loader, learning_rate, epochs):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for e in range(epochs):
        correct = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = var(X_batch).float()
            var_y_batch = var(y_batch).float()

            output = model(var_X_batch)
            loss = criterion(output, var_y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = [1 if i > 0.5 else 0 for i in output]
            for i in range(len(pred)):
                if int(pred[i]) == int(var_y_batch[i]): correct+=1
            if batch_idx % 20 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    e, batch_idx * len(X_batch), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.data[0], float(correct * 100) / float(var_y_batch.shape[0] * (batch_idx + 1))))
        print()
        evaluate(model, test_loader, 32)


def main():
    # def some stuff
    batch_size = 200
    learning_rate = 1e-3
    epochs = 10

    X, y, final, names= retrieve_and_pre(fromsave=True, tfidfpca=True, n_components=20000)
    #X = X.todense()
    #lol = lol.todense()
    #final = final.todense()

    y = np.asarray(y).reshape(25000, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=2)

    """


    encoding_dim = 1000
    ncol = X_train.shape[1]
    input_dim = Input(shape=(ncol,))
    encoded1 = Dense(2000, activation='relu')(input_dim)
    encoded13 = Dense(encoding_dim, activation='relu')(encoded1)

    # Decoder Layers
    decoded1 = Dense(2000, activation='relu')(encoded13)
    decoded13 = Dense(ncol, activation='sigmoid')(decoded1)

    # Combine Encoder and Deocder layers
    autoencoder = Model(inputs=input_dim, outputs=decoded13)

    # Compile the Model
    autoencoder.compile(optimizer='nadam', loss='binary_crossentropy')
    print(autoencoder.summary())

    autoencoder.fit(X_train, X_train, nb_epoch=1, batch_size=512, shuffle=False, validation_data=(X_test, X_test))
    save_model(autoencoder, 'autoencoder.h5', include_optimizer=True)
    encoder = Model(inputs=input_dim, outputs=encoded13)
    encoded_input = Input(shape=(encoding_dim,))

    encoded_train = encoder.predict(X_train)
    encoded_test = encoder.predict(X_test)
    encoded_final = encoder.predict(final)
    """

    #torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
    #torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # data type is long

    # create feature and targets tensor for test set.
    #torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
    #torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # data type is long

    train = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

    #Training happens here
    neural = FC()
    fit(neural, train_loader, test_loader, learning_rate, epochs)
    torch.save(neural.state_dict(), '2layfc_saved')

    #Make this kaggle submission
    torch_final = torch.from_numpy(final).type(torch.FloatTensor)
    pred = neural(torch_final)
    pred = [1 if i > 0.5 else 0 for i in pred]
    dump = []
    for i in range(len(pred)):
        dump.append([names[i], pred[i]])
    pd.DataFrame(np.asarray(dump)).to_csv('2layfc_pred' + '.csv')


def predict():
    X, y, final, names = retrieve_and_pre(fromsave=True, tfidfpca=True)
    neural = FC()
    neural.load_state_dict(torch.load('2layfc_saved'))
    neural.eval()
    final = np.asarray(final)
    torch_final = torch.from_numpy(final).type(torch.FloatTensor)
    pred = neural(torch_final)
    pred = [1 if i > 0.5 else 0 for i in pred]
    dump = []
    for i in range(len(pred)):
        dump.append([names[i], pred[i]])
    pd.DataFrame(np.asarray(dump)).to_csv('2layfc_pred' + '.csv')


if __name__ == '__main__':
    main()
    #predict()

