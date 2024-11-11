#include <iostream>
#include "./include/lstm_layer.h"
#include "./include/activation_functions.h"
#include <fstream>

using namespace Eigen;
using namespace std;

MatrixXd loadCSV(const string &path, int rows, int cols) {
    ifstream file(path); string line;
    MatrixXd data(rows, cols);
    int row = 0, col = 0;
    while (getline(file, line)) {
        stringstream lineStream(line);
        string cell;
        while (getline(lineStream, cell, ',')) {
            data(row, col++) = stod(cell);
        } row++; col = 0;
    } return data;
}
int main() {
    MatrixXd x_train = loadCSV("data/sequence_train.csv", 1000, 10);
    MatrixXd y_train = loadCSV("data/sequence_train_labels.csv", 1000, 1);

    LSTM lstm_layer(10, 20);
    lstm_layer.forward(x_train);
    cout << "LSTM katmanı başarıyla çalıştırıldı!" << endl;

    return 0;
}
