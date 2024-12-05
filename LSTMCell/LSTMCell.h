#ifndef LSTM_CELL_H
#define LSTM_CELL_H

#include <vector>
using namespace std;

class LSTMCell {
public:
    int input_size;
    int hidden_size;

    vector<double> Wf, Wi, Wo, Wc;
    vector<double> bf, bi, bo, bc;

    vector<double> h_prev, c_prev;

    // Constructor
    LSTMCell(int input_size, int hidden_size);

    // Sigmoid activation function
    static double sigmoid(double x);

    // Tanh activation function
    static double tanh_activation(double x);

    // ReLU activation function
    static double relu(double x);

    void forward(const vector<double>& x);
};

#endif