#include "LSTMCell.h"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

LSTMCell::LSTMCell(int input_size, int hidden_size)
    : input_size(input_size), hidden_size(hidden_size) {

    srand(time(0));

    Wf.resize(input_size + hidden_size);
    Wi.resize(input_size + hidden_size);
    Wo.resize(input_size + hidden_size);
    Wc.resize(input_size + hidden_size);

    for (int i = 0; i < Wf.size(); ++i) {
        Wf[i] = ((rand() % 2000 - 1000) / 1000.0) * 0.5;
        Wi[i] = ((rand() % 2000 - 1000) / 1000.0) * 0.5;
        Wo[i] = ((rand() % 2000 - 1000) / 1000.0) * 0.5;
        Wc[i] = ((rand() % 2000 - 1000) / 1000.0) * 0.5;
    }

    bf.resize(hidden_size, 0.1);
    bi.resize(hidden_size, 0.1);
    bo.resize(hidden_size, 0.1);
    bc.resize(hidden_size, 0.1);

    h_prev.resize(hidden_size, 0.0);
    c_prev.resize(hidden_size, 0.0);
}

// Sigmoid activation function
double LSTMCell::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Tanh activation function
double LSTMCell::tanh_activation(double x) {
    return tanh(x);
}

// ReLU activation function
double LSTMCell::relu(double x) {
    return (x > 0) ? x : 0;
}

void LSTMCell::forward(const vector<double>& x) {
    vector<double> combined_input(input_size + hidden_size);

    for (int i = 0; i < input_size; ++i) combined_input[i] = x[i];
    for (int i = 0; i < hidden_size; ++i) combined_input[input_size + i] = h_prev[i];

    vector<double> f(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        f[i] = sigmoid(Wf[i] * combined_input[i] + bf[i]);
    }

    vector<double> i_gate(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        i_gate[i] = sigmoid(Wi[i] * combined_input[i] + bi[i]);
    }

    vector<double> o_gate(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        o_gate[i] = sigmoid(Wo[i] * combined_input[i] + bo[i]);
    }

    vector<double> c_hat(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        c_hat[i] = relu(Wc[i] * combined_input[i] + bc[i]);
    }
}