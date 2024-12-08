#include "LSTMCell/LSTMCell.h"
#include <vector>
#include <iostream>

int main() {
    int input_size = 3;  // Number of input features
    int hidden_size = 10; // Number of hidden units

    LSTM lstm(input_size, hidden_size);

    // Training data (replace with your own data)
    std::vector<std::vector<double>> train_inputs = {{1.0, 0.5, -1.0}, {0.5, -0.5, 0.0}};
    std::vector<std::vector<double>> train_targets = {{0.0, 1.0, 0.5, -0.5}, {0.5, 0.0, -0.5, -1.0}};

    int epochs = 100;
    double learning_rate = 0.1;

    train(lstm, train_inputs, train_targets, epochs, learning_rate);

    return 0;
}
