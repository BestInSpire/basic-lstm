#ifndef LSTM_H
#define LSTM_H

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

class LSTM {
public:
    LSTM(int input_size, int hidden_size);

    void forward(const std::vector<double>& input);
    std::vector<double> get_hidden_state() const;

    // Training methods
    double calculate_loss(const std::vector<double>& predicted, const std::vector<double>& actual) const;
    void update_weights(std::vector<std::vector<double>>& weights, const std::vector<std::vector<double>>& gradients, double learning_rate);

private:
    int input_size;
    int hidden_size;

    // LSTM weights and biases
    std::vector<std::vector<double>> Wf, Wi, Wo, Wc;
    std::vector<double> bf, bi, bo, bc;

    // Hidden and cell states
    std::vector<double> h, c;

    void initialize_weights();

    // Helper functions
    static std::vector<double> mat_mul(const std::vector<std::vector<double>>& mat,
                                       const std::vector<double>& vec,
                                       const std::vector<double>& bias);
    static std::vector<double> sigmoid(const std::vector<double>& vec);
    static std::vector<double> tanh(const std::vector<double>& vec);
    static std::vector<std::vector<double>> random_matrix(int rows, int cols);
    static std::vector<double> random_vector(int size);
};

void train(LSTM& lstm, const std::vector<std::vector<double>>& train_inputs, 
           const std::vector<std::vector<double>>& train_targets, int epochs, double learning_rate);

#endif
