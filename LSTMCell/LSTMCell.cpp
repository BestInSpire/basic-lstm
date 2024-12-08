// File: lstm.cpp
#include "LSTMCell.h"

LSTM::LSTM(int input_size, int hidden_size) : input_size(input_size), hidden_size(hidden_size) {
    initialize_weights();
}

void LSTM::initialize_weights() {
    Wf = random_matrix(hidden_size, input_size + hidden_size);
    Wi = random_matrix(hidden_size, input_size + hidden_size);
    Wo = random_matrix(hidden_size, input_size + hidden_size);
    Wc = random_matrix(hidden_size, input_size + hidden_size);

    bf = random_vector(hidden_size);
    bi = random_vector(hidden_size);
    bo = random_vector(hidden_size);
    bc = random_vector(hidden_size);

    h = std::vector<double>(hidden_size, 0.0);
    c = std::vector<double>(hidden_size, 0.0);
}

__gnu_cxx::__alloc_traits<std::allocator<double>>::value_type operator*(__gnu_cxx::__alloc_traits<std::allocator<double>>::value_type lhs, const std::vector<double> & rhs);

void LSTM::forward(const std::vector<double>& input) {
    std::vector<double> combined(input_size + hidden_size);
    for (int i = 0; i < input_size; i++) combined[i] = input[i];
    for (int i = 0; i < hidden_size; i++) combined[input_size + i] = h[i];

    std::vector<double> forget_gate = sigmoid(mat_mul(Wf, combined, bf));
    std::vector<double> input_gate = sigmoid(mat_mul(Wi, combined, bi));
    std::vector<double> output_gate = sigmoid(mat_mul(Wo, combined, bo));
    std::vector<double> candidate = tanh(mat_mul(Wc, combined, bc));

    for (int i = 0; i < hidden_size; i++) {
        c[i] = forget_gate[i] * c[i] + input_gate[i] * candidate[i];
    }

    for (int i = 0; i < hidden_size; i++) {
        h[i] = output_gate[i] * std::tanh(c[i]);
    }
}

std::vector<double> LSTM::get_hidden_state() const {
    return h;
}

double LSTM::calculate_loss(const std::vector<double>& predicted, const std::vector<double>& actual) const {
    double loss = 0.0;
    for (size_t i = 0; i < predicted.size(); i++) {
        double diff = predicted[i] - actual[i];
        loss += diff * diff;
    }
    return loss / predicted.size();
}

void LSTM::update_weights(std::vector<std::vector<double>>& weights, const std::vector<std::vector<double>>& gradients, double learning_rate) {
    for (size_t i = 0; i < weights.size(); i++) {
        for (size_t j = 0; j < weights[0].size(); j++) {
            weights[i][j] -= learning_rate * gradients[i][j];
        }
    }
}

std::vector<double> LSTM::mat_mul(const std::vector<std::vector<double>>& mat,
                                  const std::vector<double>& vec,
                                  const std::vector<double>& bias) {
    std::vector<double> result(mat.size());
    for (size_t i = 0; i < mat.size(); ++i) {
        result[i] = bias[i];
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

std::vector<double> LSTM::sigmoid(const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = 1.0 / (1.0 + exp(-vec[i]));
    }
    return result;
}

std::vector<double> LSTM::tanh(const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = std::tanh(vec[i]);
    }
    return result;
}

std::vector<std::vector<double>> LSTM::random_matrix(int rows, int cols) {
    std::vector<std::vector<double>> mat(rows, std::vector<double>(cols));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = dis(gen);
        }
    }
    return mat;
}

std::vector<double> LSTM::random_vector(int size) {
    std::vector<double> vec(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);

    for (int i = 0; i < size; ++i) {
        vec[i] = dis(gen);
    }
    return vec;
}

void train(LSTM& lstm, const std::vector<std::vector<double>>& train_inputs, 
           const std::vector<std::vector<double>>& train_targets, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (size_t i = 0; i < train_inputs.size(); i++) {
            lstm.forward(train_inputs[i]);
            std::vector<double> predicted = lstm.get_hidden_state();

            double loss = lstm.calculate_loss(predicted, train_targets[i]);
            total_loss += loss;

            // lstm.update_weights(...);
        }

        std::cout << "Epoch " << epoch + 1 << " - Loss: " << total_loss / train_inputs.size() << std::endl;
    }
}
