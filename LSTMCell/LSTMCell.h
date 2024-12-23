#ifndef LSTM_H
#define LSTM_H

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

// Struct for LSTM weights and biases
struct LSTMWeights {
    double** Wf;
    double** Wi;
    double** Wo;
    double** Wc;
    double* bf;
    double* bi;
    double* bo;
    double* bc;
};

// Struct for LSTM states
struct LSTMStates {
    double* h; // Hidden state
    double* c; // Cell state
};

// LSTM Class
class LSTM {
public:
    LSTM(int input_size, int hidden_size);
    ~LSTM();

    void forward(double* input);
    double* get_hidden_state() const;

private:
    int input_size;
    int hidden_size;

    LSTMWeights* weights;
    LSTMStates* states;

    void initialize_weights();
    void free_weights();

    static double* mat_mul(double** mat, double* vec, double* bias, int rows, int cols);
    static double sigmoid(double x);
    static double tanh(double x);
    static double* vector_tanh(double* vec, int size);
    static double* vector_sigmoid(double* vec, int size);
};

#endif
