#include "lstm.h"


LSTM::LSTM(int input_size, int hidden_size) : input_size(input_size), hidden_size(hidden_size) {
    weights = new LSTMWeights;
    states = new LSTMStates;

    states->h = new double[hidden_size]{0};
    states->c = new double[hidden_size]{0};

    initialize_weights();
}

LSTM::~LSTM() {
    free_weights();
    delete[] states->h;
    delete[] states->c;
    delete states;
}

void LSTM::initialize_weights() {
    auto random_double = []() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(-0.1, 0.1);
        return dis(gen);
    };

    auto allocate_matrix = [random_double](int rows, int cols) {
        double** mat = new double*[rows];
        for (int i = 0; i < rows; ++i) {
            mat[i] = new double[cols];
            for (int j = 0; j < cols; ++j) {
                mat[i][j] = random_double();
            }
        }
        return mat;
    };

    auto allocate_vector = [random_double](int size) {
        double* vec = new double[size];
        for (int i = 0; i < size; ++i) {
            vec[i] = random_double();
        }
        return vec;
    };

    weights->Wf = allocate_matrix(hidden_size, input_size + hidden_size);
    weights->Wi = allocate_matrix(hidden_size, input_size + hidden_size);
    weights->Wo = allocate_matrix(hidden_size, input_size + hidden_size);
    weights->Wc = allocate_matrix(hidden_size, input_size + hidden_size);

    weights->bf = allocate_vector(hidden_size);
    weights->bi = allocate_vector(hidden_size);
    weights->bo = allocate_vector(hidden_size);
    weights->bc = allocate_vector(hidden_size);
}

void LSTM::free_weights() {
    auto free_matrix = [](double** mat, int rows) {
        for (int i = 0; i < rows; ++i) {
            delete[] mat[i];
        }
        delete[] mat;
    };

    free_matrix(weights->Wf, hidden_size);
    free_matrix(weights->Wi, hidden_size);
    free_matrix(weights->Wo, hidden_size);
    free_matrix(weights->Wc, hidden_size);

    delete[] weights->bf;
    delete[] weights->bi;
    delete[] weights->bo;
    delete[] weights->bc;

    delete weights;
}

void LSTM::forward(double* input) {
    // Combine input and hidden state
    double* combined = new double[input_size + hidden_size];
    for (int i = 0; i < input_size; ++i) combined[i] = input[i];
    for (int i = 0; i < hidden_size; ++i) combined[input_size + i] = states->h[i];

    // Compute gates
    double* forget_gate = vector_sigmoid(
        mat_mul(weights->Wf, combined, weights->bf, hidden_size, input_size + hidden_size),
        hidden_size
    );
    double* input_gate = vector_sigmoid(
        mat_mul(weights->Wi, combined, weights->bi, hidden_size, input_size + hidden_size),
        hidden_size
    );
    double* output_gate = vector_sigmoid(
        mat_mul(weights->Wo, combined, weights->bo, hidden_size, input_size + hidden_size),
        hidden_size
    );
    double* candidate = vector_tanh(
        mat_mul(weights->Wc, combined, weights->bc, hidden_size, input_size + hidden_size),
        hidden_size
    );

    // Update cell state
    for (int i = 0; i < hidden_size; ++i) {
        states->c[i] = forget_gate[i] * states->c[i] + input_gate[i] * candidate[i];
    }

    // Update hidden state
    for (int i = 0; i < hidden_size; ++i) {
        states->h[i] = output_gate[i] * tanh(states->c[i]);
    }

    // Clean up
    delete[] combined;
    delete[] forget_gate;
    delete[] input_gate;
    delete[] output_gate;
    delete[] candidate;
}

double* LSTM::get_hidden_state() const {
    return states->h;
}

double LSTM::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double LSTM::tanh(double x) {
    return std::tanh(x);
}

double* LSTM::mat_mul(double** mat, double* vec, double* bias, int rows, int cols) {
    double* result = new double[rows];
    for (int i = 0; i < rows; ++i) {
        result[i] = bias[i];
        for (int j = 0; j < cols; ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

double* LSTM::vector_tanh(double* vec, int size) {
    double* result = new double[size];
    for (int i = 0; i < size; ++i) {
        result[i] = tanh(vec[i]);
    }
    return result;
}

double* LSTM::vector_sigmoid(double* vec, int size) {
    double* result = new double[size];
    for (int i = 0; i < size; ++i) {
        result[i] = sigmoid(vec[i]);
    }
    return result;
}
