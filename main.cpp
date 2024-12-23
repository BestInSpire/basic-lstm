#include "LSTMCell/lstm.h"

int main() {
    int input_size = 3;
    int hidden_size = 10;

    LSTM lstm(input_size, hidden_size);

    double input[] = {1.0, 0.5, -1.0};
    lstm.forward(input);

    double* hidden_state = lstm.get_hidden_state();

    std::cout << "Hidden state: ";
    for (int i = 0; i < hidden_size; ++i) {
        std::cout << hidden_state[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
