#include <vector>
#include "./LSTMCell/LSTMCell.h"


int main() {
    int input_size = 5;
    int hidden_size = 4;

    LSTMCell lstm(input_size, hidden_size);


    vector<vector<double>> time_series_data = {
        {1.0, 0.5, -1.5, 0.75, -1.3},
        {0.8, 0.2, -1.0, 0.60, -0.9},
        {1.2, 0.7, -1.2, 0.90, -1.1}
    };

    for (const auto& input : time_series_data) {
        lstm.forward(input);
    }

    return 0;
}
