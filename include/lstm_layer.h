#ifndef LSTM_LAYER_H
#define LSTM_LAYER_H

#include <Eigen/Dense>

using namespace Eigen;
class LSTM {
public:
    LSTM(int input_size, int hidden_size);
    void forward(const MatrixXd &input);

private:
    int input_size;
    int hidden_size;

    MatrixXd Wf, Wi, Wo, Wc;
    MatrixXd Uf, Ui, Uo, Uc;
    VectorXd bf, bi, bo, bc;

    VectorXd h_prev, c_prev;
    VectorXd ft, it, ot, cct, ht, ct;
};

#endif // LSTM_LAYER_H
