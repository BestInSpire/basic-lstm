#include "../include/lstm_layer.h"
#include "../include/activation_functions.h"

using namespace Eigen;

LSTM::LSTM(int input_size, int hidden_size)
    : input_size(input_size),
    hidden_size(hidden_size),
    Wf(MatrixXd::Random(hidden_size, input_size)),
    Wi(MatrixXd::Random(hidden_size, input_size)),
    Wo(MatrixXd::Random(hidden_size, input_size)),
    Wc(MatrixXd::Random(hidden_size, input_size)),
    Uf(MatrixXd::Random(hidden_size, hidden_size)),
    Ui(MatrixXd::Random(hidden_size, hidden_size)),
    Uo(MatrixXd::Random(hidden_size, hidden_size)),
    Uc(MatrixXd::Random(hidden_size, hidden_size)),
    bf(VectorXd::Random(hidden_size)), bi(VectorXd::Random(hidden_size)),
    bo(VectorXd::Random(hidden_size)), bc(VectorXd::Random(hidden_size)),
    h_prev(VectorXd::Zero(hidden_size)), c_prev(VectorXd::Zero(hidden_size)) {}

    void LSTM::forward(const MatrixXd &input) {
    // ft = σ(Wf * input + Uf * h_prev + bf)
    ft = sigmoid((Wf * input).colwise() + Uf * h_prev + bf);
    // it = σ(Wi * input + Ui * h_prev + bi)
    it = sigmoid((Wi * input).colwise() + Ui * h_prev + bi);
    // cct = tanh(Wc * input + Uc * h_prev + bc)
    cct = tanh((Wc * input).colwise() + Uc * h_prev + bc);
    // c_next = ft * c_prev + it * cct
    ct = ft.array() * c_prev.array() + it.array() * cct.array();
    // ot = σ(Wo * input + Uo * h_prev + bo)
    ot = sigmoid((Wo * input).colwise() + Uo * h_prev + bo);
    // h_next = ot * tanh(c_next)

    ht = ot.array() * tanh(ct).array();
    h_prev = ht; c_prev = ct;
}