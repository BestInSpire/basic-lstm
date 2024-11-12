#include "../include/activation_functions.h";
using namespace Eigen;

MatrixXd sigmoid(const MatrixXd &x) {
    return 1.0 / (1.0 + (-x.array()).exp());
}
MatrixXd tanh(const MatrixXd &x) {
    return x.array().tanh();
}

