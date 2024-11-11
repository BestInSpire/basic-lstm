#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H
#include <Eigen/Dense>

using namespace Eigen;

MatrixXd sigmoid(const MatrixXd &x);
MatrixXd tanh(const MatrixXd &x);
#endif // ACTIVATION_FUNCTIONS_H