#include <algorithm>
//#include <chrono>
#include <pybind11/pybind11.h>
#include <iostream>
#include <exception>
#include "linear_regression.h"

using namespace std;
namespace py=pybind11;

LinearRegression::LinearRegression(): x(){}

void LinearRegression::fit(Matrix X, Matrix y) {
    // Resolvemos con ecuaciones normales
    //   AtA x = At y
    
    // https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
    // Uso QR con Householder porque es simple
    Matrix Xt = X.transpose();
    this->x = (Xt*X).colPivHouseholderQr().solve(Xt*y);
}

Matrix LinearRegression::predict(Matrix X) {
    return X*x;
}

Matrix LinearRegression::get_x() {
    return x;
}