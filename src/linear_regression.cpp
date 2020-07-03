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
    //       x = AtA^{-1} At y

    // TODO: Hacer sin invertir?
    Matrix Xt = X.transpose();
    this->x = (Xt*X).inverse() * Xt*y;
}

Matrix LinearRegression::predict(Matrix X) {
    return X*x;
}

Matrix LinearRegression::get_x() {
    return x;
}