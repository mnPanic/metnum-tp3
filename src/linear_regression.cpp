#include <algorithm>
//#include <chrono>
#include <pybind11/pybind11.h>
#include <iostream>
#include <exception>
#include "linear_regression.h"

using namespace std;
namespace py=pybind11;

LinearRegression::LinearRegression()
{
}

void LinearRegression::fit(Matrix X, Matrix y)
{
}


Matrix LinearRegression::predict(Matrix X)
{
    auto ret = MatrixXd::Zero(X.rows(), 1);

    return ret;
}
