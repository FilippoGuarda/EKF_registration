#pragma once
#include <Eigen/Dense> 

class UKF
{

public:

    static const int NUM_VARS = 3;
    static const int iX = 0;
    static const int iV = 0;


    using Vector =  Eigen::Matrix<double, NUM_VARS, 1>;
    using Matrix =  Eigen::Matrix<double, NUM_VARS, NUM_VARS>;

    class UKF(Vector initialX, Matrix velocityVariance)

    {

    }

    void predict(Vector input_velocity, double dt)
    {

    } 

    void update(Vector measValue, Matrix MeasVariance)
    {

    }

    Matrix cov() const
    {
        return Matrix::Identity();
    }

    Vector mean() const
    {
        return Vector::Zero();
    }

    Vector pos() const
    {
        return Vector::Zero();
    }

    Vector vel() const
    {
        return Vector::Zero();
    }
}   