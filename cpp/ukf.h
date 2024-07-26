#pragma once
#include <Eigen/Dense> 

class UKF
{

public:

    static const int NUM_VARS = 3;
    static const int iX = 0;
    static const int iY = 1;
    static const int iTheta = 2;



    using Vector =  Eigen::Matrix<double, NUM_VARS, 1>;
    using Matrix =  Eigen::Matrix<double, NUM_VARS, NUM_VARS>;

    UKF (Vector initialX, Matrix velocityVariance) :  m_velocityVariance(velocityVariance)
    {
        m_mean = initialX;


        m_cov.setIdentity();
    }

    void predict(double dt)
    {
        Matrix F;
        F.setIdentity();
        F(iX, iV) = dt;

        const Vector newX = F * m_mean;

        Vector G;
        G(iX) = 0.5 * dt * dt;
        G(iV) = dt;

        const Matrix newP = F * m_cov * F.transpose() + G * G.transpose() * m_accelVariance;

        m_cov = newP;
        m_mean = newX;
    }

    void update(double measValue, double measVariance)
    {
        Eigen::Matrix<double, 1, NUM_VARS> H;
        H.setZero();
        H(0, iX) = 1;

        const double y = measValue - H * m_mean;
        const double S = H * m_cov * H.transpose() + measVariance;

        const Vector K = m_cov * H.transpose() * 1.0 / S;

        Vector newX = m_mean + K * y;
        Matrix newP = (Matrix::Identity() - K * H) * m_cov;

        m_cov = newP;
        m_mean = newX;
    }

    void update(Vector measValue, Matrix MeasVariance)
    {  
        return m_mean;
    }

    Matrix cov() const
    {
        return m_cov;
    }

    Vector mean() const
    {
        return m_mean;
    }

    Vector pos() const
    {
        return m_mean(iX);
    }

private:
    Vector m_mean;
    Matrix m_cov;

    const Vector m_velocityVariance;

}  ;
