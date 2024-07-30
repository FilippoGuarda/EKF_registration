#ifndef UKF_H
#define UKF_H

#include <Eigen/Dense>

class UKF {
private:
    int n_x; // State dimension
    int n_z; // Measurement dimension
    double alpha, beta, kappa;
    double lambda;
    Eigen::VectorXd x; // State vector
    Eigen::MatrixXd P; // State covariance matrix
    Eigen::MatrixXd Q; // Process noise covariance
    Eigen::MatrixXd R; // Measurement noise covariance

    Eigen::MatrixXd generateSigmaPoints(const Eigen::VectorXd& x, const Eigen::MatrixXd& P);
    Eigen::VectorXd calculateWeights();
    Eigen::VectorXd processModel(Eigen::VectorXd Vel, const Eigen::VectorXd& x, double dt);
    Eigen::VectorXd measurementModel(const Eigen::VectorXd& x);

public:
    UKF(int state_dim, int meas_dim);
    void predict(Eigen::VectorXd& Vel, double dt);
    void update(const Eigen::VectorXd& z_measurement);
    void printState();
};

#endif // UKF_H