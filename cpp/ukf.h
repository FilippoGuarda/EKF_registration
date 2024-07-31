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
    Eigen::VectorXd in_var; // Input Variance
    Eigen::MatrixXd ms_var; // Measure Variance

    Eigen::MatrixXd generateSigmaPoints(const Eigen::VectorXd& x, const Eigen::MatrixXd& P);
    Eigen::VectorXd calculateWeights();
    Eigen::VectorXd processModel(Eigen::VectorXd& Vel, const Eigen::VectorXd& x, double dt);
    Eigen::VectorXd measurementModel(const Eigen::VectorXd& x);

public:
    UKF(int state_dim, int meas_dim, Eigen::VectorXd in_vars, Eigen::MatrixXd meas_vars);
    void predict(Eigen::VectorXd& Vel, double dt);
    void update(Eigen::VectorXd& z_measurement);
    void printState();

    // traslate result form vectorX to matrixX for compatibility with pybind11
    Eigen::VectorXd mean() const;
    Eigen::MatrixXd cov() const;
};

#endif // UKF_H