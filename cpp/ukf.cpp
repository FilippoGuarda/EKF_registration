#include "ukf.h"
#include <iostream>


// class initialization
UKF::UKF(int state_dim, int meas_dim, Eigen::VectorXd in_vars, Eigen::MatrixXd meas_vars) 
    : n_x(state_dim), n_z(meas_dim), in_var(in_vars), ms_var(meas_vars){
    alpha = 1e-3;
    beta = 2.0;
    kappa = 0.0;
    lambda = alpha * alpha * (n_x + kappa) - n_x;

    x = Eigen::VectorXd::Zero(n_x);
    P = Eigen::MatrixXd::Identity(n_x, n_x);
    // had to use the following walkaround because the compiler kept crashing wtf
//    Q = Eigen::MatrixXd::Identity(n_x, n_x)*in_var.transpose();
    Q = in_var.asDiagonal();
    R = Eigen::MatrixXd::Identity(n_z, n_z)* ms_var.transpose();
}

// generation of the UKF kernel points (sigma)
Eigen::MatrixXd UKF::generateSigmaPoints(const Eigen::VectorXd& x, const Eigen::MatrixXd& P) {
    int n = x.size();
    int num_sigma_points = 2 * n + 1;
    Eigen::MatrixXd sigma_points(n, num_sigma_points);
    
    Eigen::MatrixXd L = ((n + lambda) * P).llt().matrixL();
    
    sigma_points.col(0) = x;
    for (int i = 0; i < n; ++i) {
        sigma_points.col(i + 1) = x + L.col(i);
        sigma_points.col(n + i + 1) = x - L.col(i);
    }
    
    return sigma_points;
}

// generation of the weights of the sigma points
Eigen::VectorXd UKF::calculateWeights() {
    int num_sigma_points = 2 * n_x + 1;
    Eigen::VectorXd weights(num_sigma_points);
    
    double w0 = lambda / (n_x + lambda);
    weights(0) = w0;
    for (int i = 1; i < num_sigma_points; ++i) {
        weights(i) = 1 / (2 * (n_x + lambda));
    }
    
    return weights;
}

// TODO: Update model to the robot's one
Eigen::VectorXd UKF::processModel(Eigen::VectorXd& Vel, const Eigen::VectorXd& x, double dt) {

    Eigen::VectorXd x_pred = x;
    x_pred(0) += (-Vel(0)*std::cos(x(2)) -Vel(1)*std::sin(x(2)))*dt; // x = x + vx * dt
    x_pred(1) += (-Vel(0)*std::sin(x(2)) -Vel(1)*std::cos(x(2)))*dt; // y = y + sin(theta) * dt
    x_pred(2) +=  -Vel(2)*dt; // theta = theta + 0.01dt
    return x_pred;
}

// in our case this should be an Identity matrix
Eigen::VectorXd UKF::measurementModel(const Eigen::VectorXd& x) {

    Eigen::VectorXd z(3);
    z << x(0), x(1), x(2);
    return z;
}


void UKF::predict(Eigen::VectorXd& Vel, double dt) {
    // Generate sigma points
    Eigen::MatrixXd sigma_points = generateSigmaPoints(x, P);
    Eigen::VectorXd weights = calculateWeights();
    
    // Predict sigma points
    Eigen::MatrixXd predicted_sigma_points(n_x, 2*n_x+1);
    for (int i = 0; i < 2*n_x+1; ++i) {
        predicted_sigma_points.col(i) = processModel(Vel, sigma_points.col(i), dt);
    }
    
    // Calculate predicted mean and covariance
    x = predicted_sigma_points * weights;
    
    P = Eigen::MatrixXd::Zero(n_x, n_x);
    for (int i = 0; i < 2*n_x+1; ++i) {
        Eigen::VectorXd diff = predicted_sigma_points.col(i) - x;
        P += weights(i) * diff * diff.transpose();
    }
    P += Q;
}

void UKF::update(Eigen::VectorXd& z_measurement) {
    // Generate sigma points
    Eigen::MatrixXd sigma_points = generateSigmaPoints(x, P);
    Eigen::VectorXd weights = calculateWeights();
    
    // Transform sigma points into measurement space
    Eigen::MatrixXd z_sigma_points(n_z, 2*n_x+1);
    for (int i = 0; i < 2*n_x+1; ++i) {
        z_sigma_points.col(i) = measurementModel(sigma_points.col(i));
    }
    
    // Calculate predicted measurement mean and covariance
    Eigen::VectorXd z_pred = z_sigma_points * weights;
    
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(n_z, n_z);
    for (int i = 0; i < 2*n_x+1; ++i) {
        Eigen::VectorXd diff = z_sigma_points.col(i) - z_pred;
        S += weights(i) * diff * diff.transpose();
    }
    S += R;
    
    // Calculate cross-covariance matrix
    Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(n_x, n_z);
    for (int i = 0; i < 2*n_x+1; ++i) {
        Eigen::VectorXd x_diff = sigma_points.col(i) - x;
        Eigen::VectorXd z_diff = z_sigma_points.col(i) - z_pred;
        Pxz += weights(i) * x_diff * z_diff.transpose();
    }
    
    // Calculate Kalman gain
    Eigen::MatrixXd K = Pxz * S.inverse();
    
    // Update state and covariance
    x += K * (z_measurement - z_pred);
    P -= K * S * K.transpose();
}


// remember to add the eigen includes in the pybind wrapper!!
Eigen::VectorXd UKF::mean() const{
    return x;
}

Eigen::MatrixXd UKF::cov() const{
    return P;
}

void UKF::printState() {
    std::cout << "State: " << x.transpose() << std::endl;
    std::cout << "Covariance:\n" << P << std::endl;
}
