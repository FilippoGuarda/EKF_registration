#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using Matrix = Eigen::Matrix
using Vector = Eigen::Vector


class UKF {
private:
    int n_x; // State dimension
    int n_z; // Measurement dimension
    double alpha, beta, kappa;
    double lambda;
    Vector x; // State vector
    Matrix P; // State covariance matrix
    Matrix Q; // Process noise covariance
    Matrix R; // Measurement noise covariance

public:
    UKF(int state_dim, int meas_dim) : n_x(state_dim), n_z(meas_dim) {
        alpha = 1e-3;
        beta = 2.0;
        kappa = 0.0;
        lambda = alpha * alpha * (n_x + kappa) - n_x;

        x = Vector::Zero(n_x);
        P = Matrix::Identity(n_x, n_x);
        Q = Matrix::Identity(n_x, n_x) * 0.1;
        R = Matrix::Identity(n_z, n_z) * 0.1;
    }

    Matrix generateSigmaPoints(const Vector& x, const Matrix& P) {
        int n = x.size();
        int num_sigma_points = 2 * n + 1;
        Matrix sigma_points(n, num_sigma_points);
        
        Matrix L = ((n + lambda) * P).llt().matrixL();
        
        sigma_points.col(0) = x;
        for (int i = 0; i < n; ++i) {
            sigma_points.col(i + 1) = x + L.col(i);
            sigma_points.col(n + i + 1) = x - L.col(i);
        }
        
        return sigma_points;
    }

    Vector calculateWeights() {
        int num_sigma_points = 2 * n_x + 1;
        Vector weights(num_sigma_points);
        
        double w0 = lambda / (n_x + lambda);
        weights(0) = w0;
        for (int i = 1; i < num_sigma_points; ++i) {
            weights(i) = 1 / (2 * (n_x + lambda));
        }
        
        return weights;
    }

    Vector processModel(const Vector& x, double dt) {
        // Implement your process model here
        // This is a simple example assuming constant velocity model
        Vector x_pred = x;
        x_pred(0) += x(2) * dt; // x = x + vx * dt
        x_pred(1) += x(3) * dt; // y = y + vy * dt
        return x_pred;
    }

    Vector measurementModel(const Vector& x) {
        // Implement your measurement model here
        // This is a simple example assuming we measure position only
        Vector z(2);
        z << x(0), x(1);
        return z;
    }

    void predict(double dt) {
        // Generate sigma points
        Matrix sigma_points = generateSigmaPoints(x, P);
        Vector weights = calculateWeights();
        
        // Predict sigma points
        Matrix predicted_sigma_points(n_x, 2*n_x+1);
        for (int i = 0; i < 2*n_x+1; ++i) {
            predicted_sigma_points.col(i) = processModel(sigma_points.col(i), dt);
        }
        
        // Calculate predicted mean and covariance
        x = predicted_sigma_points * weights;
        
        P = Matrix::Zero(n_x, n_x);
        for (int i = 0; i < 2*n_x+1; ++i) {
            Vector diff = predicted_sigma_points.col(i) - x;
            P += weights(i) * diff * diff.transpose();
        }
        P += Q;
    }

    void update(const Vector& z_measurement) {
        // Generate sigma points
        Matrix sigma_points = generateSigmaPoints(x, P);
        Vector weights = calculateWeights();
        
        // Transform sigma points into measurement space
        Matrix z_sigma_points(n_z, 2*n_x+1);
        for (int i = 0; i < 2*n_x+1; ++i) {
            z_sigma_points.col(i) = measurementModel(sigma_points.col(i));
        }
        
        // Calculate predicted measurement mean and covariance
        Vector z_pred = z_sigma_points * weights;
        
        Matrix S = Matrix::Zero(n_z, n_z);
        for (int i = 0; i < 2*n_x+1; ++i) {
            Vector diff = z_sigma_points.col(i) - z_pred;
            S += weights(i) * diff * diff.transpose();
        }
        S += R;
        
        // Calculate cross-covariance matrix
        Matrix Pxz = Matrix::Zero(n_x, n_z);
        for (int i = 0; i < 2*n_x+1; ++i) {
            Vector x_diff = sigma_points.col(i) - x;
            Vector z_diff = z_sigma_points.col(i) - z_pred;
            Pxz += weights(i) * x_diff * z_diff.transpose();
        }
        
        // Calculate Kalman gain
        Matrix K = Pxz * S.inverse();
        
        // Update state and covariance
        x += K * (z_measurement - z_pred);
        P -= K * S * K.transpose();
    }

    void printState() {
        std::cout << "State: " << x.transpose() << std::endl;
        std::cout << "Covariance:\n" << P << std::endl;
    }
};

int main() {
    UKF ukf(3, 3);  // 4D state (x, y, vx, vy), 2D measurement (x, y)

    // Simulate some measurements
    for (int i = 0; i < 10; ++i) {
        ukf.predict(0.1);  // Predict with dt = 0.1

        // Simulated measurement (in this case, just the true position with some noise)
        Vector z(3);
        z << i * 0.1 + 0.05 * std::sin(i),
             i * 0.1 + 0.05 * std::sin(i),  
             i * 0.1 + 0.05 * std::cos(i);

        ukf.update(z);

        std::cout << "Step " << i << ":" << std::endl;
        ukf.printState();
        std::cout << std::endl;
    }

    return 0;
}
