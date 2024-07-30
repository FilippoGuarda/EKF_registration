#include "ukf.h"
#include <iostream>
#include <cmath>

int main() {
    
    UKF ukf(3, 3);  // 4D state (x, y, vx, vy), 2D measurement (x, y)

    Eigen::VectorXd Vel(3);
    Vel = Eigen::VectorXd::Zero(3);
    // Simulate some measurements
    for (int i = 0; i < 10; ++i) {

        Vel << i*0.01, i*0.02, i*0.01;
        ukf.predict(Vel ,0.1);  // Predict with dt = 0.1

        // Simulated measurement (in this case, just the true position with some noise)
        Eigen::VectorXd z(3);
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
