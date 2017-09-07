#include "davidson.hpp"
#include <iostream>
#include <Eigen/Eigenvalues>
#include <math.h>


bool is_symmetric(Eigen::MatrixXd& A) {
    auto AT = A.transpose();
    return A.isApprox(AT);
}



DavidsonSolver::DavidsonSolver(Eigen::MatrixXd& A) {
    // If the given matrix is not symmetric, throw an exception
    if (!is_symmetric(A)) {
        throw std::invalid_argument("Given matrix is not symmetric.");
    } else {
        this->A = A;
        this->dim = A.cols();
    }
}

/** Use Davidson's iterative method to find (one of) the eigenvalue(s) and eigenvector(s) of a given symmetric (diagonally-dominant) matrix
 *
 */
void DavidsonSolver::solve() {
    // See (https://github.ugent.be/lelemmen/typesetting/tree/master/Mathemagics) for a mathematical explanation of Davidson's algorithm

    std::cout << "A" << std::endl << this->A << std::endl << std::endl;

    // For now, we do maximum three iterations
    //      This means that we start with x1=e1 and add an orthogonal vector twice
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(this->dim, 3);


    // x1=e1
    Eigen::VectorXd x = Eigen::VectorXd::Unit(this->dim, 0);
    std::cout << "x" << std::endl << x << std::endl << std::endl;

    // Append x1 to V
    V.col(0) = x;
    // std::cout << "V" << std::endl << V << std::endl << std::endl;

    // Calculate lambda
    double lambda = x.transpose() * (this->A) * x;
    std::cout << "lambda" << std::endl << lambda << std::endl << std::endl;

    // Calculate the first residual r
    Eigen::VectorXd r = (this->A) * x - lambda * x;
    std::cout << "r" << std::endl << r << std::endl << std::endl;


    for (int k = 2; k <= 3; k++) {
        // A'
        Eigen::MatrixXd A_ = (this->A).diagonal().asDiagonal();
        std::cout << "A_" << std::endl << A_ << std::endl << std::endl;

        // Preconditioning step
        // Solve residual equation B dv = -r
        Eigen::MatrixXd B = (A_ - lambda * Eigen::MatrixXd::Identity(this->dim, this->dim));

        //std::cout << "B" << std::endl << B << std::endl << std::endl;
        Eigen::VectorXd dv = B.householderQr().solve(r);  // Maybe I could invert the diagonal matrix B myself
        // This is actually element-wise division
        // Check if lambda is approximately equal to a diagonal element and if so, set the correction = 0
        for (int i = 0; i < this->dim; i++) {
            if (this->A(i, i) == lambda) {   // Check if close to zero
                dv(i) = 0;
            }
        }
        std::cout << "dv" << std::endl << dv << std::endl << std::endl;

        // Expanding the subspace
        Eigen::VectorXd s = (Eigen::MatrixXd::Identity(this->dim, this->dim) - V * V.transpose()) * dv;
        Eigen::VectorXd v = s / s.norm();
        V.col(k-1) = v;
        std::cout << "V" << std::endl << V << std::endl << std::endl;

        // Diagonalize S. Since it's symmetric, we can use the SelfAdjointEigenSolver
        Eigen::MatrixXd S = V.transpose() * (this->A) * V;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes (S);
        std::cout << "S eigenvalues" << std::endl << saes.eigenvalues() << std::endl << std::endl;
        std::cout << "S eigenvectors" << std::endl << saes.eigenvectors() << std::endl << std::endl;

        // Find the appropriate Ritz pair (eigenvalue closest to lambda)
        Eigen::VectorXd lambdas = lambda * Eigen::VectorXd::Ones(3);      // We're in this example looking at a three-dimensional subspace
        Eigen::MatrixXd::Index min_index;
        Eigen::VectorXd diff = saes.eigenvalues() - lambdas;
        std::cout << "diff cwise abs" << std::endl << diff.cwiseAbs() << std::endl << std::endl;
        diff.cwiseAbs().minCoeff(&min_index);
        std::cout << "min_index: " << min_index << std::endl << std::endl;  // appropriate Ritz pair located @ min_index

        // Find ritz pair
        lambda = saes.eigenvalues()(min_index);
        std::cout << "lambda2: " << lambda << std::endl << std::endl;
        Eigen::MatrixXd evec = saes.eigenvectors();
        Eigen::VectorXd t = evec.col(min_index);
        Eigen::VectorXd x = V * t;
        std::cout << "x" << std::endl << x << std::endl << std::endl;

        // Re-calculate the residue
        r = (this->A) * x - lambda * x;
    }








}