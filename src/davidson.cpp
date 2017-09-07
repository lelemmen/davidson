#include "davidson.hpp"
#include <iostream>

/** Checks whether a given matrix is symmetric or not
 */
bool is_symmetric(Eigen::MatrixXd& A) {
    auto AT = A.transpose();
    return A.isApprox(AT);
}


/** Constructor based on a given symmetric matrix A
 *
 * @param A:    the matrix that will be diagonalized
 */
DavidsonSolver::DavidsonSolver(Eigen::MatrixXd& A) {
    // If the given matrix is not symmetric, throw an exception
    if (!is_symmetric(A)) {
        throw std::invalid_argument("Given matrix is not symmetric.");
    } else {
        this->A = A;
        this->dim = A.cols();
    }
}

/** Diagonalize the initialized matrix with Davidson's method
 */
void DavidsonSolver::solve() {
    // See (https://github.ugent.be/lelemmen/typesetting/tree/master/Mathemagics) for a mathematical explanation of Davidson's algorithm

    // For now, we do maximum three iterations
    //      This means that we start with x1=e1 and add an orthogonal vector twice
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(this->dim, 3);

    // x1=e1
    Eigen::VectorXd x = Eigen::VectorXd::Unit(this->dim, 0);

    // Append x1 to V
    V.col(0) = x;

    // Calculate lambda
    double lambda = x.transpose() * (this->A) * x;

    // Calculate the first residual r
    Eigen::VectorXd r = (this->A) * x - lambda * x;

    for (int k = 2; k <= 3; k++) {
        // A'
        Eigen::MatrixXd A_ = (this->A).diagonal().asDiagonal();

        // Preconditioning step
        // Solve residual equation B dv = -r
        Eigen::MatrixXd B = (A_ - lambda * Eigen::MatrixXd::Identity(this->dim, this->dim));

        Eigen::VectorXd dv = B.householderQr().solve(r);  // Maybe I could invert the diagonal matrix B myself
        // This is actually element-wise division
        // Check if lambda is approximately equal to a diagonal element and if so, set the correction = 0
        for (int i = 0; i < this->dim; i++) {
            if (this->A(i, i) == lambda) {   // Check if close to zero
                dv(i) = 0;
            }
        }

        // Expanding the subspace
        Eigen::VectorXd s = (Eigen::MatrixXd::Identity(this->dim, this->dim) - V * V.transpose()) * dv;
        Eigen::VectorXd v = s / s.norm();
        V.col(k-1) = v;

        // Diagonalize S. Since it's symmetric, we can use the SelfAdjointEigenSolver
        Eigen::MatrixXd S = V.transpose() * (this->A) * V;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes (S);

        // Find the appropriate Ritz pair (eigenvalue closest to lambda)
        Eigen::VectorXd lambdas = lambda * Eigen::VectorXd::Ones(3);      // We're in this example looking at a three-dimensional subspace
        Eigen::MatrixXd::Index min_index;
        Eigen::VectorXd diff = saes.eigenvalues() - lambdas;
        diff.cwiseAbs().minCoeff(&min_index);

        // Find ritz pair
        lambda = saes.eigenvalues()(min_index);
        Eigen::MatrixXd evec = saes.eigenvectors();
        Eigen::VectorXd t = evec.col(min_index);
        Eigen::VectorXd x = V * t;

        // Re-calculate the residue
        r = (this->A) * x - lambda * x;
    }








}