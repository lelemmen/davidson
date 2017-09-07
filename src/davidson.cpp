#include "davidson.hpp"
#include <iostream>

/** Checks whether a given matrix is symmetric or not
 */
bool is_symmetric(Eigen::MatrixXd& A) {
    auto AT = A.transpose();
    return A.isApprox(AT);
}


/** Constructor based on a given symmetric matrix A, number of requested eigenparis and tolerance
 *
 * @param A:    the matrix that will be diagonalized
 * @param n:    the number of requested eigenpairs
 * @param tol:  the given tolerance (norm of the residual vector) for iteration termination
 */
DavidsonSolver::DavidsonSolver(Eigen::MatrixXd& A, unsigned& n, double& tol) {
    // If the given matrix is not symmetric, throw an exception
    if (!is_symmetric(A)) {
        throw std::invalid_argument("Given matrix is not symmetric.");
    } else {
        this->A = A;
        this->dim = A.rows();

        this->tol = tol;
        this->eigenvalues_ = Eigen::VectorXd::Zero(n);       // initialize the eigenvalues and eigenvectors to zero
        this->eigenvectors_ = Eigen::MatrixXd::Zero(this->dim, n);
    }
}

/** Diagonalize the initialized matrix with Davidson's method
 */
void DavidsonSolver::solve() {
    // See (https://github.ugent.be/lelemmen/typesetting/tree/master/Mathemagics) for a mathematical explanation of Davidson's algorithm

    // For now, we do maximum three iterations
    //      This means that we start with x1=e1 and add an orthogonal vector twice
    long subspace_dim = 3;
    Eigen::MatrixXd V = Eigen::MatrixXd::Zero(this->dim, subspace_dim);

    // x1=e1
    Eigen::VectorXd x = Eigen::VectorXd::Unit(this->dim, 0);

    // Append x1 to V
    V.col(0) = x;

    // Calculate lambda
    double lambda = x.transpose() * (this->A) * x;

    // Calculate the first residual r
    Eigen::VectorXd r = (this->A) * x - lambda * x;

    unsigned k=1;
    while (r.norm() > this->tol) {
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
        V.col(k % subspace_dim) = v;

        // Diagonalize S. Since it's symmetric, we can use the SelfAdjointEigenSolver
        Eigen::MatrixXd S = V.transpose() * (this->A) * V;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes (S);

        // Find the appropriate Ritz pair (eigenvalue closest to lambda)
        Eigen::VectorXd lambdas = lambda * Eigen::VectorXd::Ones(subspace_dim);      // We're in this example looking at a three-dimensional subspace
        Eigen::MatrixXd::Index min_index;
        Eigen::VectorXd diff = saes.eigenvalues() - lambdas;
        diff.cwiseAbs().minCoeff(&min_index);

        // Find ritz pair
        lambda = saes.eigenvalues()(min_index);
        Eigen::MatrixXd evec = saes.eigenvectors();
        Eigen::VectorXd t = evec.col(min_index);
        x = V * t;

        // Re-calculate the residue
        r = (this->A) * x - lambda * x;
        k++;
    }

    // After convergence, set the eigenvalue(s) and eigenvector(s) in the DavidsonSolver instance
    this->eigenvalues_(0) = lambda;
    this->eigenvectors_.col(0) = x;
}


Eigen::VectorXd DavidsonSolver::eigenvalues() {
    return this->eigenvalues_;
}

Eigen::MatrixXd DavidsonSolver::eigenvectors() {
    return this->eigenvectors_;
}