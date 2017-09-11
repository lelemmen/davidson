#include "davidson.hpp"
#include <iostream>

/** Checks whether a given square matrix is symmetric (A=AT) or not
 */
bool is_symmetric(Eigen::MatrixXd& A) {
    Eigen::MatrixXd AT = A.transpose();
    return A.isApprox(AT);
}


/** Return the n lowest eigenvalues of a saes (SelfAdjointEigenSolver)
 */
Eigen::VectorXd lowest_evals(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> &saes, unsigned &n) {
    Eigen::VectorXd all_evals = saes.eigenvalues();
    return all_evals.head(n);
}

/** Return the n lowest eigenvectors of a saes (SelfAdjointEigenSolver)
*/
Eigen::MatrixXd lowest_evecs(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> &saes, unsigned &n) {
    auto dim = saes.eigenvalues().rows();
    Eigen::MatrixXd all_evecs = saes.eigenvectors();
    return all_evecs.topLeftCorner(dim, n);
}



/** Constructor based on a given symmetric matrix A, number of requested eigenpairs and tolerance
 *
 * @param A:    the matrix that will be diagonalized
 * @param r:    the number of requested eigenpairs (n lowest eigenvalues)
 * @param tol:  the given tolerance (norm of the residual vector) for iteration termination
 */
DavidsonSolver::DavidsonSolver(Eigen::MatrixXd& A, unsigned& r, double& tol) {
    // If the given matrix is not symmetric, throw an exception
    if (!is_symmetric(A)) {
        throw std::invalid_argument("Given matrix is not symmetric.");
    } else {
        this->A = A;
        this->dim = A.rows();
        this->r = r;

        this->tol = tol;
        this->eigenvalues_ = Eigen::VectorXd::Zero(this->r);                // initialize the eigenvalues and eigenvectors to zero
        this->eigenvectors_ = Eigen::MatrixXd::Zero(this->dim, this->r);
    }
}

/** Diagonalize the initialized matrix with Davidson's method
 */
void DavidsonSolver::solve() {
    // This solver will perform the Davidson-Liu algorithm. For a mathematical explanation, see
    //      (https://github.ugent.be/lelemmen/typesetting/tree/master/Mathemagics).


    // 1. Start with a set of L orthonormalized guess vectors
    //      L will be the dimension of the subspace in which we will diagonalize
    //      FIXME: for now, we start with sqrt(dim)
    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(this->dim, this->dim);      // W is an auxiliary matrix that can be used to expand V
    long L = static_cast<unsigned>(sqrt((this->dim)));
    Eigen::MatrixXd V = W.topLeftCorner(this->dim, L);

    auto converged = false;
    while (!converged) {
        // 2. Construct the subspace matrix
        Eigen::MatrixXd S = V.transpose() * (this->A) * V;


        // 3. Diagonalize that subspace matrix
        //      i.e. find the r lowest eigenvalues and their corresponding eigenvectors
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes (S);
        Eigen::VectorXd Lambda = lowest_evals(saes, this->r);
        Eigen::MatrixXd Z = lowest_evecs(saes, this->r);


        // 4. Construct current eigenvector estimates
        Eigen::MatrixXd X = V * Z;


        // 5. Calculate residual vectors and the correction vectors ...
        //      The residual vectors will be columns in the (this->dim)x(this->r)-matrix R
        //      The (normalized) correction vectors will be columns in the (this->dim)x(this->r)-matrix Delta
        Eigen::MatrixXd R = Eigen::MatrixXd::Zero(this->dim, this->r);
        Eigen::MatrixXd Delta = Eigen::MatrixXd::Zero(this->dim, this->r);
        for (unsigned col_index = 0; col_index < this->r; col_index++) {
            R.col(col_index) = (this->A - Lambda(col_index) * Eigen::MatrixXd::Identity(this->dim, this->dim)) * X.col(col_index);

            // (Coefficient-wise multiplication)
            Eigen::VectorXd prefactor(this->dim);
            for (unsigned i = 0; i < this->dim; i++) {
                // If the prefactor (not inverted) tends to zero, set the prefactor to zero
                if (std::abs(Lambda(col_index) - A(i, i)) < 0.1) {
                    prefactor(i) = 0.0;
                } else {
                    prefactor(i) = 1 / (Lambda(col_index) - A(i, i));
                }
            }

            // ... and normalize the correction vectors
            Delta.col(col_index) = prefactor.cwiseProduct(R.col(col_index));
            Delta.col(col_index).normalize();
        }


        // 6. Project the corrections on the orthogonal complement of the current subspace C(V) ...
        for (unsigned col_index = 0; col_index < this->r; col_index++) {
            Eigen::VectorXd q = (Eigen::MatrixXd::Identity(this->dim, this->dim) - V * V.transpose()) * Delta.col(col_index);
            auto norm = q.norm();

            // ... and if the norm is larger than a certain threshold, expand the subspace
            if (norm > 0.001) {

                // FIXME: work out the math on collapsing subspaces
                if (L + 1 > this->dim) {
                    throw std::range_error( "L got bigger than dim.");
                }

                Eigen::MatrixXd V_copy = V;
                V = W.topLeftCorner(this->dim, ++L);      // Make V bigger by one column
                V.topLeftCorner(this->dim, L-1) = V_copy;
                V.col(L-1) = q / norm;     // Add the new orthonormalized correction
            }
        }


        // 7. Check for convergence before doing other calculations
        //      If any of norms of the residuals isn't tolerated, the calculation isn't converged
        converged = true;
        for (unsigned col_index = 0; col_index < this->r; col_index++) {
            if (R.col(col_index).norm() > tol) {
                converged = false;
            }
        }
        if (converged) {
            this->eigenvalues_ = Lambda;
            this->eigenvectors_ = X;
        }
    }
}


Eigen::VectorXd DavidsonSolver::eigenvalues() {
    return (this->eigenvalues_);
}

Eigen::MatrixXd DavidsonSolver::eigenvectors() {
    return (this->eigenvectors_);
}