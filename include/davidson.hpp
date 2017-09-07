//
// Created by Laurent Lemmens on 6/09/17.
//

#ifndef DAVIDSON_DAVIDSON_HPP
#define DAVIDSON_DAVIDSON_HPP

#include <Eigen/Dense>


/** Checks whether a given matrix is symmetric or not
 */
bool is_symmetric(Eigen::MatrixXd& M);



class DavidsonSolver {
private:
    Eigen::VectorXd eigenvalues_;
    Eigen::MatrixXd eigenvectors_;   // as columns

public:
    Eigen::MatrixXd A;
    long dim;
    double tol;

    /** Constructor based on a given symmetric matrix A, number of requested eigenpairs and tolerance
     *
     * @param A:    the matrix that will be diagonalized
     * @param n:    the number of requested eigenpairs
     * @param tol:  the given tolerance (norm of the residual vector) for iteration termination
     */
    DavidsonSolver(Eigen::MatrixXd& A, unsigned& n, double& tol);

    /** Diagonalize the initialized matrix with Davidson's method
     */
    void solve();

    Eigen::VectorXd eigenvalues();
    Eigen::MatrixXd eigenvectors();

};


#endif //DAVIDSON_DAVIDSON_HPP
