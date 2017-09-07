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
public:
    Eigen::MatrixXd A;

    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;   // as columns

    long dim;

    /** Constructor based on a given symmetric matrix A
     *
     * @param A:    the matrix that will be diagonalized
     */
    DavidsonSolver(Eigen::MatrixXd& A);

    /** Diagonalize the initialized matrix with Davidson's method
     */
    void solve();
};


#endif //DAVIDSON_DAVIDSON_HPP
