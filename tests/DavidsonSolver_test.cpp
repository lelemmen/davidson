#define BOOST_ALL_DYN_LINK

#define BOOST_TEST_MODULE "Davidson diagonalization"

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>  // include this to get main(), otherwise clang++ will complain
#include <Eigen/Eigenvalues>
#include "davidson.hpp"

/** Check if two sets of eigenvalues are equal
 */
bool are_equal_evals(Eigen::VectorXd& evals1, Eigen::VectorXd& evals2, double& tol) {
    return evals1.isApprox(evals2, tol);
}

/** Check if two sets of eigenvectors are equal (up to a factor -1)
 */
bool are_equal_evecs(Eigen::MatrixXd& evecs1, Eigen::MatrixXd& evecs2, double& tol) {
    return evecs1.isApprox(evecs2, tol) || evecs1.isApprox(-evecs2, tol);
}


BOOST_AUTO_TEST_CASE( constructor ) {
    int dim = 5;
    unsigned n = 1;
    double tol = 0.001;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(dim, dim);
    Eigen::MatrixXd XT = X.transpose();
    Eigen::MatrixXd S = 0.5 * (X + XT);

    // Test if DavidsonSolver refuses to accept a non-symmetric matrix
    BOOST_CHECK_THROW(DavidsonSolver ds1 (X, n, tol), std::invalid_argument);

    // Test if DavidsonSolver accepts symmetric matrices
    BOOST_CHECK_NO_THROW(DavidsonSolver ds2 (S, n, tol));
}


BOOST_AUTO_TEST_CASE( esqc_example_solver ){
    // We can find the following example at (http://www.esqc.org/static/lectures/Malmqvist_2B.pdf)
    // Build up the example matrix
    Eigen::MatrixXd A = Eigen::MatrixXd::Constant(5, 5, 0.1);
    A(0,0) = 1.0;
    A(1,1) = 2.0;
    A(2,2) = 3.0;
    A(3,3) = 3.0;
    A(4,4) = 3.0;

    // The solutions to the problem are given in the example
    Eigen::VectorXd eval_ex (1);
    eval_ex << 0.979;
    Eigen::MatrixXd evec_ex (5, 1);
    evec_ex << 0.994, -0.083, -0.042, -0.042, -0.042;

    // Solve using the Davidson diagonalization
    unsigned n = 1;
    double tol = 0.05;
    DavidsonSolver ds (A, n, tol);
    ds.solve();
    auto eval_d = ds.eigenvalues();
    auto evec_d = ds.eigenvectors();

    // Test if the example solutions are equal to the Davidson solutions
    BOOST_CHECK(are_equal_evals(eval_d, eval_ex, tol));
    BOOST_CHECK(are_equal_evecs(evec_d, evec_ex, tol));
}


BOOST_AUTO_TEST_CASE( liu_example ){

    

}

/*
BOOST_AUTO_TEST_CASE( bigger_example_three ) {

    // Let's make a diagonally dominant, but random matrix
    int dim = 15;
    Eigen::MatrixXd A = Eigen::MatrixXd::Constant(dim, dim, 0.1);

    // First, we put i+1 on the diagonal
    for (int i = 0; i < dim; i++) {
        A(i, i) = i + 1;
    }

    // Now it's time to solve the eigensystem for S
    // Afterwards, select the 3 lowest eigenpairs
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es (A);
    auto evals_ex = es.eigenvalues();
    evals_ex = evals_ex.head(3);    // can't do this in the previous line; eigenvalues is const
    auto evecs_ex = es.eigenvectors();
    evecs_ex = evecs_ex.topLeftCorner(dim, 3);

    std::cout << "evals ex" << std::endl << evals_ex << std::endl << std::endl;
    std::cout << "evecs ex" << std::endl << evecs_ex << std::endl << std::endl;

    // Now we have to test if my Davidson solver gives the same results
    unsigned n = 3;
    double tol = 0.0001;
    DavidsonSolver ds (A, n, tol);
    ds.solve();

    auto evals_d = ds.eigenvalues();
    auto evecs_d = ds.eigenvectors();

    std::cout << "evals d" << std::endl << evals_d << std::endl << std::endl;
    std::cout << "evecs d" << std::endl << evecs_d << std::endl << std::endl;

}
*/
