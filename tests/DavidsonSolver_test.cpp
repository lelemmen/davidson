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
    auto dim = evecs1.cols();
    for (int i = 0; i < dim; i++) {
        if (!(evecs1.col(i).isApprox(evecs2.col(i), tol) || evecs1.col(i).isApprox(-evecs2.col(i), tol))) {
            return false;
        }
    }
    return true;
}


BOOST_AUTO_TEST_CASE( constructor ) {
    std::cout << "\tRunning test case 'constructor'" << std::endl;

    int dim = 5;
    unsigned r = 1;
    double tol = 0.001;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(dim, dim);
    Eigen::MatrixXd XT = X.transpose();
    Eigen::MatrixXd S = 0.5 * (X + XT);

    // Test if DavidsonSolver refuses to accept a non-symmetric matrix
    BOOST_CHECK_THROW(DavidsonSolver ds1 (X, r, tol), std::invalid_argument);

    // Test if DavidsonSolver accepts symmetric matrices
    BOOST_CHECK_NO_THROW(DavidsonSolver ds2 (S, r, tol));
}


BOOST_AUTO_TEST_CASE( esqc_example_solver ){
    std::cout << "\tRunning test case 'esqc_example_solver'" << std::endl;

    // We can find the following example at (http://www.esqc.org/static/lectures/Malmqvist_2B.pdf)
    // Build up the example matrix
    Eigen::MatrixXd A = Eigen::MatrixXd::Constant(5, 5, 0.1);
    A(0,0) = 1.0;
    A(1,1) = 2.0;
    A(2,2) = 3.0;
    A(3,3) = 3.0;
    A(4,4) = 3.0;

    // The solutions to the problem are given in the example
    Eigen::VectorXd evals_ex (1);
    evals_ex << 0.979;
    Eigen::MatrixXd evecs_ex (5, 1);
    evecs_ex << 0.994, -0.083, -0.042, -0.042, -0.042;


    // Solve using the Davidson diagonalization
    unsigned r = 1;
    double tol = 0.05;
    DavidsonSolver ds (A, r, tol);
    ds.solve();
    auto evals_d = ds.eigenvalues();
    auto evecs_d = ds.eigenvectors();


    // Test if the example solutions are equal to the Davidson solutions
    BOOST_CHECK(are_equal_evals(evals_d, evals_ex, tol));
    BOOST_CHECK(are_equal_evecs(evecs_d, evecs_ex, tol));
}


BOOST_AUTO_TEST_CASE( bigger_example_three ) {
    std::cout << "\tRunning test case 'bigger_example_three'" << std::endl;

    // Let's make a diagonally dominant, but random matrix
    int dim = 15;
    Eigen::MatrixXd A = Eigen::MatrixXd::Constant(dim, dim, 0.1);

    // First, we put i+1 on the diagonal
    for (int i = 0; i < dim; i++) {
        A(i, i) = i + 1;
    }


    // Now it's time to solve the eigensystem for S
    // Afterwards, select the 3 lowest eigenpairs
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes (A);
    Eigen::VectorXd all_evals = saes.eigenvalues();
    Eigen::VectorXd evals_ex = all_evals.head(3);    // can't do this in the previous line; eigenvalues is const

    Eigen::MatrixXd all_evecs = saes.eigenvectors();
    Eigen::MatrixXd evecs_ex = all_evecs.topLeftCorner(dim, 3);


    // Now we have to test if my Davidson solver gives the same results
    unsigned r = 3;
    double tol = 0.0001;
    DavidsonSolver ds (A, r, tol);
    ds.solve();

    Eigen::VectorXd evals_d = ds.eigenvalues();
    Eigen::MatrixXd evecs_d = ds.eigenvectors();


    BOOST_CHECK(are_equal_evals(evals_d, evals_ex, tol));
    BOOST_CHECK(are_equal_evecs(evecs_d, evecs_ex, tol));
}


BOOST_AUTO_TEST_CASE( liu_example ){
    std::cout << "\tRunning test case 'liu_example'" << std::endl;

    // Let's prepare the Liu example (liu1978)
    unsigned N = 50;
    Eigen::MatrixXd A = Eigen::MatrixXd::Ones(N, N);
    for (unsigned i = 0; i < N; i++) {
        if (i < 5) {
            A(i, i) = 1 + 0.1 * i;
        } else {
            A(i, i) = 2 * (i + 1) - 1;
        }
    }

    // Input the solutions
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes (A);
    Eigen::VectorXd all_evals = saes.eigenvalues();
    Eigen::VectorXd evals_ex = all_evals.head(4);

    Eigen::MatrixXd all_evecs = saes.eigenvectors();
    Eigen::MatrixXd evecs_ex = all_evecs.topLeftCorner(N, 4);


    // Solve using the Davidson diagonalization
    unsigned r = 4;
    double tol = 0.05;
    DavidsonSolver ds (A, r, tol);
    ds.solve();
    auto evals_d = ds.eigenvalues();
    auto evecs_d = ds.eigenvectors();


    // Test if the example solutions are equal to the Davidson solutions
    BOOST_CHECK(are_equal_evals(evals_d, evals_ex, tol));
    BOOST_CHECK(are_equal_evecs(evecs_d, evecs_ex, tol));
}


BOOST_AUTO_TEST_CASE( liu_big ){
    std::cout << "\tRunning test case 'liu_big'" << std::endl;

    // Let's prepare the Liu example (liu1978)
    unsigned N = 1000;
    Eigen::MatrixXd A = Eigen::MatrixXd::Ones(N, N);
    for (unsigned i = 0; i < N; i++) {
        if (i < 5) {
            A(i, i) = 1 + 0.1 * i;
        } else {
            A(i, i) = 2 * (i + 1) - 1;
        }
    }

    // Input the solutions
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes (A);
    Eigen::VectorXd all_evals = saes.eigenvalues();
    Eigen::VectorXd evals_ex = all_evals.head(4);

    Eigen::MatrixXd all_evecs = saes.eigenvectors();
    Eigen::MatrixXd evecs_ex = all_evecs.topLeftCorner(N, 4);


    // Solve using the Davidson diagonalization
    unsigned r = 4;
    double tol = 0.05;
    DavidsonSolver ds (A, r, tol);
    ds.solve();
    auto evals_d = ds.eigenvalues();
    auto evecs_d = ds.eigenvectors();


    // Test if the example solutions are equal to the Davidson solutions
    BOOST_CHECK(are_equal_evals(evals_d, evals_ex, tol));
    BOOST_CHECK(are_equal_evecs(evecs_d, evecs_ex, tol));
}