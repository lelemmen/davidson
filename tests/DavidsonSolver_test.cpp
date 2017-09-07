#define BOOST_ALL_DYN_LINK

#define BOOST_TEST_MODULE "Davidson diagonalization"

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>  // include this to get main(), otherwise clang++ will complain
#include <Eigen/Eigenvalues>
#include "davidson.hpp"


/*BOOST_AUTO_TEST_CASE( constructor ) {

    // DavidsonSolver can't accept a non-symmetric matrix
    int dim = 5;
    Eigen::MatrixXd X = Eigen::MatrixXd::Random(dim, dim);
    BOOST_CHECK_THROW(DavidsonSolver ds1 (X), std::invalid_argument);

    // DavidsonSolver should accept symmetric matrices
    Eigen::MatrixXd XT = X.transpose();
    Eigen::MatrixXd S = 0.5 * (X + XT);
    BOOST_CHECK_NO_THROW(DavidsonSolver ds2 (S));

}
*/

/*BOOST_AUTO_TEST_CASE( random_solver ) {

    // Joshua Goings has a nice Python example concerning the Davidson algorithm
    // http://joshuagoings.com/2013/08/23/davidsons-method/

    // Let's make a diagonally dominant matrix, but random matrix
    int dim = 10;
    Eigen::MatrixXd A (dim, dim);

    // We put i+1 on the diagonal
    for (int i = 0; i < dim; i++) {
        A(i, i) = i + 1;
    }

    // We add some random, but small noise
    Eigen::MatrixXd N = Eigen::MatrixXd::Random(dim, dim);
    double factor = 0.01;
    A += factor * N;

    // And finally, we symmetrize
    Eigen::MatrixXd AT = A.transpose();    // Aliasing issue, don't put [auto AT=A.transpose()]
    A = 0.5 * (A + AT);


    // Now it's time to solve the eigensystem for A, using EigenSolver
    Eigen::EigenSolver<Eigen::MatrixXd> es (A);
    // std::cout << "A eigenvalues" << std::endl << es.eigenvalues() << std::endl << std::endl;
    // std::cout << "A eigenvectors" << std::endl << es.eigenvectors() << std::endl << std::endl;


    // Now we have to test if my Davidson solver gives the same results
    // DavidsonSolver ds (A);
    // ds.solve();

}*/


BOOST_AUTO_TEST_CASE( esqc_example_solver ){

    // Build up the example matrix
    Eigen::MatrixXd A = Eigen::MatrixXd::Constant(5, 5, 0.1);
    A(0,0) = 1.0;
    A(1,1) = 2.0;
    A(2,2) = 3.0;
    A(3,3) = 3.0;
    A(4,4) = 3.0;

    // Solve using SelfAdjointEigenSolver
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes (A);
    std::cout << "eigenvalues" << std::endl << saes.eigenvalues() << std::endl << std::endl;
    std::cout << "eigenvectors" << std::endl << saes.eigenvectors() << std::endl << std::endl;

    // Solve using the Davidson diagonalization
    DavidsonSolver ds (A);
    ds.solve();
}