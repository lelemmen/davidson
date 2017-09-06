#define BOOST_ALL_DYN_LINK

#define BOOST_TEST_MODULE "Davidson diagonalization"

#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>  // include this to get main(), otherwise clang++ will complain
#include <Eigen/Eigenvalues>



BOOST_AUTO_TEST_CASE( davidson ) {

    // Joshua Goings has a nice Python example concerning the Davidson algorithm
    //  http://joshuagoings.com/2013/08/23/davidsons-method/

    // Let's make a diagonally dominant matrix, but random matrix
    int dim = 10;
    Eigen::MatrixXd A (dim, dim);

    //      We put i+1 on the diagonal
    for (int i = 0; i < dim; i++) {
        A(i, i) = i + 1;
    }

    //      And we add some random, but small noise
    Eigen::MatrixXd N = Eigen::MatrixXd::Random(dim, dim);
    double factor = 0.000001;
    A += factor * N;


    // Now it's time to solve the eigensystem for A, using EigenSolver
    Eigen::EigenSolver<Eigen::MatrixXd> es (A);
    std::cout << es.eigenvalues() << std::endl;
    std::cout << es.eigenvectors() << std::endl;


    // Now we have to test if my Davidson solver gives the same results
}