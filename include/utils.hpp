#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Core>
#include <vector>

// get a GT transformation
template <typename floatPrec>
Eigen::Matrix<floatPrec, 4, 4> getGTTransformations(bool verbose = false);

// Get the anti-symetric matrix that linearizes the cross prodouct
template <typename floatPrec>
Eigen::Matrix<floatPrec, 3, 3> getSkew(Eigen::Matrix<floatPrec, 3, 1> v);

// compute the roots of a four degree polynomial equation
template <typename floatPrec>
std::vector<floatPrec> o4_roots(const Eigen::Matrix<floatPrec, 5, 1> &p);

template <typename floatPrec>
bool is_nan (floatPrec d) { return std::isnan(d); }

#endif