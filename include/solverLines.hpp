#ifndef SOLVERLINES_HPP
#define SOLVERLINES_HPP

// #include <EIGEN/CORE>
#include <vector>

// ##########################################################
// Solver for 6L. The solver is from Stewenius paper,
// and was implemented by Kneip.
// ##########################################################
template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec,4,4>> solver6L(
    std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	  std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	  std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair
  );

#endif