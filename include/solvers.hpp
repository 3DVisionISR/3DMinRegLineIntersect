#ifndef SOILVER_HPP
#define SOLVERS_HPP

#include <Eigen/Core>
#include <vector>

// header for the solver
template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver2P1L(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver1P3L(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver2Q1L(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver1L1Q1P(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver3L1Q(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver3L1Q_New(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair);


template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver1M1Q(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver2M(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver1M2L(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver1M1P(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 6,1>> getPluckerCoord(std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> lPair);

#endif