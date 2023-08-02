#include <iostream>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>
#include <unsupported/Eigen/Polynomials>

#include <vector>
#include <complex>
#include <solvers.hpp>
#include <utils.hpp>

typedef Eigen::Matrix<float, 4, 4> Matrix4f;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;
typedef Eigen::Matrix<double, 4, 4> Matrix4d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

using namespace Eigen;
using namespace std;

// #####################################################################################################
// prototypes of the predefined transformation functions
// Predefined transformation to the 2P 1L case
template <typename floatPrec>
vector<Matrix<floatPrec, 4, 4>> getPredefinedTransformations2Plane1Line(Matrix<floatPrec, 4, 1> Pi1, Matrix<floatPrec, 4, 1> Pi2,Matrix<floatPrec, 4, 1> Nu1, Matrix<floatPrec, 4, 1> Nu2,bool verbose = false);
//
template <typename floatPrec>
vector<Matrix<floatPrec, 4, 4>> getPredefinedTransformations1Plane3Line(Matrix<floatPrec, 4, 1> Pi, Matrix<floatPrec, 4, 1> Nu, bool verbose = false);
//
template <typename floatPrec>
vector<Matrix<floatPrec, 4,4>> getPredefinedTransformations2Points1Line(Matrix<floatPrec, 4, 1> P1, Matrix<floatPrec, 4, 1> P2, Matrix<floatPrec, 4, 1> P1_B, Matrix<floatPrec, 4, 1> P2_B);
//
template <typename floatPrec>
vector<Matrix<floatPrec, 4,4>> getPredefinedTransformations1L1Q1P(Matrix<floatPrec, 4, 1> P1, Matrix<floatPrec, 4, 1> P1_B, Matrix<floatPrec, 4, 1> Pi, Matrix<floatPrec, 4, 1> Pi_B);
//
template <typename floatPrec>
vector<Matrix<floatPrec, 4,4>> getPredefinedTransformations3L1Q(Matrix<floatPrec, 4,1> P1, Matrix<floatPrec, 4,1> P1_B);
//
template <typename floatPrec>
vector<Matrix<floatPrec, 4,4>> getPredefinedTransformations1M1Q(Matrix<floatPrec, 6,1> l1, Matrix<floatPrec, 6,1> l2);
//
template <typename floatPrec>
vector<Matrix<floatPrec, Dynamic, 1>> genposeandscale_solvecoeffs(Matrix<floatPrec, Dynamic, Dynamic> B);
template <typename floatPrec>
void removeColumn(Matrix<complex<floatPrec>, Dynamic, Dynamic> &matrix, int colToRemove);
// debug
IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");


// #####################################################################################################
// Instantiate functions for the supported template type parameters
template vector<Matrix<float, 4, 4>> getPredefinedTransformations2Plane1Line<float>(Matrix<float, 4, 1> Pi1, Matrix<float, 4, 1> Pi2, Matrix<float, 4, 1> Nu1, Matrix<float, 4, 1> Nu2, bool verbose);
template vector<Matrix<double, 4, 4>> getPredefinedTransformations2Plane1Line<double>(Matrix<double, 4, 1> Pi1, Matrix<double, 4, 1> Pi2, Matrix<double, 4, 1> Nu1, Matrix<double, 4, 1> Nu2, bool verbose);
//
template vector<Matrix<float, 4, 4>> getPredefinedTransformations1Plane3Line<float>(Matrix<float, 4, 1> Pi, Matrix<float, 4, 1> Nu, bool verbose);
template vector<Matrix<double, 4, 4>> getPredefinedTransformations1Plane3Line<double>(Matrix<double, 4, 1> Pi, Matrix<double, 4, 1> Nu, bool verbose);
//
template vector<Matrix<float, 4,4>> getPredefinedTransformations2Points1Line<float>(Matrix<float, 4, 1> P1, Matrix<float, 4, 1> P2, Matrix<float, 4, 1> P1_B, Matrix<float, 4, 1> P2_B);
template vector<Matrix<double, 4,4>> getPredefinedTransformations2Points1Line<double>(Matrix<double, 4, 1> P1, Matrix<double, 4, 1> P2, Matrix<double, 4, 1> P1_B, Matrix<double, 4, 1> P2_B);
//
template vector<Matrix<float, 4,4>> getPredefinedTransformations1L1Q1P<float>(Matrix<float, 4, 1> P1, Matrix<float, 4, 1> P1_B, Matrix<float, 4, 1> Pi, Matrix<float, 4, 1> Pi_B);
template vector<Matrix<double, 4,4>> getPredefinedTransformations1L1Q1P<double>(Matrix<double, 4, 1> P1, Matrix<double, 4, 1> P1_B, Matrix<double, 4, 1> Pi, Matrix<double, 4, 1> Pi_B);
//
template vector<Matrix<float, 4,4>> getPredefinedTransformations3L1Q<float>(Matrix<float, 4,1> P1, Matrix<float, 4,1> P1_B);
template vector<Matrix<double, 4,4>> getPredefinedTransformations3L1Q<double>(Matrix<double, 4,1> P1, Matrix<double, 4,1> P1_B);
template vector<Matrix<float, Dynamic, 1>> genposeandscale_solvecoeffs<float>(Matrix<float, Dynamic, Dynamic> B);
template vector<Matrix<double, Dynamic, 1>> genposeandscale_solvecoeffs<double>(Matrix<double, Dynamic, Dynamic> B);
template void removeColumn<float>(Matrix<complex<float>, Dynamic, Dynamic> &matrix, int colToRemove);
template void removeColumn<double>(Matrix<complex<double>, Dynamic, Dynamic> &matrix, int colToRemove);

// #######################################################################################################
// SOLVERS
// Solver for the case of 2 point correspondences and 1 line intersection
template vector<Matrix<float, 4, 4>> solver2Q1L<float>(
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lCPair);
template vector<Matrix<double, 4, 4>> solver2Q1L<double>(
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lCPair);

template <typename floatPrec>
vector<Matrix<floatPrec, 4, 4>> solver2Q1L(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair)
{
	// check input vectors size
	if(ptPair.size() < 2 || lPair.size() < 1){
		std::cerr << "Solver 2Q 1L requires at least 2 point and 1 line pairs!" << std::endl;
		exit(-1);
	}

	// parse inputs
	Eigen::Matrix<floatPrec,4,1> P1,P1_B,P2,P2_B,Q1,Q2,R1_B,R2_B;
	P1 = ptPair[0].first; P1_B = ptPair[0].second;
	P2 = ptPair[1].first; P2_B = ptPair[1].second;

	Q1 = lPair[0].first.first; Q2 = lPair[0].first.second;
	R1_B = lPair[0].second.first; R2_B = lPair[0].second.second;

	// initialize the transformation matrix
	vector<Matrix<floatPrec, 4,4>> out;

	Matrix<floatPrec, 3,1> tP1, tP2, tQ1, tQ2;
	tP1 = Q1.head(3) / P1(3);
	tP2 = Q2.head(3) / P2(3);
	tQ1 = R1_B.head(3) / Q1(3);
	tQ2 = R2_B.head(3) / Q2(3);

	// computing the plucker coordinates of the lines
	// starting from line 1
	Matrix<floatPrec, 3,1> D1 = tP2 - tP1;
	Matrix<floatPrec, 6,1> L1;
	L1.head(3) = D1;
	L1.tail(3) = tP2.cross(tP1);
	L1 /= L1.head(3).norm();
	// starting from line 2
	Matrix<floatPrec, 3,1> D2 = tQ2 - tQ1;
	Matrix<floatPrec, 6,1> L2;
	L2.head(3) = D2;
	L2.tail(3) = tQ2.cross(tQ1);
	L2 /= L2.head(3).norm();

	// compute predefined transformations
	vector<Matrix<floatPrec, 4,4>> predTrans = getPredefinedTransformations2Points1Line<floatPrec>(P1, P2, P1_B, P2_B);
	Matrix<floatPrec, 4,4> TV = predTrans.back();
	predTrans.pop_back();
	Matrix<floatPrec, 4,4> TU = predTrans.back();
	predTrans.pop_back();

	// transform lines to predefined frames
	Matrix<floatPrec, 6,6> transLineU, transLineV;
	transLineU.setZero();
	transLineV.setZero();
	transLineU.topLeftCorner(3, 3) = TU.topLeftCorner(3, 3);
	transLineU.bottomRightCorner(3, 3) = TU.topLeftCorner(3, 3);
	transLineU.bottomLeftCorner(3, 3) = -1 * getSkew<floatPrec>(TU.col(3).head(3)) * TU.topLeftCorner(3, 3);
	transLineV.topLeftCorner(3, 3) = TV.topLeftCorner(3, 3);
	transLineV.bottomRightCorner(3, 3) = TV.topLeftCorner(3, 3);
	transLineV.bottomLeftCorner(3, 3) = -1 * getSkew<floatPrec>(TV.col(3).head(3)) * TV.topLeftCorner(3, 3);

	Matrix<floatPrec, 6,1> L1_2, L2_2;
	L1_2 = transLineU * L1;
	L2_2 = transLineV * L2;

	// compute rotation around Z axis
	// initialize two transformation solutions
	Matrix<floatPrec, 4,4> TF1, TF2;
	TF1.setIdentity(4,4);
	TF2.setIdentity(4,4);

	// compute polynomial coefficients
	floatPrec l11, l12, l13, l14, l15, l16, l21, l22, l23, l24, l25, l26;
	l11 = L1_2[0];
	l12 = L1_2[1];
	l13 = L1_2[2];
	l14 = L1_2[3];
	l15 = L1_2[4];
	l16 = L1_2[5];
	l21 = L2_2[0];
	l22 = L2_2[1];
	l23 = L2_2[2];
	l24 = L2_2[3];
	l25 = L2_2[4];
	l26 = L2_2[5];
	floatPrec a, b, c;
	a = l13 * l26 - l14 * l21 - l12 * l25 - l15 * l22 - l11 * l24 + l16 * l23;
	b = 2 * l11 * l25 - 2 * l12 * l24 + 2 * l14 * l22 - 2 * l15 * l21;
	c = l11 * l24 + l12 * l25 + l13 * l26 + l14 * l21 + l15 * l22 + l16 * l23;

	// apply quadratic formula
	floatPrec s1, s2;
	s1 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
	s2 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);

	if(!isnan(s1)){
		TF1 << (1 - s1 * s1) / (1 + s1 * s1), -2 * s1 / (1 + s1 * s1), 0, 0,
			2 * s1 / (1 + s1 * s1), (1 - s1 * s1) / (1 + s1 * s1), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;
		out.push_back(TV.inverse() * TF1 * TU);
		TF2 << (1 - s2 * s2) / (1 + s2 * s2), -2 * s2 / (1 + s2 * s2), 0, 0,
			2 * s2 / (1 + s2 * s2), (1 - s2 * s2) / (1 + s2 * s2), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;
		out.push_back(TV.inverse() * TF2 * TU);
	}

	return out;
}

// Solver for the case of 1 line, 1 point, and 1 plane
template vector<Matrix<float, 4, 4>> solver1L1Q1P<float>(
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lCPair);
template vector<Matrix<double, 4, 4>> solver1L1Q1P<double>(
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lCPair);

template <typename floatPrec>
vector<Matrix<floatPrec, 4, 4>> solver1L1Q1P(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair)
{
	// check input vectors size
	if(ptPair.size() < 1 || lPair.size() < 1 || plPair.size() < 1){
		std::cerr << "Solver 1L1Q1P requires at least 1 point, 1 plane, and 1 line pair!" << std::endl;
		exit(-1);
	}

	// parse inputs
	Eigen::Matrix<floatPrec,4,1> P1,P1_B,Pi,Pi_B,Q1,Q2,R1_B,R2_B;
	P1 = ptPair[0].first; P1_B = ptPair[0].second;
	Pi = plPair[0].first; Pi_B = plPair[0].second;

	Q1 = lPair[0].first.first; Q2 = lPair[0].first.second;
	R1_B = lPair[0].second.first; R2_B = lPair[0].second.second;

	// initialize the transformation matrix
	vector<Matrix<floatPrec, 4,4>> out;

	Matrix<floatPrec, 3,1> tP1, tP2, tQ1, tQ2;
	tP1 = Q1.head(3) / Q1(3);
	tP2 = Q2.head(3) / Q2(3);
	tQ1 = R1_B.head(3) / R1_B(3);
	tQ2 = R2_B.head(3) / R2_B(3);

	// computing the plucker coordinates of the lines
	// starting from line 1
	Matrix<floatPrec, 3,1> D1 = tP2 - tP1;
	Matrix<floatPrec, 6,1> L1;
	L1.head(3) = D1;
	L1.tail(3) = tP2.cross(tP1);
	L1 /= L1.head(3).norm();
	// starting from line 2
	Matrix<floatPrec, 3,1> D2 = tQ2 - tQ1;
	Matrix<floatPrec, 6,1> L2;
	L2.head(3) = D2;
	L2.tail(3) = tQ2.cross(tQ1);
	L2 /= L2.head(3).norm();

	// compute predefined transformations
	vector<Matrix<floatPrec, 4,4>> predTrans = getPredefinedTransformations1L1Q1P<floatPrec>(P1, P1_B, Pi, Pi_B);

	Matrix<floatPrec, 4,4> TV = predTrans.back();
	predTrans.pop_back();
	Matrix<floatPrec, 4,4> TU = predTrans.back();
	predTrans.pop_back();

	// transform lines to predefined frames
	Matrix<floatPrec, 6,6> transLineU, transLineV;
	transLineU.setZero();
	transLineV.setZero();
	transLineU.topLeftCorner(3, 3) = TU.topLeftCorner(3, 3);
	transLineU.bottomRightCorner(3, 3) = TU.topLeftCorner(3, 3);
	transLineU.bottomLeftCorner(3, 3) = -1 * getSkew<floatPrec>(TU.col(3).head(3)) * TU.topLeftCorner(3, 3);
	transLineV.topLeftCorner(3, 3) = TV.topLeftCorner(3, 3);
	transLineV.bottomRightCorner(3, 3) = TV.topLeftCorner(3, 3);
	transLineV.bottomLeftCorner(3, 3) = -1 * getSkew<floatPrec>(TV.col(3).head(3)) * TV.topLeftCorner(3, 3);

	Matrix<floatPrec, 6,1> L1_2, L2_2;
	L1_2 = transLineU * L1;
	L2_2 = transLineV * L2;

	// compute rotation around Z axis
	// initialize two transformation solutions
	Matrix<floatPrec, 4,4> TF1, TF2;
	TF1.setIdentity(4,4);
	TF2.setIdentity(4,4);

	// compute polynomial coefficients
	floatPrec l11, l12, l13, l14, l15, l16, l21, l22, l23, l24, l25, l26;
	l11 = L1_2[0];
	l12 = L1_2[1];
	l13 = L1_2[2];
	l14 = L1_2[3];
	l15 = L1_2[4];
	l16 = L1_2[5];
	l21 = L2_2[0];
	l22 = L2_2[1];
	l23 = L2_2[2];
	l24 = L2_2[3];
	l25 = L2_2[4];
	l26 = L2_2[5];
	floatPrec a, b, c;
	a = l13 * l26 - l14 * l21 - l12 * l25 - l15 * l22 - l11 * l24 + l16 * l23;
	b = 2 * l11 * l25 - 2 * l12 * l24 + 2 * l14 * l22 - 2 * l15 * l21;
	c = l11 * l24 + l12 * l25 + l13 * l26 + l14 * l21 + l15 * l22 + l16 * l23;

	// apply quadratic formula
	floatPrec s1, s2;
	s1 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
	s2 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);

	if(!isnan(s1)){
		TF1 << (1 - s1 * s1) / (1 + s1 * s1), -2 * s1 / (1 + s1 * s1), 0, 0,
			2 * s1 / (1 + s1 * s1), (1 - s1 * s1) / (1 + s1 * s1), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;
		out.push_back(TV.inverse() * TF1 * TU);
		TF2 << (1 - s2 * s2) / (1 + s2 * s2), -2 * s2 / (1 + s2 * s2), 0, 0,
			2 * s2 / (1 + s2 * s2), (1 - s2 * s2) / (1 + s2 * s2), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;
		out.push_back(TV.inverse() * TF2 * TU);


	}
	return out;
}


// Solver for the case of 3 lines, and 1 point
template vector<Matrix<float, 4, 4>> solver3L1Q<float>(
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lCPair);
template vector<Matrix<double, 4, 4>> solver3L1Q<double>(
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver3L1Q(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair)
{
	// check input vectors size
	if(ptPair.size() < 1 || lPair.size() < 3){
		std::cerr << "Solver 3L1Q requires at least 1 point, and 3 line pairs!" << std::endl;
		exit(-1);
	}

	// parse inputs
	Eigen::Matrix<floatPrec, 4, 1> P1, P1_B, Q11, Q12, R11_B, R12_B, Q21, Q22, R21_B, R22_B, Q31, Q32, R31_B, R32_B;
	P1 = ptPair[0].first; P1_B = ptPair[0].second;

	Q11 = lPair[0].first.first; Q12 = lPair[0].first.second;
	R11_B = lPair[0].second.first; R12_B = lPair[0].second.second;
	Q21 = lPair[1].first.first; Q22 = lPair[1].first.second;
	R21_B = lPair[1].second.first; R22_B = lPair[1].second.second;
	Q31 = lPair[2].first.first; Q32 = lPair[2].first.second;
	R31_B = lPair[2].second.first; R32_B = lPair[2].second.second;


	// initialize the transformation matrix
	vector<Matrix<floatPrec, 4,4>> out;

	Matrix<floatPrec, 3,1> tP11, tP12, tP21, tP22, tP31, tP32, tQ11, tQ12, tQ21, tQ22, tQ31, tQ32;
	// points first lines
	tP11 = Q11.head(3) / Q11(3);
	tP12 = Q12.head(3) / Q12(3);
	tQ11 = R11_B.head(3) / R11_B(3);
	tQ12 = R12_B.head(3) / R12_B(3);
	// points second lines
	tP21 = Q21.head(3) / Q21(3);
	tP22 = Q22.head(3) / Q22(3);
	tQ21 = R21_B.head(3) / R21_B(3);
	tQ22 = R22_B.head(3) / R22_B(3);
	// points third lines
	tP31 = Q31.head(3) / Q31(3);
	tP32 = Q32.head(3) / Q32(3);
	tQ31 = R31_B.head(3) / R31_B(3);
	tQ32 = R32_B.head(3) / R32_B(3);

	// computing the plucker coordinates of the lines
	// ref A
	Matrix<floatPrec, 3,1> D1 = tP12 - tP11;
	Matrix<floatPrec, 6,1> L1;
	L1.head(3) = D1;
	L1.tail(3) = tP12.cross(tP11);
	L1 /= L1.head(3).norm();
	Matrix<floatPrec, 3,1> D2 = tP22 - tP21;
	Matrix<floatPrec, 6,1> L2;
	L2.head(3) = D2;
	L2.tail(3) = tP22.cross(tP21);
	L2 /= L2.head(3).norm();
	Matrix<floatPrec, 3,1> D3 = tP32 - tP31;
	Matrix<floatPrec, 6,1> L3;
	L3.head(3) = D3;
	L3.tail(3) = tP32.cross(tP31);
	L3 /= L3.head(3).norm();
	// ref B
	D1 = tQ12 - tQ11;
	Matrix<floatPrec, 6,1> M1;
	M1.head(3) = D1;
	M1.tail(3) = tQ12.cross(tQ11);
	M1 /= M1.head(3).norm();
	D2 = tQ22 - tQ21;
	Matrix<floatPrec, 6,1> M2;
	M2.head(3) = D2;
	M2.tail(3) = tQ22.cross(tQ21);
	M2 /= M2.head(3).norm();
	D3 = tQ32 - tQ31;
	Matrix<floatPrec, 6,1> M3;
	M3.head(3) = D3;
	M3.tail(3) = tQ32.cross(tQ31);
	M3 /= M3.head(3).norm();

	// compute predefined transformations
	vector<Matrix<floatPrec, 4,4>> predTrans = getPredefinedTransformations3L1Q(P1, P1_B);
	Matrix<floatPrec, 4,4> TV = predTrans.back();
	predTrans.pop_back();
	Matrix<floatPrec, 4,4> TU = predTrans.back();
	predTrans.pop_back();

	// transform lines to predefined frames
	Matrix<floatPrec, 6,6> transLineU, transLineV;
	transLineU.setZero();
	transLineV.setZero();
	transLineU.topLeftCorner(3, 3) = TU.topLeftCorner(3, 3);
	transLineU.bottomRightCorner(3, 3) = TU.topLeftCorner(3, 3);
	transLineU.bottomLeftCorner(3, 3) = -1 * getSkew<floatPrec>(TU.col(3).head(3)) * TU.topLeftCorner(3, 3);
	transLineV.topLeftCorner(3, 3) = TV.topLeftCorner(3, 3);
	transLineV.bottomRightCorner(3, 3) = TV.topLeftCorner(3, 3);
	transLineV.bottomLeftCorner(3, 3) = -1 * getSkew<floatPrec>(TV.col(3).head(3)) * TV.topLeftCorner(3, 3);

	Matrix<floatPrec, 6,1> L1_1, L2_1, L3_1, M1_1, M2_1, M3_1;
	L1_1 = transLineU * L1;
	L2_1 = transLineU * L2;
	L3_1 = transLineU * L3;
	M1_1 = transLineV * M1;
	M2_1 = transLineV * M2;
	M3_1 = transLineV * M3;

	floatPrec l11, l12, l13, l14, l15, l16, l21, l22, l23, l24, l25, l26, l31, l32, l33, l34, l35, l36;
	l11 = L1_1[0];
	l12 = L1_1[1];
	l13 = L1_1[2];
	l14 = L1_1[3];
	l15 = L1_1[4];
	l16 = L1_1[5];
	l21 = L2_1[0];
	l22 = L2_1[1];
	l23 = L2_1[2];
	l24 = L2_1[3];
	l25 = L2_1[4];
	l26 = L2_1[5];
	l31 = L3_1[0];
	l32 = L3_1[1];
	l33 = L3_1[2];
	l34 = L3_1[3];
	l35 = L3_1[4];
	l36 = L3_1[5];
	floatPrec m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26, m31, m32, m33, m34, m35, m36;
	m11 = M1_1[0];
	m12 = M1_1[1];
	m13 = M1_1[2];
	m14 = M1_1[3];
	m15 = M1_1[4];
	m16 = M1_1[5];
	m21 = M2_1[0];
	m22 = M2_1[1];
	m23 = M2_1[2];
	m24 = M2_1[3];
	m25 = M2_1[4];
	m26 = M2_1[5];
	m31 = M3_1[0];
	m32 = M3_1[1];
	m33 = M3_1[2];
	m34 = M3_1[3];
	m35 = M3_1[4];
	m36 = M3_1[5];

	Matrix<floatPrec, 3,9> A;
	A << l11 * m14 + l14 * m11, l12 * m14 + l15 * m11, l13 * m14 + l16 * m11, l11 * m15 + l14 * m12, l12 * m15 + l15 * m12, l13 * m15 + l16 * m12, l11 * m16 + l14 * m13, l12 * m16 + l15 * m13, l13 * m16 + l16 * m13,
		l21 * m24 + l24 * m21, l22 * m24 + l25 * m21, l23 * m24 + l26 * m21, l21 * m25 + l24 * m22, l22 * m25 + l25 * m22, l23 * m25 + l26 * m22, l21 * m26 + l24 * m23, l22 * m26 + l25 * m23, l23 * m26 + l26 * m23,
		l31 * m34 + l34 * m31, l32 * m34 + l35 * m31, l33 * m34 + l36 * m31, l31 * m35 + l34 * m32, l32 * m35 + l35 * m32, l33 * m35 + l36 * m32, l31 * m36 + l34 * m33, l32 * m36 + l35 * m33, l33 * m36 + l36 * m33;

	BDCSVD<Matrix<floatPrec, Dynamic,Dynamic>> svd(A, ComputeFullU | ComputeFullV);
	Matrix<floatPrec, 9,6> B = svd.matrixV().topRightCorner(9, 6);

	vector<Matrix<floatPrec, Dynamic,1>> b = genposeandscale_solvecoeffs<floatPrec>(B);

	Matrix<floatPrec, Dynamic,1> b1, b2, b3, b4, b5;
	b5 = b.back();
	b.pop_back();
	b4 = b.back();
	b.pop_back();
	b3 = b.back();
	b.pop_back();
	b2 = b.back();
	b.pop_back();
	b1 = b.back();
	b.pop_back();

	Matrix<floatPrec, 4,4> TP;
	Matrix<floatPrec, 6,1> b_sol;
	Matrix<floatPrec, Dynamic,Dynamic> P(9,1);
	floatPrec a;
	for (int i = 0; i < b1.size(); i++)
	{
		TP.setIdentity(4,4);

		b_sol << b1[i], b2[i], b3[i], b4[i], b5[i], 1;
		
		P = B * b_sol;

		P.resize(3, 3);
		a = P.row(0).norm();

		if (P.determinant() < 0)
		{
			a = -a;
		}

		P = P / a;

		TP.topLeftCorner(3, 3) = P.transpose();

		out.push_back(TV.inverse() * TP * TU);
	}

	return out;
}


// Solver for the case of 3 line intersections and 1 point quadric intersection solver
template vector<Matrix<float, 4, 4>> solver3L1Q_New<float>(
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lCPair);
template vector<Matrix<double, 4, 4>> solver3L1Q_New<double>(
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver3L1Q_New(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair)
{
	// check input vectors size
	if(ptPair.size() < 1 || lPair.size() < 3){
		std::cerr << "Solver 3L1Q requires at least 1 point, and 3 line pairs!" << std::endl;
		exit(-1);
	}

	// parse inputs
	Eigen::Matrix<floatPrec, 4, 1> P1, P1_B, Q11, Q12, R11_B, R12_B, Q21, Q22, R21_B, R22_B, Q31, Q32, R31_B, R32_B;
	P1 = ptPair[0].first; P1_B = ptPair[0].second;

	Q11 = lPair[0].first.first; Q12 = lPair[0].first.second;
	R11_B = lPair[0].second.first; R12_B = lPair[0].second.second;
	Q21 = lPair[1].first.first; Q22 = lPair[1].first.second;
	R21_B = lPair[1].second.first; R22_B = lPair[1].second.second;
	Q31 = lPair[2].first.first; Q32 = lPair[2].first.second;
	R31_B = lPair[2].second.first; R32_B = lPair[2].second.second;


	// initialize the transformation matrix
	vector<Matrix<floatPrec, 4,4>> out;

	Matrix<floatPrec, 3,1> tP11, tP12, tP21, tP22, tP31, tP32, tQ11, tQ12, tQ21, tQ22, tQ31, tQ32;
	// points first lines
	tP11 = Q11.head(3) / Q11(3);
	tP12 = Q12.head(3) / Q12(3);
	tQ11 = R11_B.head(3) / R11_B(3);
	tQ12 = R12_B.head(3) / R12_B(3);
	// points second lines
	tP21 = Q21.head(3) / Q21(3);
	tP22 = Q22.head(3) / Q22(3);
	tQ21 = R21_B.head(3) / R21_B(3);
	tQ22 = R22_B.head(3) / R22_B(3);
	// points third lines
	tP31 = Q31.head(3) / Q31(3);
	tP32 = Q32.head(3) / Q32(3);
	tQ31 = R31_B.head(3) / R31_B(3);
	tQ32 = R32_B.head(3) / R32_B(3);

	// computing the plucker coordinates of the lines
	// ref A
	Matrix<floatPrec, 3,1> D1 = tP12 - tP11;
	Matrix<floatPrec, 6,1> L1;
	L1.head(3) = D1;
	L1.tail(3) = tP12.cross(tP11);
	L1 /= L1.head(3).norm();
	Matrix<floatPrec, 3,1> D2 = tP22 - tP21;
	Matrix<floatPrec, 6,1> L2;
	L2.head(3) = D2;
	L2.tail(3) = tP22.cross(tP21);
	L2 /= L2.head(3).norm();
	Matrix<floatPrec, 3,1> D3 = tP32 - tP31;
	Matrix<floatPrec, 6,1> L3;
	L3.head(3) = D3;
	L3.tail(3) = tP32.cross(tP31);
	L3 /= L3.head(3).norm();
	// ref B
	D1 = tQ12 - tQ11;
	Matrix<floatPrec, 6,1> M1;
	M1.head(3) = D1;
	M1.tail(3) = tQ12.cross(tQ11);
	M1 /= M1.head(3).norm();
	D2 = tQ22 - tQ21;
	Matrix<floatPrec, 6,1> M2;
	M2.head(3) = D2;
	M2.tail(3) = tQ22.cross(tQ21);
	M2 /= M2.head(3).norm();
	D3 = tQ32 - tQ31;
	Matrix<floatPrec, 6,1> M3;
	M3.head(3) = D3;
	M3.tail(3) = tQ32.cross(tQ31);
	M3 /= M3.head(3).norm();

	// compute predefined transformations
	vector<Matrix<floatPrec, 4,4>> predTrans = getPredefinedTransformations3L1Q(P1, P1_B);
	Matrix<floatPrec, 4,4> TV = predTrans.back();
	predTrans.pop_back();
	Matrix<floatPrec, 4,4> TU = predTrans.back();
	predTrans.pop_back();

	// transform lines to predefined frames
	Matrix<floatPrec, 6,6> transLineU, transLineV;
	transLineU.setZero();
	transLineV.setZero();
	transLineU.topLeftCorner(3, 3) = TU.topLeftCorner(3, 3);
	transLineU.bottomRightCorner(3, 3) = TU.topLeftCorner(3, 3);
	transLineU.bottomLeftCorner(3, 3) = -1 * getSkew<floatPrec>(TU.col(3).head(3)) * TU.topLeftCorner(3, 3);
	transLineV.topLeftCorner(3, 3) = TV.topLeftCorner(3, 3);
	transLineV.bottomRightCorner(3, 3) = TV.topLeftCorner(3, 3);
	transLineV.bottomLeftCorner(3, 3) = -1 * getSkew<floatPrec>(TV.col(3).head(3)) * TV.topLeftCorner(3, 3);

	Matrix<floatPrec, 6,1> L1_1, L2_1, L3_1, M1_1, M2_1, M3_1;
	L1_1 = transLineU * L1;
	L2_1 = transLineU * L2;
	L3_1 = transLineU * L3;
	M1_1 = transLineV * M1;
	M2_1 = transLineV * M2;
	M3_1 = transLineV * M3;

	floatPrec l11, l12, l13, l14, l15, l16, l21, l22, l23, l24, l25, l26, l31, l32, l33, l34, l35, l36;
	floatPrec m11, m12, m13, m14, m15, m16, m21, m22, m23, m24, m25, m26, m31, m32, m33, m34, m35, m36;

	l11 = L1_1[0];
	l12 = L1_1[1];
	l13 = L1_1[2];
	l14 = L1_1[3];
	l15 = L1_1[4];
	l16 = L1_1[5];
	l21 = L2_1[0];
	l22 = L2_1[1];
	l23 = L2_1[2];
	l24 = L2_1[3];
	l25 = L2_1[4];
	l26 = L2_1[5];
	l31 = L3_1[0];
	l32 = L3_1[1];
	l33 = L3_1[2];
	l34 = L3_1[3];
	l35 = L3_1[4];
	l36 = L3_1[5];
	
	m11 = M1_1[0];
	m12 = M1_1[1];
	m13 = M1_1[2];
	m14 = M1_1[3];
	m15 = M1_1[4];
	m16 = M1_1[5];
	m21 = M2_1[0];
	m22 = M2_1[1];
	m23 = M2_1[2];
	m24 = M2_1[3];
	m25 = M2_1[4];
	m26 = M2_1[5];
	m31 = M3_1[0];
	m32 = M3_1[1];
	m33 = M3_1[2];
	m34 = M3_1[3];
	m35 = M3_1[4];
	m36 = M3_1[5];

	floatPrec k1_0, k1_1, k1_2, k1_3, k1_4, k1_5, k1_6, k1_7, k1_8, k1_9;
	k1_0 = l11*m14 + l12*m15 + l13*m16 + l14*m11 + l15*m12 + l16*m13;
	k1_1 = 2*l11*m15 - 2*l12*m14 + 2*l14*m12 - 2*l15*m11;
	k1_2 = -l11*m14 - l12*m15 + l13*m16 - l14*m11 - l15*m12 + l16*m13;
	k1_3 = -2*l11*m16 + 2*l13*m14 - 2*l14*m13 + 2*l16*m11;
	k1_4 = 2*l12*m16 + 2*l13*m15 + 2*l15*m13 + 2*l16*m12;
	k1_5 = -l11*m14 + l12*m15 - l13*m16 - l14*m11 + l15*m12 - l16*m13;
	k1_6 = 2*l12*m16 - 2*l13*m15 + 2*l15*m13 - 2*l16*m12;
	k1_7 = 2*l11*m16 + 2*l13*m14 + 2*l14*m13 + 2*l16*m11;
	k1_8 = 2*l11*m15 + 2*l12*m14 + 2*l14*m12 + 2*l15*m11;
	k1_9 = l11*m14 - l12*m15 - l13*m16 + l14*m11 - l15*m12 - l16*m13;

	floatPrec k2_0, k2_1, k2_2, k2_3, k2_4, k2_5, k2_6, k2_7, k2_8, k2_9;
	k2_0 = l21*m24 + l22*m25 + l23*m26 + l24*m21 + l25*m22 + l26*m23;
	k2_1 = 2*l21*m25 - 2*l22*m24 + 2*l24*m22 - 2*l25*m21;
	k2_2 = -l21*m24 - l22*m25 + l23*m26 - l24*m21 - l25*m22 + l26*m23;
	k2_3 = -2*l21*m26 + 2*l23*m24 - 2*l24*m23 + 2*l26*m21;
	k2_4 = 2*l22*m26 + 2*l23*m25 + 2*l25*m23 + 2*l26*m22;
	k2_5 = -l21*m24 + l22*m25 - l23*m26 - l24*m21 + l25*m22 - l26*m23;
	k2_6 = 2*l22*m26 - 2*l23*m25 + 2*l25*m23 - 2*l26*m22;
	k2_7 = 2*l21*m26 + 2*l23*m24 + 2*l24*m23 + 2*l26*m21;
	k2_8 = 2*l21*m25 + 2*l22*m24 + 2*l24*m22 + 2*l25*m21;
	k2_9 = l21*m24 - l22*m25 - l23*m26 + l24*m21 - l25*m22 - l26*m23;

	floatPrec k3_0, k3_1, k3_2, k3_3, k3_4, k3_5, k3_6, k3_7, k3_8, k3_9;
	k3_0 = l31*m34 + l32*m35 + l33*m36 + l34*m31 + l35*m32 + l36*m33;
	k3_1 = 2*l31*m35 - 2*l32*m34 + 2*l34*m32 - 2*l35*m31;
	k3_2 = -l31*m34 - l32*m35 + l33*m36 - l34*m31 - l35*m32 + l36*m33;
	k3_3 = -2*l31*m36 + 2*l33*m34 - 2*l34*m33 + 2*l36*m31;
	k3_4 = 2*l32*m36 + 2*l33*m35 + 2*l35*m33 + 2*l36*m32;
	k3_5 = -l31*m34 + l32*m35 - l33*m36 - l34*m31 + l35*m32 - l36*m33;
	k3_6 = 2*l32*m36 - 2*l33*m35 + 2*l35*m33 - 2*l36*m32;
	k3_7 = 2*l31*m36 + 2*l33*m34 + 2*l34*m33 + 2*l36*m31;
	k3_8 = 2*l31*m35 + 2*l32*m34 + 2*l34*m32 + 2*l35*m31;
	k3_9 = l31*m34 - l32*m35 - l33*m36 + l34*m31 - l35*m32 - l36*m33;

	floatPrec m1_0, m1_1;
	m1_0 = (k1_2*k2_3*k3_4 - k1_2*k2_4*k3_3 - k1_3*k2_2*k3_4 + k1_3*k2_4*k3_2 + k1_4*k2_2*k3_3 - k1_4*k2_3*k3_2)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);
	m1_1 = (-k1_2*k2_4*k3_8 + k1_2*k2_8*k3_4 + k1_4*k2_2*k3_8 - k1_4*k2_8*k3_2 - k1_8*k2_2*k3_4 + k1_8*k2_4*k3_2)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);

	floatPrec m2_0, m2_1;
	m2_0 = (-k1_1*k2_2*k3_4 + k1_1*k2_4*k3_2 + k1_2*k2_1*k3_4 - k1_2*k2_4*k3_1 - k1_4*k2_1*k3_2 + k1_4*k2_2*k3_1)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);
	m2_1 = (-k1_2*k2_4*k3_7 + k1_2*k2_7*k3_4 + k1_4*k2_2*k3_7 - k1_4*k2_7*k3_2 - k1_7*k2_2*k3_4 + k1_7*k2_4*k3_2)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);

	floatPrec m3_0, m3_1, m3_2;
	m3_0 = (-k1_0*k2_2*k3_4 + k1_0*k2_4*k3_2 + k1_2*k2_0*k3_4 - k1_2*k2_4*k3_0 - k1_4*k2_0*k3_2 + k1_4*k2_2*k3_0)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);
	m3_1 = (-k1_2*k2_4*k3_6 + k1_2*k2_6*k3_4 + k1_4*k2_2*k3_6 - k1_4*k2_6*k3_2 - k1_6*k2_2*k3_4 + k1_6*k2_4*k3_2)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);
	m3_2 = (-k1_2*k2_4*k3_9 + k1_2*k2_9*k3_4 + k1_4*k2_2*k3_9 - k1_4*k2_9*k3_2 - k1_9*k2_2*k3_4 + k1_9*k2_4*k3_2)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);

	floatPrec m4_0, m4_1;
	m4_0 = (-k1_3*k2_4*k3_5 + k1_3*k2_5*k3_4 + k1_4*k2_3*k3_5 - k1_4*k2_5*k3_3 - k1_5*k2_3*k3_4 + k1_5*k2_4*k3_3)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);
	m4_1 = (-k1_4*k2_5*k3_8 + k1_4*k2_8*k3_5 + k1_5*k2_4*k3_8 - k1_5*k2_8*k3_4 - k1_8*k2_4*k3_5 + k1_8*k2_5*k3_4)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);

	floatPrec m5_0, m5_1;
	m5_0 = (-k1_1*k2_4*k3_5 + k1_1*k2_5*k3_4 + k1_4*k2_1*k3_5 - k1_4*k2_5*k3_1 - k1_5*k2_1*k3_4 + k1_5*k2_4*k3_1)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);
	m5_1 = (-k1_4*k2_5*k3_7 + k1_4*k2_7*k3_5 + k1_5*k2_4*k3_7 - k1_5*k2_7*k3_4 - k1_7*k2_4*k3_5 + k1_7*k2_5*k3_4)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);

	floatPrec m6_0, m6_1, m6_2;
	m6_0 = (-k1_0*k2_4*k3_5 + k1_0*k2_5*k3_4 + k1_4*k2_0*k3_5 - k1_4*k2_5*k3_0 - k1_5*k2_0*k3_4 + k1_5*k2_4*k3_0)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);
	m6_1 = (-k1_4*k2_5*k3_6 + k1_4*k2_6*k3_5 + k1_5*k2_4*k3_6 - k1_5*k2_6*k3_4 - k1_6*k2_4*k3_5 + k1_6*k2_5*k3_4)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);
	m6_2 = (-k1_4*k2_5*k3_9 + k1_4*k2_9*k3_5 + k1_5*k2_4*k3_9 - k1_5*k2_9*k3_4 - k1_9*k2_4*k3_5 + k1_9*k2_5*k3_4)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);

	floatPrec m7_0, m7_1;
	m7_0 = (-k1_2*k2_3*k3_5 + k1_2*k2_5*k3_3 + k1_3*k2_2*k3_5 - k1_3*k2_5*k3_2 - k1_5*k2_2*k3_3 + k1_5*k2_3*k3_2)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);
	m7_1 = (k1_2*k2_5*k3_8 - k1_2*k2_8*k3_5 - k1_5*k2_2*k3_8 + k1_5*k2_8*k3_2 + k1_8*k2_2*k3_5 - k1_8*k2_5*k3_2)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);

	floatPrec m8_0, m8_1;
	m8_0 = (k1_1*k2_2*k3_5 - k1_1*k2_5*k3_2 - k1_2*k2_1*k3_5 + k1_2*k2_5*k3_1 + k1_5*k2_1*k3_2 - k1_5*k2_2*k3_1)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);
	m8_1 = (k1_2*k2_5*k3_7 - k1_2*k2_7*k3_5 - k1_5*k2_2*k3_7 + k1_5*k2_7*k3_2 + k1_7*k2_2*k3_5 - k1_7*k2_5*k3_2)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);

	floatPrec m9_0, m9_1, m9_2;
	m9_0 = (k1_0*k2_2*k3_5 - k1_0*k2_5*k3_2 - k1_2*k2_0*k3_5 + k1_2*k2_5*k3_0 + k1_5*k2_0*k3_2 - k1_5*k2_2*k3_0)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);
	m9_1 = (k1_2*k2_5*k3_6 - k1_2*k2_6*k3_5 - k1_5*k2_2*k3_6 + k1_5*k2_6*k3_2 + k1_6*k2_2*k3_5 - k1_6*k2_5*k3_2)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);
	m9_2 = (k1_2*k2_5*k3_9 - k1_2*k2_9*k3_5 - k1_5*k2_2*k3_9 + k1_5*k2_9*k3_2 + k1_9*k2_2*k3_5 - k1_9*k2_5*k3_2)/(k1_2*k2_4*k3_5 - k1_2*k2_5*k3_4 - k1_4*k2_2*k3_5 + k1_4*k2_5*k3_2 + k1_5*k2_2*k3_4 - k1_5*k2_4*k3_2);

	floatPrec n1_0, n1_1, n1_2;
	n1_0 = m2_0*m4_0 - m7_0*m8_0 - m9_0;
	n1_1 = m2_0*m4_1 + m2_1*m4_0 - m7_0*m8_1 - m7_1*m8_0 - m9_1;
	n1_2 = m2_1*m4_1 - m7_1*m8_1 - m9_2;

	floatPrec n2_0, n2_1, n2_2;
	n2_0 = m1_0*m8_0 + m2_0*m5_0 - m2_0*m7_0 + m3_0 - m8_0*m8_0;
	n2_1 = m1_0*m8_1 + m1_1*m8_0 + m2_0*m5_1 - m2_0*m7_1 + m2_1*m5_0 - m2_1*m7_0 + m3_1 - 2*m8_0*m8_1;
	n2_2 = m1_1*m8_1 + m2_1*m5_1 - m2_1*m7_1 + m3_2 - m8_1*m8_1;

	floatPrec n3_0, n3_1, n3_2, n3_3;
	n3_0 = m1_0*m9_0 + m2_0*m6_0 - m3_0*m7_0 - m8_0*m9_0;
	n3_1 = m1_0*m9_1 + m1_1*m9_0 + m2_0*m6_1 + m2_1*m6_0 - m3_0*m7_1 - m3_1*m7_0 - m8_0*m9_1 - m8_1*m9_0;
	n3_2 = m1_0*m9_2 + m1_1*m9_1 + m2_0*m6_2 + m2_1*m6_1 - m3_1*m7_1 - m3_2*m7_0 - m8_0*m9_2 - m8_1*m9_1;
	n3_3 = m1_1*m9_2 + m2_1*m6_2 - m3_2*m7_1 - m8_1*m9_2;

	floatPrec n4_0, n4_1, n4_2;
	n4_0 = -m1_0*m4_0 + m4_0*m8_0 - m5_0*m7_0 - m6_0 + m7_0*m7_0;
	n4_1 = -m1_0*m4_1 - m1_1*m4_0 + m4_0*m8_1 + m4_1*m8_0 - m5_0*m7_1 - m5_1*m7_0 - m6_1 + 2*m7_0*m7_1;
	n4_2 = -m1_1*m4_1 + m4_1*m8_1 - m5_1*m7_1 - m6_2 + m7_1*m7_1;

	floatPrec n5_0, n5_1, n5_2;
	n5_0 = -m2_0*m4_0 + m7_0*m8_0 + m9_0;
	n5_1 = -m2_0*m4_1 - m2_1*m4_0 + m7_0*m8_1 + m7_1*m8_0 + m9_1;
	n5_2 = -m2_1*m4_1 + m7_1*m8_1 + m9_2;

	floatPrec n6_0, n6_1, n6_2, n6_3;
	n6_0 = -m3_0*m4_0 - m5_0*m9_0 + m6_0*m8_0 + m7_0*m9_0;
	n6_1 = -m3_0*m4_1 - m3_1*m4_0 - m5_0*m9_1 - m5_1*m9_0 + m6_0*m8_1 + m6_1*m8_0 + m7_0*m9_1 + m7_1*m9_0;
	n6_2 = -m3_1*m4_1 - m3_2*m4_0 - m5_0*m9_2 - m5_1*m9_1 + m6_1*m8_1 + m6_2*m8_0 + m7_0*m9_2 + m7_1*m9_1;
	n6_3 = -m3_2*m4_1 - m5_1*m9_2 + m6_2*m8_1 + m7_1*m9_2;

	floatPrec n7_0, n7_1, n7_2, n7_3;
	n7_0 = -m1_0*m1_0*m4_0 - m1_0*m5_0*m7_0 - m1_0*m6_0 + m1_0*m7_0*m7_0 - m2_0*m4_0*m5_0 - m2_0*m4_0*m7_0 - m3_0*m4_0 + m4_0*m8_0*m8_0 + 2*m7_0*m7_0*m8_0 + 2*m7_0*m9_0;
	n7_1 = -m1_0*m1_0*m4_1 - 2*m1_0*m1_1*m4_0 - m1_0*m5_0*m7_1 - m1_0*m5_1*m7_0 - m1_0*m6_1 + 2*m1_0*m7_0*m7_1 - m1_1*m5_0*m7_0 - m1_1*m6_0 + m1_1*m7_0*m7_0 - m2_0*m4_0*m5_1 - m2_0*m4_0*m7_1 - m2_0*m4_1*m5_0 - m2_0*m4_1*m7_0 - m2_1*m4_0*m5_0 - m2_1*m4_0*m7_0 - m3_0*m4_1 - m3_1*m4_0 + 2*m4_0*m8_0*m8_1 + m4_1*m8_0*m8_0 + 2*m7_0*m7_0*m8_1 + 4*m7_0*m7_1*m8_0 + 2*m7_0*m9_1 + 2*m7_1*m9_0;
	n7_2 = -2*m1_0*m1_1*m4_1 - m1_0*m5_1*m7_1 - m1_0*m6_2 + m1_0*m7_1*m7_1 - m1_1*m1_1*m4_0 - m1_1*m5_0*m7_1 - m1_1*m5_1*m7_0 - m1_1*m6_1 + 2*m1_1*m7_0*m7_1 - m2_0*m4_1*m5_1 - m2_0*m4_1*m7_1 - m2_1*m4_0*m5_1 - m2_1*m4_0*m7_1 - m2_1*m4_1*m5_0 - m2_1*m4_1*m7_0 - m3_1*m4_1 - m3_2*m4_0 + m4_0*m8_1*m8_1 + 2*m4_1*m8_0*m8_1 + 4*m7_0*m7_1*m8_1 + 2*m7_0*m9_2 + 2*m7_1*m7_1*m8_0 + 2*m7_1*m9_1;
	n7_3 = -m1_1*m1_1*m4_1 - m1_1*m5_1*m7_1 - m1_1*m6_2 + m1_1*m7_1*m7_1 - m2_1*m4_1*m5_1 - m2_1*m4_1*m7_1 - m3_2*m4_1 + m4_1*m8_1*m8_1 + 2*m7_1*m7_1*m8_1 + 2*m7_1*m9_2;

	floatPrec n8_0, n8_1, n8_2, n8_3;
	n8_0 = -m1_0*m2_0*m4_0 - m1_0*m5_0*m8_0 - m2_0*m4_0*m8_0 - m2_0*m5_0*m5_0 - m2_0*m6_0 + m2_0*m7_0*m7_0 - m3_0*m5_0 + m5_0*m8_0*m8_0 + 2*m7_0*m8_0*m8_0 + 2*m8_0*m9_0;
	n8_1 = -m1_0*m2_0*m4_1 - m1_0*m2_1*m4_0 - m1_0*m5_0*m8_1 - m1_0*m5_1*m8_0 - m1_1*m2_0*m4_0 - m1_1*m5_0*m8_0 - m2_0*m4_0*m8_1 - m2_0*m4_1*m8_0 - 2*m2_0*m5_0*m5_1 - m2_0*m6_1 + 2*m2_0*m7_0*m7_1 - m2_1*m4_0*m8_0 - m2_1*m5_0*m5_0 - m2_1*m6_0 + m2_1*m7_0*m7_0 - m3_0*m5_1 - m3_1*m5_0 + 2*m5_0*m8_0*m8_1 + m5_1*m8_0*m8_0 + 4*m7_0*m8_0*m8_1 + 2*m7_1*m8_0*m8_0 + 2*m8_0*m9_1 + 2*m8_1*m9_0;
	n8_2 = -m1_0*m2_1*m4_1 - m1_0*m5_1*m8_1 - m1_1*m2_0*m4_1 - m1_1*m2_1*m4_0 - m1_1*m5_0*m8_1 - m1_1*m5_1*m8_0 - m2_0*m4_1*m8_1 - m2_0*m5_1*m5_1 - m2_0*m6_2 + m2_0*m7_1*m7_1 - m2_1*m4_0*m8_1 - m2_1*m4_1*m8_0 - 2*m2_1*m5_0*m5_1 - m2_1*m6_1 + 2*m2_1*m7_0*m7_1 - m3_1*m5_1 - m3_2*m5_0 + m5_0*m8_1*m8_1 + 2*m5_1*m8_0*m8_1 + 2*m7_0*m8_1*m8_1 + 4*m7_1*m8_0*m8_1 + 2*m8_0*m9_2 + 2*m8_1*m9_1;
	n8_3 = -m1_1*m2_1*m4_1 - m1_1*m5_1*m8_1 - m2_1*m4_1*m8_1 - m2_1*m5_1*m5_1 - m2_1*m6_2 + m2_1*m7_1*m7_1 - m3_2*m5_1 + m5_1*m8_1*m8_1 + 2*m7_1*m8_1*m8_1 + 2*m8_1*m9_2;

	floatPrec n9_0, n9_1, n9_2, n9_3, n9_4;
	n9_0 = -m1_0*m3_0*m4_0 - m1_0*m5_0*m9_0 - m2_0*m4_0*m9_0 - m2_0*m5_0*m6_0 - m3_0*m6_0 + m3_0*m7_0*m7_0 + m6_0*m8_0*m8_0 + 2*m7_0*m8_0*m9_0 + m9_0*m9_0;
	n9_1 = -m1_0*m3_0*m4_1 - m1_0*m3_1*m4_0 - m1_0*m5_0*m9_1 - m1_0*m5_1*m9_0 - m1_1*m3_0*m4_0 - m1_1*m5_0*m9_0 - m2_0*m4_0*m9_1 - m2_0*m4_1*m9_0 - m2_0*m5_0*m6_1 - m2_0*m5_1*m6_0 - m2_1*m4_0*m9_0 - m2_1*m5_0*m6_0 - m3_0*m6_1 + 2*m3_0*m7_0*m7_1 - m3_1*m6_0 + m3_1*m7_0*m7_0 + 2*m6_0*m8_0*m8_1 + m6_1*m8_0*m8_0 + 2*m7_0*m8_0*m9_1 + 2*m7_0*m8_1*m9_0 + 2*m7_1*m8_0*m9_0 + 2*m9_0*m9_1;
	n9_2 = -m1_0*m3_1*m4_1 - m1_0*m3_2*m4_0 - m1_0*m5_0*m9_2 - m1_0*m5_1*m9_1 - m1_1*m3_0*m4_1 - m1_1*m3_1*m4_0 - m1_1*m5_0*m9_1 - m1_1*m5_1*m9_0 - m2_0*m4_0*m9_2 - m2_0*m4_1*m9_1 - m2_0*m5_0*m6_2 - m2_0*m5_1*m6_1 - m2_1*m4_0*m9_1 - m2_1*m4_1*m9_0 - m2_1*m5_0*m6_1 - m2_1*m5_1*m6_0 - m3_0*m6_2 + m3_0*m7_1*m7_1 - m3_1*m6_1 + 2*m3_1*m7_0*m7_1 - m3_2*m6_0 + m3_2*m7_0*m7_0 + m6_0*m8_1*m8_1 + 2*m6_1*m8_0*m8_1 + m6_2*m8_0*m8_0 + 2*m7_0*m8_0*m9_2 + 2*m7_0*m8_1*m9_1 + 2*m7_1*m8_0*m9_1 + 2*m7_1*m8_1*m9_0 + 2*m9_0*m9_2 + m9_1*m9_1;
	n9_3 = -m1_0*m3_2*m4_1 - m1_0*m5_1*m9_2 - m1_1*m3_1*m4_1 - m1_1*m3_2*m4_0 - m1_1*m5_0*m9_2 - m1_1*m5_1*m9_1 - m2_0*m4_1*m9_2 - m2_0*m5_1*m6_2 - m2_1*m4_0*m9_2 - m2_1*m4_1*m9_1 - m2_1*m5_0*m6_2 - m2_1*m5_1*m6_1 - m3_1*m6_2 + m3_1*m7_1*m7_1 - m3_2*m6_1 + 2*m3_2*m7_0*m7_1 + m6_1*m8_1*m8_1 + 2*m6_2*m8_0*m8_1 + 2*m7_0*m8_1*m9_2 + 2*m7_1*m8_0*m9_2 + 2*m7_1*m8_1*m9_1 + 2*m9_1*m9_2;
	n9_4 = -m1_1*m3_2*m4_1 - m1_1*m5_1*m9_2 - m2_1*m4_1*m9_2 - m2_1*m5_1*m6_2 - m3_2*m6_2 + m3_2*m7_1*m7_1 + m6_2*m8_1*m8_1 + 2*m7_1*m8_1*m9_2 + m9_2*m9_2;

	floatPrec o7_0, o7_1, o7_2, o7_3, o7_4, o7_5, o7_6, o7_7, o7_8;
	o7_0 = n1_0*n5_0*n9_0 - n1_0*n6_0*n8_0 - n2_0*n4_0*n9_0 + n2_0*n6_0*n7_0 + n3_0*n4_0*n8_0 - n3_0*n5_0*n7_0;
	o7_1 = n1_0*n5_0*n9_1 + n1_0*n5_1*n9_0 - n1_0*n6_0*n8_1 - n1_0*n6_1*n8_0 + n1_1*n5_0*n9_0 - n1_1*n6_0*n8_0 - n2_0*n4_0*n9_1 - n2_0*n4_1*n9_0 + n2_0*n6_0*n7_1 + n2_0*n6_1*n7_0 - n2_1*n4_0*n9_0 + n2_1*n6_0*n7_0 + n3_0*n4_0*n8_1 + n3_0*n4_1*n8_0 - n3_0*n5_0*n7_1 - n3_0*n5_1*n7_0 + n3_1*n4_0*n8_0 - n3_1*n5_0*n7_0;
	o7_2 = n1_0*n5_0*n9_2 + n1_0*n5_1*n9_1 + n1_0*n5_2*n9_0 - n1_0*n6_0*n8_2 - n1_0*n6_1*n8_1 - n1_0*n6_2*n8_0 + n1_1*n5_0*n9_1 + n1_1*n5_1*n9_0 - n1_1*n6_0*n8_1 - n1_1*n6_1*n8_0 + n1_2*n5_0*n9_0 - n1_2*n6_0*n8_0 - n2_0*n4_0*n9_2 - n2_0*n4_1*n9_1 - n2_0*n4_2*n9_0 + n2_0*n6_0*n7_2 + n2_0*n6_1*n7_1 + n2_0*n6_2*n7_0 - n2_1*n4_0*n9_1 - n2_1*n4_1*n9_0 + n2_1*n6_0*n7_1 + n2_1*n6_1*n7_0 - n2_2*n4_0*n9_0 + n2_2*n6_0*n7_0 + n3_0*n4_0*n8_2 + n3_0*n4_1*n8_1 + n3_0*n4_2*n8_0 - n3_0*n5_0*n7_2 - n3_0*n5_1*n7_1 - n3_0*n5_2*n7_0 + n3_1*n4_0*n8_1 + n3_1*n4_1*n8_0 - n3_1*n5_0*n7_1 - n3_1*n5_1*n7_0 + n3_2*n4_0*n8_0 - n3_2*n5_0*n7_0;
	o7_3 = n1_0*n5_0*n9_3 + n1_0*n5_1*n9_2 + n1_0*n5_2*n9_1 - n1_0*n6_0*n8_3 - n1_0*n6_1*n8_2 - n1_0*n6_2*n8_1 - n1_0*n6_3*n8_0 + n1_1*n5_0*n9_2 + n1_1*n5_1*n9_1 + n1_1*n5_2*n9_0 - n1_1*n6_0*n8_2 - n1_1*n6_1*n8_1 - n1_1*n6_2*n8_0 + n1_2*n5_0*n9_1 + n1_2*n5_1*n9_0 - n1_2*n6_0*n8_1 - n1_2*n6_1*n8_0 - n2_0*n4_0*n9_3 - n2_0*n4_1*n9_2 - n2_0*n4_2*n9_1 + n2_0*n6_0*n7_3 + n2_0*n6_1*n7_2 + n2_0*n6_2*n7_1 + n2_0*n6_3*n7_0 - n2_1*n4_0*n9_2 - n2_1*n4_1*n9_1 - n2_1*n4_2*n9_0 + n2_1*n6_0*n7_2 + n2_1*n6_1*n7_1 + n2_1*n6_2*n7_0 - n2_2*n4_0*n9_1 - n2_2*n4_1*n9_0 + n2_2*n6_0*n7_1 + n2_2*n6_1*n7_0 + n3_0*n4_0*n8_3 + n3_0*n4_1*n8_2 + n3_0*n4_2*n8_1 - n3_0*n5_0*n7_3 - n3_0*n5_1*n7_2 - n3_0*n5_2*n7_1 + n3_1*n4_0*n8_2 + n3_1*n4_1*n8_1 + n3_1*n4_2*n8_0 - n3_1*n5_0*n7_2 - n3_1*n5_1*n7_1 - n3_1*n5_2*n7_0 + n3_2*n4_0*n8_1 + n3_2*n4_1*n8_0 - n3_2*n5_0*n7_1 - n3_2*n5_1*n7_0 + n3_3*n4_0*n8_0 - n3_3*n5_0*n7_0;
	o7_4 = n1_0*n5_0*n9_4 + n1_0*n5_1*n9_3 + n1_0*n5_2*n9_2 - n1_0*n6_1*n8_3 - n1_0*n6_2*n8_2 - n1_0*n6_3*n8_1 + n1_1*n5_0*n9_3 + n1_1*n5_1*n9_2 + n1_1*n5_2*n9_1 - n1_1*n6_0*n8_3 - n1_1*n6_1*n8_2 - n1_1*n6_2*n8_1 - n1_1*n6_3*n8_0 + n1_2*n5_0*n9_2 + n1_2*n5_1*n9_1 + n1_2*n5_2*n9_0 - n1_2*n6_0*n8_2 - n1_2*n6_1*n8_1 - n1_2*n6_2*n8_0 - n2_0*n4_0*n9_4 - n2_0*n4_1*n9_3 - n2_0*n4_2*n9_2 + n2_0*n6_1*n7_3 + n2_0*n6_2*n7_2 + n2_0*n6_3*n7_1 - n2_1*n4_0*n9_3 - n2_1*n4_1*n9_2 - n2_1*n4_2*n9_1 + n2_1*n6_0*n7_3 + n2_1*n6_1*n7_2 + n2_1*n6_2*n7_1 + n2_1*n6_3*n7_0 - n2_2*n4_0*n9_2 - n2_2*n4_1*n9_1 - n2_2*n4_2*n9_0 + n2_2*n6_0*n7_2 + n2_2*n6_1*n7_1 + n2_2*n6_2*n7_0 + n3_0*n4_1*n8_3 + n3_0*n4_2*n8_2 - n3_0*n5_1*n7_3 - n3_0*n5_2*n7_2 + n3_1*n4_0*n8_3 + n3_1*n4_1*n8_2 + n3_1*n4_2*n8_1 - n3_1*n5_0*n7_3 - n3_1*n5_1*n7_2 - n3_1*n5_2*n7_1 + n3_2*n4_0*n8_2 + n3_2*n4_1*n8_1 + n3_2*n4_2*n8_0 - n3_2*n5_0*n7_2 - n3_2*n5_1*n7_1 - n3_2*n5_2*n7_0 + n3_3*n4_0*n8_1 + n3_3*n4_1*n8_0 - n3_3*n5_0*n7_1 - n3_3*n5_1*n7_0;
	o7_5 = n1_0*n5_1*n9_4 + n1_0*n5_2*n9_3 - n1_0*n6_2*n8_3 - n1_0*n6_3*n8_2 + n1_1*n5_0*n9_4 + n1_1*n5_1*n9_3 + n1_1*n5_2*n9_2 - n1_1*n6_1*n8_3 - n1_1*n6_2*n8_2 - n1_1*n6_3*n8_1 + n1_2*n5_0*n9_3 + n1_2*n5_1*n9_2 + n1_2*n5_2*n9_1 - n1_2*n6_0*n8_3 - n1_2*n6_1*n8_2 - n1_2*n6_2*n8_1 - n1_2*n6_3*n8_0 - n2_0*n4_1*n9_4 - n2_0*n4_2*n9_3 + n2_0*n6_2*n7_3 + n2_0*n6_3*n7_2 - n2_1*n4_0*n9_4 - n2_1*n4_1*n9_3 - n2_1*n4_2*n9_2 + n2_1*n6_1*n7_3 + n2_1*n6_2*n7_2 + n2_1*n6_3*n7_1 - n2_2*n4_0*n9_3 - n2_2*n4_1*n9_2 - n2_2*n4_2*n9_1 + n2_2*n6_0*n7_3 + n2_2*n6_1*n7_2 + n2_2*n6_2*n7_1 + n2_2*n6_3*n7_0 + n3_0*n4_2*n8_3 - n3_0*n5_2*n7_3 + n3_1*n4_1*n8_3 + n3_1*n4_2*n8_2 - n3_1*n5_1*n7_3 - n3_1*n5_2*n7_2 + n3_2*n4_0*n8_3 + n3_2*n4_1*n8_2 + n3_2*n4_2*n8_1 - n3_2*n5_0*n7_3 - n3_2*n5_1*n7_2 - n3_2*n5_2*n7_1 + n3_3*n4_0*n8_2 + n3_3*n4_1*n8_1 + n3_3*n4_2*n8_0 - n3_3*n5_0*n7_2 - n3_3*n5_1*n7_1 - n3_3*n5_2*n7_0;
	o7_6 = n1_0*n5_2*n9_4 - n1_0*n6_3*n8_3 + n1_1*n5_1*n9_4 + n1_1*n5_2*n9_3 - n1_1*n6_2*n8_3 - n1_1*n6_3*n8_2 + n1_2*n5_0*n9_4 + n1_2*n5_1*n9_3 + n1_2*n5_2*n9_2 - n1_2*n6_1*n8_3 - n1_2*n6_2*n8_2 - n1_2*n6_3*n8_1 - n2_0*n4_2*n9_4 + n2_0*n6_3*n7_3 - n2_1*n4_1*n9_4 - n2_1*n4_2*n9_3 + n2_1*n6_2*n7_3 + n2_1*n6_3*n7_2 - n2_2*n4_0*n9_4 - n2_2*n4_1*n9_3 - n2_2*n4_2*n9_2 + n2_2*n6_1*n7_3 + n2_2*n6_2*n7_2 + n2_2*n6_3*n7_1 + n3_1*n4_2*n8_3 - n3_1*n5_2*n7_3 + n3_2*n4_1*n8_3 + n3_2*n4_2*n8_2 - n3_2*n5_1*n7_3 - n3_2*n5_2*n7_2 + n3_3*n4_0*n8_3 + n3_3*n4_1*n8_2 + n3_3*n4_2*n8_1 - n3_3*n5_0*n7_3 - n3_3*n5_1*n7_2 - n3_3*n5_2*n7_1;
	o7_7 = n1_1*n5_2*n9_4 - n1_1*n6_3*n8_3 + n1_2*n5_1*n9_4 + n1_2*n5_2*n9_3 - n1_2*n6_2*n8_3 - n1_2*n6_3*n8_2 - n2_1*n4_2*n9_4 + n2_1*n6_3*n7_3 - n2_2*n4_1*n9_4 - n2_2*n4_2*n9_3 + n2_2*n6_2*n7_3 + n2_2*n6_3*n7_2 + n3_2*n4_2*n8_3 - n3_2*n5_2*n7_3 + n3_3*n4_1*n8_3 + n3_3*n4_2*n8_2 - n3_3*n5_1*n7_3 - n3_3*n5_2*n7_2;
	o7_8 = n1_2*n5_2*n9_4 - n1_2*n6_3*n8_3 - n2_2*n4_2*n9_4 + n2_2*n6_3*n7_3 + n3_3*n4_2*n8_3 - n3_3*n5_2*n7_3;

	Eigen::Matrix<floatPrec,9,1> coeff(9);
	coeff[0] = o7_0;
	coeff[1] = o7_1;
	coeff[2] = o7_2;
	coeff[3] = o7_3;
	coeff[4] = o7_4;
	coeff[5] = o7_5;
	coeff[6] = o7_6;
	coeff[7] = o7_7;
	coeff[8] = o7_8;

	std::vector<floatPrec> calc_realRoots;
	Eigen::PolynomialSolver<floatPrec, Eigen::Dynamic> solver;
    unsigned int numHyp = 0;
	if(coeff[coeff.size()-1] != floatPrec(0.0)){
		// std::cout << coeff.transpose() << std::endl;
		solver.compute(coeff);
		solver.realRoots(calc_realRoots);    
		numHyp = calc_realRoots.size();
	}
	else{
		return out;
	}

	floatPrec x, y, z;
	floatPrec r11, r12, r13, r21, r22, r23, r31, r32, r33, r_;

	Matrix<floatPrec, 4,4> TP;
	for(unsigned int i = 0; i < numHyp; i++){

		TP.setIdentity(4,4);

		x = calc_realRoots[i];
		y = (n2_0*n6_0 + n2_0*n6_1*x + n2_0*n6_2*x*x + n2_0*n6_3*x*x*x + n2_1*n6_0*x + n2_1*n6_1*x*x + n2_1*n6_2*x*x*x + n2_1*n6_3*x*x*x*x + n2_2*n6_0*x*x + n2_2*n6_1*x*x*x + n2_2*n6_2*x*x*x*x + n2_2*n6_3*x*x*x*x*x - n3_0*n5_0 - n3_0*n5_1*x - n3_0*n5_2*x*x - n3_1*n5_0*x - n3_1*n5_1*x*x - n3_1*n5_2*x*x*x - n3_2*n5_0*x*x - n3_2*n5_1*x*x*x - n3_2*n5_2*x*x*x*x - n3_3*n5_0*x*x*x - n3_3*n5_1*x*x*x*x - n3_3*n5_2*x*x*x*x*x)/(n1_0*n5_0 + n1_0*n5_1*x + n1_0*n5_2*x*x + n1_1*n5_0*x + n1_1*n5_1*x*x + n1_1*n5_2*x*x*x + n1_2*n5_0*x*x + n1_2*n5_1*x*x*x + n1_2*n5_2*x*x*x*x - n2_0*n4_0 - n2_0*n4_1*x - n2_0*n4_2*x*x - n2_1*n4_0*x - n2_1*n4_1*x*x - n2_1*n4_2*x*x*x - n2_2*n4_0*x*x - n2_2*n4_1*x*x*x - n2_2*n4_2*x*x*x*x);
		z = (-n1_0*n6_0 - n1_0*n6_1*x - n1_0*n6_2*x*x - n1_0*n6_3*x*x*x - n1_1*n6_0*x - n1_1*n6_1*x*x - n1_1*n6_2*x*x*x - n1_1*n6_3*x*x*x*x - n1_2*n6_0*x*x - n1_2*n6_1*x*x*x - n1_2*n6_2*x*x*x*x - n1_2*n6_3*x*x*x*x*x + n3_0*n4_0 + n3_0*n4_1*x + n3_0*n4_2*x*x + n3_1*n4_0*x + n3_1*n4_1*x*x + n3_1*n4_2*x*x*x + n3_2*n4_0*x*x + n3_2*n4_1*x*x*x + n3_2*n4_2*x*x*x*x + n3_3*n4_0*x*x*x + n3_3*n4_1*x*x*x*x + n3_3*n4_2*x*x*x*x*x)/(n1_0*n5_0 + n1_0*n5_1*x + n1_0*n5_2*x*x + n1_1*n5_0*x + n1_1*n5_1*x*x + n1_1*n5_2*x*x*x + n1_2*n5_0*x*x + n1_2*n5_1*x*x*x + n1_2*n5_2*x*x*x*x - n2_0*n4_0 - n2_0*n4_1*x - n2_0*n4_2*x*x - n2_1*n4_0*x - n2_1*n4_1*x*x - n2_1*n4_2*x*x*x - n2_2*n4_0*x*x - n2_2*n4_1*x*x*x - n2_2*n4_2*x*x*x*x);

		r_  = 1 + x*x + y*y + z*z;
		r11 = 1 + x*x - y*y - z*z; r11 /= r_;
		r12 = 2*x*y - 2*z; r12 /= r_;
		r13 = 2*y + 2*x*z; r13 /= r_;
		r21 = 2*x*y + 2*z; r21 /= r_;
		r22 = 1 - x*x + y*y - z*z; r22 /= r_;
		r23 = 2*y*z - 2*x; r23 /= r_;
		r31 = 2*x*z - 2*y; r31 /= r_;
		r32 = 2*x + 2*y*z;  r32 /= r_;
		r33 = 1 - x*x - y*y + z*z; r33 /= r_;

		TP.topLeftCorner(3, 3) << r11, r12, r13, r21, r22, r23, r31, r32, r33;
		Matrix<floatPrec, 4,4> outi = TV.inverse() * TP * TU;

		out.push_back(outi);

	}

	return out;
}


// Solver for the case of 2 planes and 1 line
template vector<Matrix<float, 4, 4>> solver2P1L<float>(
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lCPair);

template vector<Matrix<double, 4, 4>> solver2P1L<double>(
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver2P1L(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair)
{

	// check input vectors size
	if(lPair.size() < 1 || plPair.size() < 2){
		std::cerr << "Solver 2P1L requires at least 2 plane, and 1 line pair!" << std::endl;
		exit(-1);
	}

	// parse inputs
	Eigen::Matrix<floatPrec,4,1> Pi1,Pi2,Nu1,Nu2,P1,P2,Q1,Q2;
	Pi1 = plPair[0].first; Nu1 = plPair[0].second;
	Pi2 = plPair[1].first; Nu2 = plPair[1].second;

	P1 = lPair[0].first.first; P2 = lPair[0].first.second;
	Q1 = lPair[0].second.first; Q2 = lPair[0].second.second;

	bool verbose = false;

	vector<Matrix<floatPrec,4,4>> out_vec;
	Matrix<floatPrec, 4, 4> out;
	out = Matrix<floatPrec, 4, 4>::Zero(4, 4);
	Matrix<floatPrec, 3, 1> tP1, tP2, tQ1, tQ2;
	tP1 = P1.head(3) / P1(3);
	tP2 = P2.head(3) / P2(3);
	tQ1 = Q1.head(3) / Q1(3);
	tQ2 = Q2.head(3) / Q2(3);

	// computing the plucker coordinates of the lines
	// starting from line 1
	Matrix<floatPrec, 3, 1> D1 = tP2 - tP1;
	Matrix<floatPrec, 6, 1> L1;
	L1.head(3) = D1;
	L1.tail(3) = tP2.cross(tP1);
	L1 /= L1.head(3).norm();
	// starting from line 2
	Matrix<floatPrec, 3, 1> D2 = tQ2 - tQ1;
	Matrix<floatPrec, 6, 1> L2;
	L2.head(3) = D2;
	L2.tail(3) = tQ2.cross(tQ1);
	L2 /= L2.head(3).norm();

	// get predefined transformations to simplify the problem
	// see the paper for more details
	vector<Matrix<floatPrec, 4, 4>> predTrans = getPredefinedTransformations2Plane1Line<floatPrec>(Pi1, Pi2, Nu1, Nu2);

	Matrix<floatPrec, 4, 4> TV = predTrans.back();
	predTrans.pop_back();
	Matrix<floatPrec, 4, 4> TU = predTrans.back();
	predTrans.pop_back();

	if (verbose)
		cout << "[VERBOSE] Matrix TU" << endl
			 << TU << endl;
	if (verbose)
		cout << "[VERBOSE] Matrix TV" << endl
			 << TV << endl;

	Matrix<floatPrec, 6, 6> TL, TK;
	// TL = [U,zeros(3,3);-skew(u)*U,U];
	TL = Matrix<floatPrec, 6, 6>::Zero(6, 6);
	TL.topLeftCorner(3, 3) = TU.topLeftCorner(3, 3);
	TL.bottomRightCorner(3, 3) = TU.topLeftCorner(3, 3);
	TL.bottomLeftCorner(3, 3) = -1 * getSkew<floatPrec>(TU.col(3).head(3)) * TU.topLeftCorner(3, 3);
	if (verbose)
		cout << "[VERBOSE] Matrix TL" << endl
			 << TL << endl;

	TK = Matrix<floatPrec, 6, 6>::Zero(6, 6);
	TK.topLeftCorner(3, 3) = TV.topLeftCorner(3, 3);
	TK.bottomRightCorner(3, 3) = TV.topLeftCorner(3, 3);
	TK.bottomLeftCorner(3, 3) = -1 * getSkew<floatPrec>(TV.col(3).head(3)) * TV.topLeftCorner(3, 3);
	if (verbose)
		cout << "[VERBOSE] Matrix TK" << endl
			 << TK << endl;

	floatPrec u11, u12, u13, u14, u15, u16;
	floatPrec v11, v12, v13, v14, v15, v16;

	L1 = TL * L1;
	u11 = L1(0);
	u12 = L1(1);
	u13 = L1(2);
	u14 = L1(3);
	u15 = L1(4);
	u16 = L1(5);

	L2 = TK * L2;
	v11 = L2(0);
	v12 = L2(1);
	v13 = L2(2);
	v14 = L2(3);
	v15 = L2(4);
	v16 = L2(5);

	floatPrec tx = (u11 * v14 + u14 * v11 + u12 * v15 + u15 * v12 + u13 * v16 + u16 * v13) / (u12 * v13 - u13 * v12);

	Matrix<floatPrec, 4, 4> L;
	L << 1, 0, 0, tx,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1;

	out = TV.inverse() * L * TU;

	if (verbose)
		cout << "[VERBOSE] The estimated transformation is:" << endl
			 << out << endl;

	out_vec.push_back(out);
	return out_vec;
}

// Solver for the case of 3 lines, and 1 plane
template vector<Matrix<float, 4, 4>> solver1P3L<float>(
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lCPair);

template vector<Matrix<double, 4, 4>> solver1P3L<double>(
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lCPair);

// Solver for the case of one plane correspondence and two lines intersection
template <typename floatPrec>
vector<Matrix<floatPrec, 4, 4>> solver1P3L(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair)
{

	// check input vectors size
	if(plPair.size() < 1 || lPair.size() < 3){
		std::cerr << "Solver 3L1Q requires at least 1 plane, and 3 line pairs!" << std::endl;
		exit(-1);
	}

	// parse inputs
	Eigen::Matrix<floatPrec, 4, 1> Pi, Nu, P11, P12, Q11, Q12, P21, P22, Q21, Q22, P31, P32, Q31, Q32;
	Pi = plPair[0].first; Nu = plPair[0].second;

	P11 = lPair[0].first.first; P12 = lPair[0].first.second;
	Q11 = lPair[0].second.first; Q12 = lPair[0].second.second;
	P21 = lPair[1].first.first; P22 = lPair[1].first.second;
	Q21 = lPair[1].second.first; Q22 = lPair[1].second.second;
	P31 = lPair[2].first.first; P32 = lPair[2].first.second;
	Q31 = lPair[2].second.first; Q32 = lPair[2].second.second;

	bool verbose = false;

	Matrix<floatPrec, 3, 1> tP11, tP12, tQ11, tQ12;
	Matrix<floatPrec, 3, 1> tP21, tP22, tQ21, tQ22;
	Matrix<floatPrec, 3, 1> tP31, tP32, tQ31, tQ32;
	tP11 = P11.head(3) / P11(3);
	tP12 = P12.head(3) / P12(3);
	tQ11 = Q11.head(3) / Q11(3);
	tQ12 = Q12.head(3) / Q12(3);
	tP21 = P21.head(3) / P21(3);
	tP22 = P22.head(3) / P22(3);
	tQ21 = Q21.head(3) / Q21(3);
	tQ22 = Q22.head(3) / Q22(3);
	tP31 = P31.head(3) / P31(3);
	tP32 = P32.head(3) / P32(3);
	tQ31 = Q31.head(3) / Q31(3);
	tQ32 = Q32.head(3) / Q32(3);

	// computing the plucker coordinates of the lines
	// starting from line 1
	Matrix<floatPrec, 3, 1> D1;
	Matrix<floatPrec, 6, 1> L11;
	Matrix<floatPrec, 6, 1> L21;
	Matrix<floatPrec, 6, 1> L31;

	D1 = tP12 - tP11;
	L11.head(3) = D1;
	L11.tail(3) = tP12.cross(tP11);
	L11 /= L11.head(3).norm();
	D1 = tP22 - tP21;
	L21.head(3) = D1;
	L21.tail(3) = tP22.cross(tP21);
	L21 /= L21.head(3).norm();
	D1 = tP32 - tP31;
	L31.head(3) = D1;
	L31.tail(3) = tP32.cross(tP31);
	L31 /= L31.head(3).norm();

	// starting from line 2
	Matrix<floatPrec, 6, 1> L12;
	Matrix<floatPrec, 6, 1> L22;
	Matrix<floatPrec, 6, 1> L32;
	D1 = tQ12 - tQ11;
	L12.head(3) = D1;
	L12.tail(3) = tQ12.cross(tQ11);
	L12 /= L12.head(3).norm();
	D1 = tQ22 - tQ21;
	L22.head(3) = D1;
	L22.tail(3) = tQ22.cross(tQ21);
	L22 /= L22.head(3).norm();
	D1 = tQ32 - tQ31;
	L32.head(3) = D1;
	L32.tail(3) = tQ32.cross(tQ31);
	L32 /= L32.head(3).norm();

	// get the predefined transformation matrices
	vector<Matrix<floatPrec, 4, 4>> predTrans = getPredefinedTransformations1Plane3Line<floatPrec>(Pi, Nu);
	Matrix<floatPrec, 4, 4> TV = predTrans.back();
	predTrans.pop_back();
	Matrix<floatPrec, 4, 4> TU = predTrans.back();
	predTrans.pop_back();

	if (verbose)
		cout << "[VERBOSE] Matrix TU" << endl
			 << TU << endl;
	if (verbose)
		cout << "[VERBOSE] Matrix TV" << endl
			 << TV << endl;

	// define the transformation to the lines
	Matrix<floatPrec, 6, 6> TL, TK;
	TL = Matrix<floatPrec, 6, 6>::Zero(6, 6);
	TL.topLeftCorner(3, 3) = TU.topLeftCorner(3, 3);
	TL.bottomRightCorner(3, 3) = TU.topLeftCorner(3, 3);
	TL.bottomLeftCorner(3, 3) = -1 * getSkew<floatPrec>(TU.col(3).head(3)) * TU.topLeftCorner(3, 3);
	if (verbose)
		cout << "[VERBOSE] Matrix TL" << endl
			 << TL << endl;

	TK = Matrix<floatPrec, 6, 6>::Zero(6, 6);
	TK.topLeftCorner(3, 3) = TV.topLeftCorner(3, 3);
	TK.bottomRightCorner(3, 3) = TV.topLeftCorner(3, 3);
	TK.bottomLeftCorner(3, 3) = -1 * getSkew<floatPrec>(TV.col(3).head(3)) * TV.topLeftCorner(3, 3);
	if (verbose)
		cout << "[VERBOSE] Matrix TK" << endl
			 << TK << endl;

	// apply predefined transformations to the lines
	// get the coefficients
	// get parameters of the first line correspondence
	Matrix<floatPrec, 6, 1> u1;
	floatPrec u11, u12, u13, u14, u15, u16;
	u1 = TL * L11;
	u11 = u1(0);
	u12 = u1(1);
	u13 = u1(2);
	u14 = u1(3);
	u15 = u1(4);
	u16 = u1(5);
	Matrix<floatPrec, 6, 1> v1;
	floatPrec v11, v12, v13, v14, v15, v16;
	v1 = TK * L12;
	v11 = v1(0);
	v12 = v1(1);
	v13 = v1(2);
	v14 = v1(3);
	v15 = v1(4);
	v16 = v1(5);

	// get the parameters of the secod line correspondence
	Matrix<floatPrec, 6, 1> u2;
	floatPrec u21, u22, u23, u24, u25, u26;
	u2 = TL * L21;
	u21 = u2(0);
	u22 = u2(1);
	u23 = u2(2);
	u24 = u2(3);
	u25 = u2(4);
	u26 = u2(5);

	Matrix<floatPrec, 6, 1> v2;
	floatPrec v21, v22, v23, v24, v25, v26;
	v2 = TK * L22;
	v21 = v2(0);
	v22 = v2(1);
	v23 = v2(2);
	v24 = v2(3);
	v25 = v2(4);
	v26 = v2(5);

	// get the parameters of the third line correspondence
	Matrix<floatPrec, 6, 1> u3;
	floatPrec u31, u32, u33, u34, u35, u36;
	u3 = TL * L31;
	u31 = u3(0);
	u32 = u3(1);
	u33 = u3(2);
	u34 = u3(3);
	u35 = u3(4);
	u36 = u3(5);

	Matrix<floatPrec, 6, 1> v3;
	floatPrec v31, v32, v33, v34, v35, v36;
	v3 = TK * L32;
	v31 = v3(0);
	v32 = v3(1);
	v33 = v3(2);
	v34 = v3(3);
	v35 = v3(4);
	v36 = v3(5);

	// compute the coefficients
	floatPrec c1, c2, c3, c4, c5;
	c1 = u11*u21*u32*v13*v24*v33 - u11*u21*u32*v14*v23*v33 + u11*u21*u33*v13*v24*v32 - u11*u21*u33*v14*v23*v32 - u11*u22*u31*v13*v23*v34 + u11*u22*u31*v14*v23*v33 + u11*u22*u33*v14*v23*v31 - u11*u22*u34*v13*v23*v31 - u11*u23*u31*v13*v22*v34 + u11*u23*u31*v14*v22*v33 - u11*u23*u32*v14*v21*v33 - u11*u23*u33*v14*v21*v32 + u11*u23*u33*v14*v22*v31 - u11*u23*u34*v13*v22*v31 + u11*u24*u32*v13*v21*v33 + u11*u24*u33*v13*v21*v32 + u12*u21*u31*v13*v23*v34 - u12*u21*u31*v13*v24*v33 - u12*u21*u33*v13*v24*v31 + u12*u21*u34*v13*v23*v31 + u12*u23*u31*v13*v21*v34 + u12*u23*u34*v13*v21*v31 - u12*u24*u31*v13*v21*v33 - u12*u24*u33*v13*v21*v31 + u13*u21*u31*v12*v23*v34 - u13*u21*u31*v12*v24*v33 + u13*u21*u32*v11*v24*v33 + u13*u21*u33*v11*v24*v32 - u13*u21*u33*v12*v24*v31 + u13*u21*u34*v12*v23*v31 - u13*u22*u31*v11*v23*v34 - u13*u22*u34*v11*v23*v31 - u13*u23*u31*v11*v22*v34 + u13*u23*u31*v12*v21*v34 - u13*u23*u34*v11*v22*v31 + u13*u23*u34*v12*v21*v31 - u13*u24*u31*v12*v21*v33 + u13*u24*u32*v11*v21*v33 + u13*u24*u33*v11*v21*v32 - u13*u24*u33*v12*v21*v31 - u14*u21*u32*v11*v23*v33 - u14*u21*u33*v11*v23*v32 + u14*u22*u31*v11*v23*v33 + u14*u22*u33*v11*v23*v31 + u14*u23*u31*v11*v22*v33 - u14*u23*u32*v11*v21*v33 - u14*u23*u33*v11*v21*v32 + u14*u23*u33*v11*v22*v31 - u11*u22*u32*v13*v23*v35 + u11*u22*u32*v13*v25*v33 + u11*u22*u33*v13*v25*v32 - u11*u22*u35*v13*v23*v32 - u11*u23*u32*v13*v22*v35 - u11*u23*u35*v13*v22*v32 + u11*u25*u32*v13*v22*v33 + u11*u25*u33*v13*v22*v32 + u12*u21*u32*v13*v23*v35 - u12*u21*u32*v15*v23*v33 - u12*u21*u33*v15*v23*v32 + u12*u21*u35*v13*v23*v32 - u12*u22*u31*v13*v25*v33 + u12*u22*u31*v15*v23*v33 - u12*u22*u33*v13*v25*v31 + u12*u22*u33*v15*v23*v31 + u12*u23*u31*v15*v22*v33 + u12*u23*u32*v13*v21*v35 - u12*u23*u32*v15*v21*v33 - u12*u23*u33*v15*v21*v32 + u12*u23*u33*v15*v22*v31 + u12*u23*u35*v13*v21*v32 - u12*u25*u31*v13*v22*v33 - u12*u25*u33*v13*v22*v31 + u13*u21*u32*v12*v23*v35 + u13*u21*u35*v12*v23*v32 - u13*u22*u31*v12*v25*v33 - u13*u22*u32*v11*v23*v35 + u13*u22*u32*v11*v25*v33 + u13*u22*u33*v11*v25*v32 - u13*u22*u33*v12*v25*v31 - u13*u22*u35*v11*v23*v32 - u13*u23*u32*v11*v22*v35 + u13*u23*u32*v12*v21*v35 - u13*u23*u35*v11*v22*v32 + u13*u23*u35*v12*v21*v32 - u13*u25*u31*v12*v22*v33 + u13*u25*u32*v11*v22*v33 + u13*u25*u33*v11*v22*v32 - u13*u25*u33*v12*v22*v31 - u15*u21*u32*v12*v23*v33 - u15*u21*u33*v12*v23*v32 + u15*u22*u31*v12*v23*v33 + u15*u22*u33*v12*v23*v31 + u15*u23*u31*v12*v22*v33 - u15*u23*u32*v12*v21*v33 - u15*u23*u33*v12*v21*v32 + u15*u23*u33*v12*v22*v31 + u11*u22*u33*v13*v23*v36 + u11*u22*u36*v13*v23*v33 - u11*u23*u32*v13*v26*v33 + u11*u23*u33*v13*v22*v36 - u11*u23*u33*v13*v26*v32 + u11*u23*u36*v13*v22*v33 - u11*u26*u32*v13*v23*v33 - u11*u26*u33*v13*v23*v32 - u12*u21*u33*v13*v23*v36 - u12*u21*u36*v13*v23*v33 + u12*u23*u31*v13*v26*v33 - u12*u23*u33*v13*v21*v36 + u12*u23*u33*v13*v26*v31 - u12*u23*u36*v13*v21*v33 + u12*u26*u31*v13*v23*v33 + u12*u26*u33*v13*v23*v31 + u13*u21*u32*v16*v23*v33 - u13*u21*u33*v12*v23*v36 + u13*u21*u33*v16*v23*v32 - u13*u21*u36*v12*v23*v33 - u13*u22*u31*v16*v23*v33 + u13*u22*u33*v11*v23*v36 - u13*u22*u33*v16*v23*v31 + u13*u22*u36*v11*v23*v33 + u13*u23*u31*v12*v26*v33 - u13*u23*u31*v16*v22*v33 - u13*u23*u32*v11*v26*v33 + u13*u23*u32*v16*v21*v33 + u13*u23*u33*v11*v22*v36 - u13*u23*u33*v11*v26*v32 - u13*u23*u33*v12*v21*v36 + u13*u23*u33*v12*v26*v31 + u13*u23*u33*v16*v21*v32 - u13*u23*u33*v16*v22*v31 + u13*u23*u36*v11*v22*v33 - u13*u23*u36*v12*v21*v33 + u13*u26*u31*v12*v23*v33 - u13*u26*u32*v11*v23*v33 - u13*u26*u33*v11*v23*v32 + u13*u26*u33*v12*v23*v31 + u16*u21*u32*v13*v23*v33 + u16*u21*u33*v13*v23*v32 - u16*u22*u31*v13*v23*v33 - u16*u22*u33*v13*v23*v31 - u16*u23*u31*v13*v22*v33 + u16*u23*u32*v13*v21*v33 + u16*u23*u33*v13*v21*v32 - u16*u23*u33*v13*v22*v31;
	c2 = 2*u11*u21*u33*v13*v24*v31 - 2*u11*u21*u33*v14*v23*v31 - 2*u11*u23*u31*v13*v21*v34 + 2*u11*u23*u31*v14*v21*v33 - 2*u11*u23*u34*v13*v21*v31 + 2*u11*u24*u33*v13*v21*v31 + 2*u13*u21*u31*v11*v23*v34 - 2*u13*u21*u31*v11*v24*v33 + 2*u13*u21*u34*v11*v23*v31 - 2*u13*u24*u31*v11*v21*v33 - 2*u14*u21*u33*v11*v23*v31 + 2*u14*u23*u31*v11*v21*v33 - 2*u11*u21*u32*v13*v25*v33 + 2*u11*u21*u32*v15*v23*v33 - 2*u11*u21*u33*v13*v25*v32 + 2*u11*u21*u33*v15*v23*v32 + 2*u11*u22*u31*v13*v23*v35 - 2*u11*u22*u31*v15*v23*v33 - 2*u11*u22*u32*v13*v23*v34 + 2*u11*u22*u32*v13*v24*v33 + 2*u11*u22*u33*v13*v24*v32 + 2*u11*u22*u33*v13*v25*v31 - 2*u11*u22*u33*v14*v23*v32 - 2*u11*u22*u33*v15*v23*v31 + 2*u11*u22*u34*v13*v23*v32 - 2*u11*u22*u35*v13*v23*v31 + 2*u11*u23*u31*v13*v22*v35 - 2*u11*u23*u31*v15*v22*v33 - 2*u11*u23*u32*v13*v21*v35 - 2*u11*u23*u32*v13*v22*v34 + 2*u11*u23*u32*v14*v22*v33 + 2*u11*u23*u32*v15*v21*v33 + 2*u11*u23*u33*v15*v21*v32 - 2*u11*u23*u33*v15*v22*v31 + 2*u11*u23*u34*v13*v22*v32 - 2*u11*u23*u35*v13*v21*v32 - 2*u11*u23*u35*v13*v22*v31 - 2*u11*u24*u32*v13*v22*v33 - 2*u11*u24*u33*v13*v22*v32 + 2*u11*u25*u32*v13*v21*v33 + 2*u11*u25*u33*v13*v21*v32 + 2*u11*u25*u33*v13*v22*v31 - 2*u12*u21*u31*v13*v23*v35 + 2*u12*u21*u31*v13*v25*v33 + 2*u12*u21*u32*v13*v23*v34 - 2*u12*u21*u32*v14*v23*v33 + 2*u12*u21*u33*v13*v24*v32 + 2*u12*u21*u33*v13*v25*v31 - 2*u12*u21*u33*v14*v23*v32 - 2*u12*u21*u33*v15*v23*v31 - 2*u12*u21*u34*v13*v23*v32 + 2*u12*u21*u35*v13*v23*v31 - 2*u12*u22*u31*v13*v24*v33 + 2*u12*u22*u31*v14*v23*v33 - 2*u12*u22*u33*v13*v24*v31 + 2*u12*u22*u33*v14*v23*v31 - 2*u12*u23*u31*v13*v21*v35 - 2*u12*u23*u31*v13*v22*v34 + 2*u12*u23*u31*v14*v22*v33 + 2*u12*u23*u31*v15*v21*v33 + 2*u12*u23*u32*v13*v21*v34 - 2*u12*u23*u32*v14*v21*v33 - 2*u12*u23*u33*v14*v21*v32 + 2*u12*u23*u33*v14*v22*v31 - 2*u12*u23*u34*v13*v21*v32 - 2*u12*u23*u34*v13*v22*v31 + 2*u12*u23*u35*v13*v21*v31 + 2*u12*u24*u31*v13*v22*v33 + 2*u12*u24*u33*v13*v21*v32 + 2*u12*u24*u33*v13*v22*v31 - 2*u12*u25*u31*v13*v21*v33 - 2*u12*u25*u33*v13*v21*v31 - 2*u13*u21*u31*v12*v23*v35 + 2*u13*u21*u31*v12*v25*v33 + 2*u13*u21*u32*v11*v23*v35 - 2*u13*u21*u32*v11*v25*v33 + 2*u13*u21*u32*v12*v23*v34 - 2*u13*u21*u32*v12*v24*v33 - 2*u13*u21*u33*v11*v25*v32 + 2*u13*u21*u33*v12*v25*v31 - 2*u13*u21*u34*v12*v23*v32 + 2*u13*u21*u35*v11*v23*v32 + 2*u13*u21*u35*v12*v23*v31 + 2*u13*u22*u31*v11*v23*v35 - 2*u13*u22*u31*v11*v25*v33 + 2*u13*u22*u31*v12*v23*v34 - 2*u13*u22*u31*v12*v24*v33 - 2*u13*u22*u32*v11*v23*v34 + 2*u13*u22*u32*v11*v24*v33 + 2*u13*u22*u33*v11*v24*v32 - 2*u13*u22*u33*v12*v24*v31 + 2*u13*u22*u34*v11*v23*v32 + 2*u13*u22*u34*v12*v23*v31 - 2*u13*u22*u35*v11*v23*v31 + 2*u13*u23*u31*v11*v22*v35 - 2*u13*u23*u31*v12*v21*v35 - 2*u13*u23*u32*v11*v22*v34 + 2*u13*u23*u32*v12*v21*v34 + 2*u13*u23*u34*v11*v22*v32 - 2*u13*u23*u34*v12*v21*v32 - 2*u13*u23*u35*v11*v22*v31 + 2*u13*u23*u35*v12*v21*v31 + 2*u13*u24*u31*v12*v22*v33 - 2*u13*u24*u32*v11*v22*v33 - 2*u13*u24*u32*v12*v21*v33 - 2*u13*u24*u33*v11*v22*v32 + 2*u13*u24*u33*v12*v22*v31 - 2*u13*u25*u31*v11*v22*v33 - 2*u13*u25*u31*v12*v21*v33 + 2*u13*u25*u32*v11*v21*v33 + 2*u13*u25*u33*v11*v21*v32 - 2*u13*u25*u33*v12*v21*v31 + 2*u14*u21*u32*v12*v23*v33 + 2*u14*u21*u33*v12*v23*v32 - 2*u14*u22*u31*v12*v23*v33 - 2*u14*u22*u33*v11*v23*v32 - 2*u14*u22*u33*v12*v23*v31 - 2*u14*u23*u31*v12*v22*v33 + 2*u14*u23*u32*v11*v22*v33 + 2*u14*u23*u32*v12*v21*v33 + 2*u14*u23*u33*v12*v21*v32 - 2*u14*u23*u33*v12*v22*v31 - 2*u15*u21*u32*v11*v23*v33 - 2*u15*u21*u33*v11*v23*v32 - 2*u15*u21*u33*v12*v23*v31 + 2*u15*u22*u31*v11*v23*v33 + 2*u15*u22*u33*v11*v23*v31 + 2*u15*u23*u31*v11*v22*v33 + 2*u15*u23*u31*v12*v21*v33 - 2*u15*u23*u32*v11*v21*v33 - 2*u15*u23*u33*v11*v21*v32 + 2*u15*u23*u33*v11*v22*v31 + 2*u11*u23*u33*v13*v21*v36 - 2*u11*u23*u33*v13*v26*v31 + 2*u11*u23*u36*v13*v21*v33 - 2*u11*u26*u33*v13*v23*v31 + 2*u12*u22*u33*v13*v25*v32 - 2*u12*u22*u33*v15*v23*v32 - 2*u12*u23*u32*v13*v22*v35 + 2*u12*u23*u32*v15*v22*v33 - 2*u12*u23*u35*v13*v22*v32 + 2*u12*u25*u33*v13*v22*v32 - 2*u13*u21*u33*v11*v23*v36 + 2*u13*u21*u33*v16*v23*v31 - 2*u13*u21*u36*v11*v23*v33 + 2*u13*u22*u32*v12*v23*v35 - 2*u13*u22*u32*v12*v25*v33 + 2*u13*u22*u35*v12*v23*v32 + 2*u13*u23*u31*v11*v26*v33 - 2*u13*u23*u31*v16*v21*v33 - 2*u13*u25*u32*v12*v22*v33 + 2*u13*u26*u31*v11*v23*v33 - 2*u15*u22*u33*v12*v23*v32 + 2*u15*u23*u32*v12*v22*v33 + 2*u16*u21*u33*v13*v23*v31 - 2*u16*u23*u31*v13*v21*v33 + 2*u12*u23*u33*v13*v22*v36 - 2*u12*u23*u33*v13*v26*v32 + 2*u12*u23*u36*v13*v22*v33 - 2*u12*u26*u33*v13*v23*v32 - 2*u13*u22*u33*v12*v23*v36 + 2*u13*u22*u33*v16*v23*v32 - 2*u13*u22*u36*v12*v23*v33 + 2*u13*u23*u32*v12*v26*v33 - 2*u13*u23*u32*v16*v22*v33 + 2*u13*u26*u32*v12*v23*v33 + 2*u16*u22*u33*v13*v23*v32 - 2*u16*u23*u32*v13*v22*v33;
	c3 = 2*u11*u21*u33*v14*v23*v32 - 4*u11*u21*u33*v13*v25*v31 - 2*u11*u21*u33*v13*v24*v32 + 4*u11*u21*u33*v15*v23*v31 + 4*u11*u22*u33*v13*v24*v31 - 2*u11*u22*u33*v14*v23*v31 + 4*u11*u23*u31*v13*v21*v35 + 2*u11*u23*u31*v13*v22*v34 - 2*u11*u23*u31*v14*v22*v33 - 4*u11*u23*u31*v15*v21*v33 - 4*u11*u23*u32*v13*v21*v34 + 2*u11*u23*u32*v14*v21*v33 + 4*u11*u23*u34*v13*v21*v32 + 2*u11*u23*u34*v13*v22*v31 - 4*u11*u23*u35*v13*v21*v31 - 2*u11*u24*u33*v13*v21*v32 - 4*u11*u24*u33*v13*v22*v31 + 4*u11*u25*u33*v13*v21*v31 + 2*u12*u21*u33*v13*v24*v31 - 4*u12*u21*u33*v14*v23*v31 - 2*u12*u23*u31*v13*v21*v34 + 4*u12*u23*u31*v14*v21*v33 - 2*u12*u23*u34*v13*v21*v31 + 2*u12*u24*u33*v13*v21*v31 - 4*u13*u21*u31*v11*v23*v35 + 4*u13*u21*u31*v11*v25*v33 - 2*u13*u21*u31*v12*v23*v34 + 2*u13*u21*u31*v12*v24*v33 + 4*u13*u21*u32*v11*v23*v34 - 2*u13*u21*u32*v11*v24*v33 - 4*u13*u21*u34*v11*v23*v32 - 2*u13*u21*u34*v12*v23*v31 + 4*u13*u21*u35*v11*v23*v31 + 2*u13*u22*u31*v11*v23*v34 - 4*u13*u22*u31*v11*v24*v33 + 2*u13*u22*u34*v11*v23*v31 + 4*u13*u24*u31*v11*v22*v33 + 2*u13*u24*u31*v12*v21*v33 - 2*u13*u24*u32*v11*v21*v33 - 4*u13*u25*u31*v11*v21*v33 + 2*u14*u21*u33*v11*v23*v32 + 4*u14*u21*u33*v12*v23*v31 - 2*u14*u22*u33*v11*v23*v31 - 2*u14*u23*u31*v11*v22*v33 - 4*u14*u23*u31*v12*v21*v33 + 2*u14*u23*u32*v11*v21*v33 - 4*u15*u21*u33*v11*v23*v31 + 4*u15*u23*u31*v11*v21*v33 - 2*u11*u22*u33*v13*v25*v32 + 4*u11*u22*u33*v15*v23*v32 + 2*u11*u23*u32*v13*v22*v35 - 4*u11*u23*u32*v15*v22*v33 + 2*u11*u23*u35*v13*v22*v32 - 2*u11*u25*u33*v13*v22*v32 - 4*u12*u21*u33*v13*v25*v32 + 2*u12*u21*u33*v15*v23*v32 + 4*u12*u22*u33*v13*v24*v32 + 2*u12*u22*u33*v13*v25*v31 - 4*u12*u22*u33*v14*v23*v32 - 2*u12*u22*u33*v15*v23*v31 + 4*u12*u23*u31*v13*v22*v35 - 2*u12*u23*u31*v15*v22*v33 - 2*u12*u23*u32*v13*v21*v35 - 4*u12*u23*u32*v13*v22*v34 + 4*u12*u23*u32*v14*v22*v33 + 2*u12*u23*u32*v15*v21*v33 + 4*u12*u23*u34*v13*v22*v32 - 2*u12*u23*u35*v13*v21*v32 - 4*u12*u23*u35*v13*v22*v31 - 4*u12*u24*u33*v13*v22*v32 + 4*u12*u25*u33*v13*v21*v32 + 2*u12*u25*u33*v13*v22*v31 - 2*u13*u21*u32*v12*v23*v35 + 4*u13*u21*u32*v12*v25*v33 - 2*u13*u21*u35*v12*v23*v32 - 4*u13*u22*u31*v12*v23*v35 + 2*u13*u22*u31*v12*v25*v33 + 2*u13*u22*u32*v11*v23*v35 - 2*u13*u22*u32*v11*v25*v33 + 4*u13*u22*u32*v12*v23*v34 - 4*u13*u22*u32*v12*v24*v33 - 4*u13*u22*u34*v12*v23*v32 + 2*u13*u22*u35*v11*v23*v32 + 4*u13*u22*u35*v12*v23*v31 + 4*u13*u24*u32*v12*v22*v33 + 2*u13*u25*u31*v12*v22*v33 - 2*u13*u25*u32*v11*v22*v33 - 4*u13*u25*u32*v12*v21*v33 + 4*u14*u22*u33*v12*v23*v32 - 4*u14*u23*u32*v12*v22*v33 + 2*u15*u21*u33*v12*v23*v32 - 4*u15*u22*u33*v11*v23*v32 - 2*u15*u22*u33*v12*v23*v31 - 2*u15*u23*u31*v12*v22*v33 + 4*u15*u23*u32*v11*v22*v33 + 2*u15*u23*u32*v12*v21*v33 + 2*u11*u22*u33*v13*v23*v36 + 2*u11*u22*u36*v13*v23*v33 - 2*u11*u23*u32*v13*v26*v33 - 2*u11*u26*u32*v13*v23*v33 - 2*u12*u21*u33*v13*v23*v36 - 2*u12*u21*u36*v13*v23*v33 + 2*u12*u23*u31*v13*v26*v33 + 2*u12*u26*u31*v13*v23*v33 + 2*u13*u21*u32*v16*v23*v33 - 2*u13*u22*u31*v16*v23*v33 + 2*u13*u23*u33*v11*v22*v36 - 2*u13*u23*u33*v11*v26*v32 - 2*u13*u23*u33*v12*v21*v36 + 2*u13*u23*u33*v12*v26*v31 + 2*u13*u23*u33*v16*v21*v32 - 2*u13*u23*u33*v16*v22*v31 + 2*u13*u23*u36*v11*v22*v33 - 2*u13*u23*u36*v12*v21*v33 - 2*u13*u26*u33*v11*v23*v32 + 2*u13*u26*u33*v12*v23*v31 + 2*u16*u21*u32*v13*v23*v33 - 2*u16*u22*u31*v13*v23*v33 + 2*u16*u23*u33*v13*v21*v32 - 2*u16*u23*u33*v13*v22*v31;
	c4 = 2*u11*u21*u33*v14*v23*v31 - 2*u11*u21*u33*v13*v24*v31 + 2*u11*u23*u31*v13*v21*v34 - 2*u11*u23*u31*v14*v21*v33 + 2*u11*u23*u34*v13*v21*v31 - 2*u11*u24*u33*v13*v21*v31 - 2*u13*u21*u31*v11*v23*v34 + 2*u13*u21*u31*v11*v24*v33 - 2*u13*u21*u34*v11*v23*v31 + 2*u13*u24*u31*v11*v21*v33 + 2*u14*u21*u33*v11*v23*v31 - 2*u14*u23*u31*v11*v21*v33 - 2*u11*u21*u32*v13*v25*v33 + 2*u11*u21*u32*v15*v23*v33 + 2*u11*u21*u33*v13*v25*v32 - 2*u11*u21*u33*v15*v23*v32 + 2*u11*u22*u31*v13*v23*v35 - 2*u11*u22*u31*v15*v23*v33 - 2*u11*u22*u32*v13*v23*v34 + 2*u11*u22*u32*v13*v24*v33 - 2*u11*u22*u33*v13*v24*v32 - 2*u11*u22*u33*v13*v25*v31 + 2*u11*u22*u33*v14*v23*v32 + 2*u11*u22*u33*v15*v23*v31 + 2*u11*u22*u34*v13*v23*v32 - 2*u11*u22*u35*v13*v23*v31 - 2*u11*u23*u31*v13*v22*v35 + 2*u11*u23*u31*v15*v22*v33 + 2*u11*u23*u32*v13*v21*v35 + 2*u11*u23*u32*v13*v22*v34 - 2*u11*u23*u32*v14*v22*v33 - 2*u11*u23*u32*v15*v21*v33 + 2*u11*u23*u33*v15*v21*v32 - 2*u11*u23*u33*v15*v22*v31 - 2*u11*u23*u34*v13*v22*v32 + 2*u11*u23*u35*v13*v21*v32 + 2*u11*u23*u35*v13*v22*v31 - 2*u11*u24*u32*v13*v22*v33 + 2*u11*u24*u33*v13*v22*v32 + 2*u11*u25*u32*v13*v21*v33 - 2*u11*u25*u33*v13*v21*v32 - 2*u11*u25*u33*v13*v22*v31 - 2*u12*u21*u31*v13*v23*v35 + 2*u12*u21*u31*v13*v25*v33 + 2*u12*u21*u32*v13*v23*v34 - 2*u12*u21*u32*v14*v23*v33 - 2*u12*u21*u33*v13*v24*v32 - 2*u12*u21*u33*v13*v25*v31 + 2*u12*u21*u33*v14*v23*v32 + 2*u12*u21*u33*v15*v23*v31 - 2*u12*u21*u34*v13*v23*v32 + 2*u12*u21*u35*v13*v23*v31 - 2*u12*u22*u31*v13*v24*v33 + 2*u12*u22*u31*v14*v23*v33 + 2*u12*u22*u33*v13*v24*v31 - 2*u12*u22*u33*v14*v23*v31 + 2*u12*u23*u31*v13*v21*v35 + 2*u12*u23*u31*v13*v22*v34 - 2*u12*u23*u31*v14*v22*v33 - 2*u12*u23*u31*v15*v21*v33 - 2*u12*u23*u32*v13*v21*v34 + 2*u12*u23*u32*v14*v21*v33 - 2*u12*u23*u33*v14*v21*v32 + 2*u12*u23*u33*v14*v22*v31 + 2*u12*u23*u34*v13*v21*v32 + 2*u12*u23*u34*v13*v22*v31 - 2*u12*u23*u35*v13*v21*v31 + 2*u12*u24*u31*v13*v22*v33 - 2*u12*u24*u33*v13*v21*v32 - 2*u12*u24*u33*v13*v22*v31 - 2*u12*u25*u31*v13*v21*v33 + 2*u12*u25*u33*v13*v21*v31 + 2*u13*u21*u31*v12*v23*v35 - 2*u13*u21*u31*v12*v25*v33 - 2*u13*u21*u32*v11*v23*v35 + 2*u13*u21*u32*v11*v25*v33 - 2*u13*u21*u32*v12*v23*v34 + 2*u13*u21*u32*v12*v24*v33 - 2*u13*u21*u33*v11*v25*v32 + 2*u13*u21*u33*v12*v25*v31 + 2*u13*u21*u34*v12*v23*v32 - 2*u13*u21*u35*v11*v23*v32 - 2*u13*u21*u35*v12*v23*v31 - 2*u13*u22*u31*v11*v23*v35 + 2*u13*u22*u31*v11*v25*v33 - 2*u13*u22*u31*v12*v23*v34 + 2*u13*u22*u31*v12*v24*v33 + 2*u13*u22*u32*v11*v23*v34 - 2*u13*u22*u32*v11*v24*v33 + 2*u13*u22*u33*v11*v24*v32 - 2*u13*u22*u33*v12*v24*v31 - 2*u13*u22*u34*v11*v23*v32 - 2*u13*u22*u34*v12*v23*v31 + 2*u13*u22*u35*v11*v23*v31 + 2*u13*u23*u31*v11*v22*v35 - 2*u13*u23*u31*v12*v21*v35 - 2*u13*u23*u32*v11*v22*v34 + 2*u13*u23*u32*v12*v21*v34 + 2*u13*u23*u34*v11*v22*v32 - 2*u13*u23*u34*v12*v21*v32 - 2*u13*u23*u35*v11*v22*v31 + 2*u13*u23*u35*v12*v21*v31 - 2*u13*u24*u31*v12*v22*v33 + 2*u13*u24*u32*v11*v22*v33 + 2*u13*u24*u32*v12*v21*v33 - 2*u13*u24*u33*v11*v22*v32 + 2*u13*u24*u33*v12*v22*v31 + 2*u13*u25*u31*v11*v22*v33 + 2*u13*u25*u31*v12*v21*v33 - 2*u13*u25*u32*v11*v21*v33 + 2*u13*u25*u33*v11*v21*v32 - 2*u13*u25*u33*v12*v21*v31 + 2*u14*u21*u32*v12*v23*v33 - 2*u14*u21*u33*v12*v23*v32 - 2*u14*u22*u31*v12*v23*v33 + 2*u14*u22*u33*v11*v23*v32 + 2*u14*u22*u33*v12*v23*v31 + 2*u14*u23*u31*v12*v22*v33 - 2*u14*u23*u32*v11*v22*v33 - 2*u14*u23*u32*v12*v21*v33 + 2*u14*u23*u33*v12*v21*v32 - 2*u14*u23*u33*v12*v22*v31 - 2*u15*u21*u32*v11*v23*v33 + 2*u15*u21*u33*v11*v23*v32 + 2*u15*u21*u33*v12*v23*v31 + 2*u15*u22*u31*v11*v23*v33 - 2*u15*u22*u33*v11*v23*v31 - 2*u15*u23*u31*v11*v22*v33 - 2*u15*u23*u31*v12*v21*v33 + 2*u15*u23*u32*v11*v21*v33 - 2*u15*u23*u33*v11*v21*v32 + 2*u15*u23*u33*v11*v22*v31 + 2*u11*u23*u33*v13*v21*v36 - 2*u11*u23*u33*v13*v26*v31 + 2*u11*u23*u36*v13*v21*v33 - 2*u11*u26*u33*v13*v23*v31 - 2*u12*u22*u33*v13*v25*v32 + 2*u12*u22*u33*v15*v23*v32 + 2*u12*u23*u32*v13*v22*v35 - 2*u12*u23*u32*v15*v22*v33 + 2*u12*u23*u35*v13*v22*v32 - 2*u12*u25*u33*v13*v22*v32 - 2*u13*u21*u33*v11*v23*v36 + 2*u13*u21*u33*v16*v23*v31 - 2*u13*u21*u36*v11*v23*v33 - 2*u13*u22*u32*v12*v23*v35 + 2*u13*u22*u32*v12*v25*v33 - 2*u13*u22*u35*v12*v23*v32 + 2*u13*u23*u31*v11*v26*v33 - 2*u13*u23*u31*v16*v21*v33 + 2*u13*u25*u32*v12*v22*v33 + 2*u13*u26*u31*v11*v23*v33 + 2*u15*u22*u33*v12*v23*v32 - 2*u15*u23*u32*v12*v22*v33 + 2*u16*u21*u33*v13*v23*v31 - 2*u16*u23*u31*v13*v21*v33 + 2*u12*u23*u33*v13*v22*v36 - 2*u12*u23*u33*v13*v26*v32 + 2*u12*u23*u36*v13*v22*v33 - 2*u12*u26*u33*v13*v23*v32 - 2*u13*u22*u33*v12*v23*v36 + 2*u13*u22*u33*v16*v23*v32 - 2*u13*u22*u36*v12*v23*v33 + 2*u13*u23*u32*v12*v26*v33 - 2*u13*u23*u32*v16*v22*v33 + 2*u13*u26*u32*v12*v23*v33 + 2*u16*u22*u33*v13*v23*v32 - 2*u16*u23*u32*v13*v22*v33;
	c5 = u11*u21*u32*v14*v23*v33 - u11*u21*u32*v13*v24*v33 + u11*u21*u33*v13*v24*v32 - u11*u21*u33*v14*v23*v32 + u11*u22*u31*v13*v23*v34 - u11*u22*u31*v14*v23*v33 + u11*u22*u33*v14*v23*v31 + u11*u22*u34*v13*v23*v31 - u11*u23*u31*v13*v22*v34 + u11*u23*u31*v14*v22*v33 - u11*u23*u32*v14*v21*v33 + u11*u23*u33*v14*v21*v32 - u11*u23*u33*v14*v22*v31 - u11*u23*u34*v13*v22*v31 - u11*u24*u32*v13*v21*v33 + u11*u24*u33*v13*v21*v32 - u12*u21*u31*v13*v23*v34 + u12*u21*u31*v13*v24*v33 - u12*u21*u33*v13*v24*v31 - u12*u21*u34*v13*v23*v31 + u12*u23*u31*v13*v21*v34 + u12*u23*u34*v13*v21*v31 + u12*u24*u31*v13*v21*v33 - u12*u24*u33*v13*v21*v31 + u13*u21*u31*v12*v23*v34 - u13*u21*u31*v12*v24*v33 + u13*u21*u32*v11*v24*v33 - u13*u21*u33*v11*v24*v32 + u13*u21*u33*v12*v24*v31 + u13*u21*u34*v12*v23*v31 - u13*u22*u31*v11*v23*v34 - u13*u22*u34*v11*v23*v31 + u13*u23*u31*v11*v22*v34 - u13*u23*u31*v12*v21*v34 + u13*u23*u34*v11*v22*v31 - u13*u23*u34*v12*v21*v31 - u13*u24*u31*v12*v21*v33 + u13*u24*u32*v11*v21*v33 - u13*u24*u33*v11*v21*v32 + u13*u24*u33*v12*v21*v31 + u14*u21*u32*v11*v23*v33 - u14*u21*u33*v11*v23*v32 - u14*u22*u31*v11*v23*v33 + u14*u22*u33*v11*v23*v31 + u14*u23*u31*v11*v22*v33 - u14*u23*u32*v11*v21*v33 + u14*u23*u33*v11*v21*v32 - u14*u23*u33*v11*v22*v31 + u11*u22*u32*v13*v23*v35 - u11*u22*u32*v13*v25*v33 + u11*u22*u33*v13*v25*v32 + u11*u22*u35*v13*v23*v32 - u11*u23*u32*v13*v22*v35 - u11*u23*u35*v13*v22*v32 - u11*u25*u32*v13*v22*v33 + u11*u25*u33*v13*v22*v32 - u12*u21*u32*v13*v23*v35 + u12*u21*u32*v15*v23*v33 - u12*u21*u33*v15*v23*v32 - u12*u21*u35*v13*v23*v32 + u12*u22*u31*v13*v25*v33 - u12*u22*u31*v15*v23*v33 - u12*u22*u33*v13*v25*v31 + u12*u22*u33*v15*v23*v31 + u12*u23*u31*v15*v22*v33 + u12*u23*u32*v13*v21*v35 - u12*u23*u32*v15*v21*v33 + u12*u23*u33*v15*v21*v32 - u12*u23*u33*v15*v22*v31 + u12*u23*u35*v13*v21*v32 + u12*u25*u31*v13*v22*v33 - u12*u25*u33*v13*v22*v31 + u13*u21*u32*v12*v23*v35 + u13*u21*u35*v12*v23*v32 - u13*u22*u31*v12*v25*v33 - u13*u22*u32*v11*v23*v35 + u13*u22*u32*v11*v25*v33 - u13*u22*u33*v11*v25*v32 + u13*u22*u33*v12*v25*v31 - u13*u22*u35*v11*v23*v32 + u13*u23*u32*v11*v22*v35 - u13*u23*u32*v12*v21*v35 + u13*u23*u35*v11*v22*v32 - u13*u23*u35*v12*v21*v32 - u13*u25*u31*v12*v22*v33 + u13*u25*u32*v11*v22*v33 - u13*u25*u33*v11*v22*v32 + u13*u25*u33*v12*v22*v31 + u15*u21*u32*v12*v23*v33 - u15*u21*u33*v12*v23*v32 - u15*u22*u31*v12*v23*v33 + u15*u22*u33*v12*v23*v31 + u15*u23*u31*v12*v22*v33 - u15*u23*u32*v12*v21*v33 + u15*u23*u33*v12*v21*v32 - u15*u23*u33*v12*v22*v31 + u11*u22*u33*v13*v23*v36 + u11*u22*u36*v13*v23*v33 - u11*u23*u32*v13*v26*v33 - u11*u23*u33*v13*v22*v36 + u11*u23*u33*v13*v26*v32 - u11*u23*u36*v13*v22*v33 - u11*u26*u32*v13*v23*v33 + u11*u26*u33*v13*v23*v32 - u12*u21*u33*v13*v23*v36 - u12*u21*u36*v13*v23*v33 + u12*u23*u31*v13*v26*v33 + u12*u23*u33*v13*v21*v36 - u12*u23*u33*v13*v26*v31 + u12*u23*u36*v13*v21*v33 + u12*u26*u31*v13*v23*v33 - u12*u26*u33*v13*v23*v31 + u13*u21*u32*v16*v23*v33 + u13*u21*u33*v12*v23*v36 - u13*u21*u33*v16*v23*v32 + u13*u21*u36*v12*v23*v33 - u13*u22*u31*v16*v23*v33 - u13*u22*u33*v11*v23*v36 + u13*u22*u33*v16*v23*v31 - u13*u22*u36*v11*v23*v33 - u13*u23*u31*v12*v26*v33 + u13*u23*u31*v16*v22*v33 + u13*u23*u32*v11*v26*v33 - u13*u23*u32*v16*v21*v33 + u13*u23*u33*v11*v22*v36 - u13*u23*u33*v11*v26*v32 - u13*u23*u33*v12*v21*v36 + u13*u23*u33*v12*v26*v31 + u13*u23*u33*v16*v21*v32 - u13*u23*u33*v16*v22*v31 + u13*u23*u36*v11*v22*v33 - u13*u23*u36*v12*v21*v33 - u13*u26*u31*v12*v23*v33 + u13*u26*u32*v11*v23*v33 - u13*u26*u33*v11*v23*v32 + u13*u26*u33*v12*v23*v31 + u16*u21*u32*v13*v23*v33 - u16*u21*u33*v13*v23*v32 - u16*u22*u31*v13*v23*v33 + u16*u22*u33*v13*v23*v31 + u16*u23*u31*v13*v22*v33 - u16*u23*u32*v13*v21*v33 + u16*u23*u33*v13*v21*v32 - u16*u23*u33*v13*v22*v31;

	//	std::cerr << "[ERR]: new version working" << std::endl;

	// get the roots (will be sTheta)
	Eigen::Matrix<floatPrec, 5, 1> factors;
	factors(0, 0) = c1;
	factors(1, 0) = c2;
	factors(2, 0) = c3;
	factors(3, 0) = c4;
	factors(4, 0) = c5;
	vector<floatPrec> realRoots = o4_roots(factors);
	if (verbose)
		cout << "[VERBOSE] The number of valid solutions is: " << realRoots.size() << endl;

	// estimate the resulting transformations for each of the roots
	floatPrec s, sTheta1, sTheta2, tx, ty;
	floatPrec a, b, shouldBeZero1, shouldBeZero2;
	Matrix<floatPrec, 4, 4> transOut;
	vector<Matrix<floatPrec, 4, 4>> out;
	unsigned int numSolutions = realRoots.size();
	for (unsigned int iter = 0; iter < numSolutions; iter++)
	{

		// get the cTheta
		s = realRoots.back();
		realRoots.pop_back();

		// get the transformation for sTheta1
        a = (1-s*s) / (1+s*s);
        b = 2*s / (1+s*s);
		
		tx = -(u13 * u23 * v11 * v26 - u13 * u23 * v16 * v21 + u13 * u26 * v11 * v23 - u16 * u23 * v13 * v21 - a * u11 * u23 * v14 * v21 + a * u13 * u21 * v11 * v24 + a * u13 * u24 * v11 * v21 - a * u14 * u23 * v11 * v21 - a * u12 * u23 * v15 * v21 + a * u13 * u22 * v11 * v25 + a * u13 * u25 * v11 * v22 - a * u15 * u23 * v12 * v21 - a * u11 * u23 * v13 * v26 - a * u11 * u26 * v13 * v23 + a * u13 * u21 * v16 * v23 + a * u16 * u21 * v13 * v23 - b * u11 * u23 * v15 * v21 + b * u12 * u23 * v14 * v21 + b * u13 * u21 * v11 * v25 - b * u13 * u22 * v11 * v24 + b * u13 * u24 * v11 * v22 - b * u13 * u25 * v11 * v21 - b * u14 * u23 * v12 * v21 + b * u15 * u23 * v11 * v21 + b * u12 * u23 * v13 * v26 + b * u12 * u26 * v13 * v23 - b * u13 * u22 * v16 * v23 - b * u16 * u22 * v13 * v23 - (a * a) * u11 * u21 * v13 * v24 + (a * a) * u11 * u21 * v14 * v23 - (a * a) * u11 * u24 * v13 * v21 + (a * a) * u14 * u21 * v11 * v23 - (a * a) * u11 * u22 * v13 * v25 - (a * a) * u11 * u25 * v13 * v22 + (a * a) * u12 * u21 * v15 * v23 + (a * a) * u15 * u21 * v12 * v23 - (b * b) * u11 * u22 * v15 * v23 + (b * b) * u12 * u21 * v13 * v25 - (b * b) * u12 * u22 * v13 * v24 + (b * b) * u12 * u22 * v14 * v23 + (b * b) * u12 * u24 * v13 * v22 - (b * b) * u12 * u25 * v13 * v21 - (b * b) * u14 * u22 * v12 * v23 + (b * b) * u15 * u22 * v11 * v23 - a * b * u11 * u21 * v13 * v25 + a * b * u11 * u21 * v15 * v23 + a * b * u11 * u22 * v13 * v24 - a * b * u11 * u22 * v14 * v23 - a * b * u11 * u24 * v13 * v22 + a * b * u11 * u25 * v13 * v21 + a * b * u12 * u21 * v13 * v24 - a * b * u12 * u21 * v14 * v23 + a * b * u12 * u24 * v13 * v21 + a * b * u14 * u21 * v12 * v23 - a * b * u14 * u22 * v11 * v23 - a * b * u15 * u21 * v11 * v23 + a * b * u12 * u22 * v13 * v25 - a * b * u12 * u22 * v15 * v23 + a * b * u12 * u25 * v13 * v22 - a * b * u15 * u22 * v12 * v23) / (u13 * u23 * v11 * v22 - u13 * u23 * v12 * v21 - a * u11 * u23 * v13 * v22 + a * u12 * u23 * v13 * v21 + a * u13 * u21 * v12 * v23 - a * u13 * u22 * v11 * v23 + b * u11 * u23 * v13 * v21 - b * u13 * u21 * v11 * v23 + b * u12 * u23 * v13 * v22 - b * u13 * u22 * v12 * v23 + (a * a) * u11 * u22 * v13 * v23 - (a * a) * u12 * u21 * v13 * v23 + (b * b) * u11 * u22 * v13 * v23 - (b * b) * u12 * u21 * v13 * v23);
		ty = -(u13 * u23 * v12 * v26 - u13 * u23 * v16 * v22 + u13 * u26 * v12 * v23 - u16 * u23 * v13 * v22 - a * u11 * u23 * v14 * v22 + a * u13 * u21 * v12 * v24 + a * u13 * u24 * v12 * v21 - a * u14 * u23 * v11 * v22 - a * u12 * u23 * v15 * v22 + a * u13 * u22 * v12 * v25 + a * u13 * u25 * v12 * v22 - a * u15 * u23 * v12 * v22 - a * u12 * u23 * v13 * v26 - a * u12 * u26 * v13 * v23 + a * u13 * u22 * v16 * v23 + a * u16 * u22 * v13 * v23 - b * u11 * u23 * v15 * v22 + b * u12 * u23 * v14 * v22 + b * u13 * u21 * v12 * v25 - b * u13 * u22 * v12 * v24 + b * u13 * u24 * v12 * v22 - b * u13 * u25 * v12 * v21 - b * u14 * u23 * v12 * v22 + b * u15 * u23 * v11 * v22 - b * u11 * u23 * v13 * v26 - b * u11 * u26 * v13 * v23 + b * u13 * u21 * v16 * v23 + b * u16 * u21 * v13 * v23 + (a * a) * u11 * u22 * v14 * v23 - (a * a) * u12 * u21 * v13 * v24 - (a * a) * u12 * u24 * v13 * v21 + (a * a) * u14 * u22 * v11 * v23 - (a * a) * u12 * u22 * v13 * v25 + (a * a) * u12 * u22 * v15 * v23 - (a * a) * u12 * u25 * v13 * v22 + (a * a) * u15 * u22 * v12 * v23 - (b * b) * u11 * u21 * v13 * v25 + (b * b) * u11 * u21 * v15 * v23 + (b * b) * u11 * u22 * v13 * v24 - (b * b) * u11 * u24 * v13 * v22 + (b * b) * u11 * u25 * v13 * v21 - (b * b) * u12 * u21 * v14 * v23 + (b * b) * u14 * u21 * v12 * v23 - (b * b) * u15 * u21 * v11 * v23 - a * b * u11 * u21 * v13 * v24 + a * b * u11 * u21 * v14 * v23 - a * b * u11 * u24 * v13 * v21 + a * b * u14 * u21 * v11 * v23 - a * b * u11 * u22 * v13 * v25 + a * b * u11 * u22 * v15 * v23 - a * b * u11 * u25 * v13 * v22 - a * b * u12 * u21 * v13 * v25 + a * b * u12 * u21 * v15 * v23 + a * b * u12 * u22 * v13 * v24 - a * b * u12 * u22 * v14 * v23 - a * b * u12 * u24 * v13 * v22 + a * b * u12 * u25 * v13 * v21 + a * b * u14 * u22 * v12 * v23 + a * b * u15 * u21 * v12 * v23 - a * b * u15 * u22 * v11 * v23) / (u13 * u23 * v11 * v22 - u13 * u23 * v12 * v21 - a * u11 * u23 * v13 * v22 + a * u12 * u23 * v13 * v21 + a * u13 * u21 * v12 * v23 - a * u13 * u22 * v11 * v23 + b * u11 * u23 * v13 * v21 - b * u13 * u21 * v11 * v23 + b * u12 * u23 * v13 * v22 - b * u13 * u22 * v12 * v23 + (a * a) * u11 * u22 * v13 * v23 - (a * a) * u12 * u21 * v13 * v23 + (b * b) * u11 * u22 * v13 * v23 - (b * b) * u12 * u21 * v13 * v23);
		
		transOut << a, -b, 0, tx,
			b, a, 0, ty,
			0, 0, 1, 0,
			0, 0, 0, 1;
		transOut = TV.inverse() * transOut * TU;
		
		// push the valid transformation to the Output
		out.push_back(transOut);
		if (verbose)
			cout << "[VERBOSE] The estimated " << iter + 1 << " solution is:" << endl
				 << transOut << endl;
	}

	return out;
}

// Solver for the case of 1 line correspondence and 1 point
template std::vector<Eigen::Matrix<float, 4, 4>> solver1M1Q(
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lCPair);

template std::vector<Eigen::Matrix<double, 4, 4>> solver1M1Q(
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver1M1Q(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair){

	// check input vectors size
	if(ptPair.size() < 1 || lCPair.size() < 1){
		std::cerr << "Solver 1M1Q requires at least 1 point, and 1 line correspondence!" << std::endl;
		exit(-1);
	}

	std::vector<Eigen::Matrix<floatPrec,4,4>> out;
	
	// get lines in plucker coordinates
	std::vector<Eigen::Matrix<floatPrec,6,1>> lines1 = getPluckerCoord(lCPair[0]);
	

	// move frames to first line and z axis with its direction
	std::vector<Eigen::Matrix<floatPrec,4,4>> trans = getPredefinedTransformations1M1Q(lines1[0],lines1[1]);
	Eigen::Matrix<floatPrec,4,4> TU1 = trans[0], TU2 = trans[1];
	Eigen::Matrix<floatPrec,6,6> TUL1 = Eigen::Matrix<floatPrec,6,6>::Identity(), TUL2 = Eigen::Matrix<floatPrec,6,6>::Identity();
	TUL1.topLeftCorner(3,3) = TU1.topLeftCorner(3,3);
	TUL1.block(3,3,3,3) = TU1.topLeftCorner(3,3);
	TUL1.block(3,0,3,3) = -1*getSkew<floatPrec>(TU1.topRightCorner(3,1))*TU1.topLeftCorner(3,3);
	TUL2.topLeftCorner(3,3) = TU2.topLeftCorner(3,3);
	TUL2.block(3,3,3,3) = TU2.topLeftCorner(3,3);
	TUL2.block(3,0,3,3) = -1*getSkew<floatPrec>(TU2.topRightCorner(3,1))*TU2.topLeftCorner(3,3);

	// transform point and line to new ref
	Eigen::Matrix<floatPrec,4,1> pU1 = TU1*ptPair[0].first;
	Eigen::Matrix<floatPrec,4,1> pU2 = TU2*ptPair[0].second;
	Eigen::Matrix<floatPrec,6,1> lU1 = TUL1*lines1[0];
	Eigen::Matrix<floatPrec,6,1> lU2 = TUL2*lines1[1];

	// move origin to point projection to line
	Eigen::Matrix<floatPrec,3,1> v1 = -pU1.head(3).transpose()*lU1.head(3)*lU1.head(3);
	Eigen::Matrix<floatPrec,3,1> v2 = -pU2.head(3).transpose()*lU2.head(3)*lU2.head(3);
	Eigen::Matrix<floatPrec,4,4> Tv1 = Eigen::Matrix<floatPrec,4,4>::Identity(), Tv2 = Eigen::Matrix<floatPrec,4,4>::Identity();
	Tv1.topRightCorner(3,1) = v1; Tv2.topRightCorner(3,1) = v2;

	// transform to new frame
	Eigen::Matrix<floatPrec,4,1> pv1 = Tv1*pU1;
	Eigen::Matrix<floatPrec,4,1> pv2 = Tv2*pU2;

	// align P to x axis
	floatPrec alpha1 = atan2(pv1(1),pv1(0));
	floatPrec alpha2 = atan2(pv2(1),pv2(0));
	Eigen::Matrix<floatPrec,3,3> V1, V2;
	V1 << cos(alpha1),sin(alpha1),0,-sin(alpha1),cos(alpha1),0,0,0,1;
	V2 << cos(alpha2),sin(alpha2),0,-sin(alpha2),cos(alpha2),0,0,0,1;
	Eigen::Matrix<floatPrec,4,4> TV1 = Eigen::Matrix<floatPrec,4,4>::Identity();
	TV1.topLeftCorner(3,3) = V1;
	Eigen::Matrix<floatPrec,4,4> TV2 = Eigen::Matrix<floatPrec,4,4>::Identity();
	TV2.topLeftCorner(3,3) = V2;

	Eigen::Matrix<floatPrec,4,4> estTrans = TU2.inverse()*Tv2.inverse()*TV2.inverse()*TV1*Tv1*TU1;
	// std::cout << estTrans << std::endl;
	out.push_back(estTrans);

	return out;
}


// Solver for the case of 2 line correspondences
template std::vector<Eigen::Matrix<float, 4, 4>> solver2M(
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lCPair);

template std::vector<Eigen::Matrix<double, 4, 4>> solver2M(
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver2M(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair){

	// check input vectors size
	if(lCPair.size() < 2){
		std::cerr << "Solver 2M requires at least 2 line correspondences!" << std::endl;
		exit(-1);
	}

	std::vector<Eigen::Matrix<floatPrec,4,4>> out;

	// get lines in plucker coordinates
	std::vector<Eigen::Matrix<floatPrec,6,1>> lines1 = getPluckerCoord(lCPair[0]);
	std::vector<Eigen::Matrix<floatPrec,6,1>> lines2 = getPluckerCoord(lCPair[1]);
	

	// move frames to first line and z axis with its direction
	std::vector<Eigen::Matrix<floatPrec,4,4>> trans = getPredefinedTransformations1M1Q(lines1[0],lines1[1]);
	Eigen::Matrix<floatPrec,4,4> TU1 = trans[0], TU2 = trans[1];
	Eigen::Matrix<floatPrec,6,6> TUL1 = Eigen::Matrix<floatPrec,6,6>::Identity(), TUL2 = Eigen::Matrix<floatPrec,6,6>::Identity();
	TUL1.topLeftCorner(3,3) = TU1.topLeftCorner(3,3);
	TUL1.block(3,3,3,3) = TU1.topLeftCorner(3,3);
	TUL1.block(3,0,3,3) = -getSkew<floatPrec>(TU1.topRightCorner(3,1))*TU1.topLeftCorner(3,3);
	TUL2.topLeftCorner(3,3) = TU2.topLeftCorner(3,3);
	TUL2.block(3,3,3,3) = TU2.topLeftCorner(3,3);
	TUL2.block(3,0,3,3) = -getSkew<floatPrec>(TU2.topRightCorner(3,1))*TU2.topLeftCorner(3,3);

	// transform lines to new frame
	Eigen::Matrix<floatPrec,6,1> lU1 = TUL1*lines1[0];
	Eigen::Matrix<floatPrec,6,1> lU2 = TUL2*lines1[1];
	Eigen::Matrix<floatPrec,6,1> lU3 = TUL1*lines2[0];
	Eigen::Matrix<floatPrec,6,1> lU4 = TUL2*lines2[1];

	// get direction and moment vectors
	Eigen::Matrix<floatPrec,3,1> dir1 = lU1.head(3), m1 = lU1.tail(3);
	Eigen::Matrix<floatPrec,3,1> dir2 = lU2.head(3), m2 = lU2.tail(3);
	Eigen::Matrix<floatPrec,3,1> dir3 = lU3.head(3), m3 = lU3.tail(3);
	Eigen::Matrix<floatPrec,3,1> dir4 = lU4.head(3), m4 = lU4.tail(3);

	// find closest points on the lines
	// frame A
	Eigen::Matrix<floatPrec,3,1> n = dir1.cross(dir3);
	Eigen::Matrix<floatPrec,3,1> p1 = dir1.cross(m1);
	Eigen::Matrix<floatPrec,3,1> p2 = dir3.cross(m3);
	Eigen::Matrix<floatPrec,3,1> n2 = dir3.cross(n);
	Eigen::Matrix<floatPrec,3,1> n1 = dir1.cross(n);
	Eigen::Matrix<floatPrec,3,1> c1 = p1 + (p2-p1).dot(n2)/(dir1.dot(n2))*dir1;
	Eigen::Matrix<floatPrec,4,4> Tv1 = Eigen::Matrix<floatPrec,4,4>::Identity();
	Tv1.topRightCorner(3,1) = c1;
	Eigen::Matrix<floatPrec,6,6> TvL1 = Eigen::Matrix<floatPrec,6,6>::Identity();
	TvL1.block(3,0,3,3) = -getSkew<floatPrec>(c1);
	// frame B
	n = dir2.cross(dir4);
	p1 = dir2.cross(m2);
	p2 = dir4.cross(m4);
	n2 = dir4.cross(n);
	n1 = dir2.cross(n);
	c1 = p1 + (p2-p1).dot(n2)/(dir2.dot(n2))*dir2;
	Eigen::Matrix<floatPrec,4,4> Tv2 = Eigen::Matrix<floatPrec,4,4>::Identity();
	Tv2.topRightCorner(3,1) = c1;
	Eigen::Matrix<floatPrec,6,6> TvL2 = Eigen::Matrix<floatPrec,6,6>::Identity();
	TvL2.block(3,0,3,3) = -getSkew<floatPrec>(c1);

	// transform lines
	Eigen::Matrix<floatPrec,6,1> lv3 = TvL1*lU3;
	Eigen::Matrix<floatPrec,6,1> lv4 = TvL2*lU4;

	// rotate to align x axis with c1 c2
	dir3 = lv3.head(3); m3 = lv3.tail(3);
	dir4 = lv4.head(3); m4 = lv4.tail(3); 
	p1 = dir3.cross(m3);
	p2 = dir4.cross(m4);
	floatPrec alpha1 = atan2(p1(1),p1(0));
	floatPrec alpha2 = atan2(p2(1),p2(0));
	Eigen::Matrix<floatPrec,3,3> V1, V2;
	V1 << cos(alpha1),sin(alpha1),0,-sin(alpha1),cos(alpha1),0,0,0,1;
	V2 << cos(alpha2),sin(alpha2),0,-sin(alpha2),cos(alpha2),0,0,0,1;
	Eigen::Matrix<floatPrec,4,4> TV1 = Eigen::Matrix<floatPrec,4,4>::Identity();
	TV1.topLeftCorner(3,3) = V1;
	Eigen::Matrix<floatPrec,4,4> TV2 = Eigen::Matrix<floatPrec,4,4>::Identity();
	TV2.topLeftCorner(3,3) = V2;

	Eigen::Matrix<floatPrec,4,4> estTrans = TU2.inverse()*Tv2.inverse()*TV2.inverse()*TV1*Tv1*TU1;
	// std::cout << estTrans << std::endl;
	out.push_back(estTrans);

	return out;
}

// Solver for the case of 1 line correspondence and 2 line intersections
template std::vector<Eigen::Matrix<float, 4, 4>> solver1M2L(
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lCPair);

template std::vector<Eigen::Matrix<double, 4, 4>> solver1M2L(
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver1M2L(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair){
	
	// check input vectors size
	if(lCPair.size() < 1 && lPair.size() < 2){
		std::cerr << "Solver 1M2L requires at least 1 line correspondence and 2 line intersections!" << std::endl;
		exit(-1);
	}

	std::vector<Eigen::Matrix<floatPrec,4,4>> out;

	// get lines in plucker coordinates
	std::vector<Eigen::Matrix<floatPrec,6,1>> lines1 = getPluckerCoord(lCPair[0]);
	std::vector<Eigen::Matrix<floatPrec,6,1>> lines2 = getPluckerCoord(lPair[0]);
	std::vector<Eigen::Matrix<floatPrec,6,1>> lines3 = getPluckerCoord(lPair[1]);
	

	// move frames to first line and z axis with its direction
	std::vector<Eigen::Matrix<floatPrec,4,4>> trans = getPredefinedTransformations1M1Q(lines1[0],lines1[1]);
	Eigen::Matrix<floatPrec,4,4> TU1 = trans[0], TU2 = trans[1];
	Eigen::Matrix<floatPrec,6,6> TUL1 = Eigen::Matrix<floatPrec,6,6>::Identity(), TUL2 = Eigen::Matrix<floatPrec,6,6>::Identity();
	TUL1.topLeftCorner(3,3) = TU1.topLeftCorner(3,3);
	TUL1.block(3,3,3,3) = TU1.topLeftCorner(3,3);
	TUL1.block(3,0,3,3) = -getSkew<floatPrec>(TU1.topRightCorner(3,1))*TU1.topLeftCorner(3,3);
	TUL2.topLeftCorner(3,3) = TU2.topLeftCorner(3,3);
	TUL2.block(3,3,3,3) = TU2.topLeftCorner(3,3);
	TUL2.block(3,0,3,3) = -getSkew<floatPrec>(TU2.topRightCorner(3,1))*TU2.topLeftCorner(3,3);

	// transform lines to new frame
	Eigen::Matrix<floatPrec,6,1> lU1 = TUL1*lines1[0];
	Eigen::Matrix<floatPrec,6,1> lU2 = TUL2*lines1[1];
	Eigen::Matrix<floatPrec,6,1> lU3 = TUL1*lines2[0];
	Eigen::Matrix<floatPrec,6,1> lU4 = TUL2*lines2[1];
	Eigen::Matrix<floatPrec,6,1> lU5 = TUL1*lines3[0];
	Eigen::Matrix<floatPrec,6,1> lU6 = TUL2*lines3[1];

	// define coeffs
	floatPrec l11 = lU3(0),l12 = lU3(1),l13 = lU3(2),l14 = lU3(3),l15 = lU3(4),l16 = lU3(5);
	floatPrec l21 = lU5(0),l22 = lU5(1),l23 = lU5(2),l24 = lU5(3),l25 = lU5(4),l26 = lU5(5);
	floatPrec m11 = lU4(0),m12 = lU4(1),m13 = lU4(2),m14 = lU4(3),m15 = lU4(4),m16 = lU4(5);
	floatPrec m21 = lU6(0),m22 = lU6(1),m23 = lU6(2),m24 = lU6(3),m25 = lU6(4),m26 = lU6(5);

	floatPrec a = l11*l21*m14*m22 - l11*l21*m12*m24 - l11*l22*m14*m21 - l11*l24*m12*m21 + l12*l21*m11*m24 + l12*l24*m11*m21 + l14*l21*m11*m22 - l14*l22*m11*m21 - l11*l22*m12*m25 - l11*l25*m12*m22 + l12*l21*m15*m22     + l12*l22*m11*m25 - l12*l22*m15*m21 + l12*l25*m11*m22 + l15*l21*m12*m22 - l15*l22*m12*m21 + l11*l23*m12*m26 + l11*l26*m12*m23 - l12*l23*m11*m26 - l12*l26*m11*m23 - l13*l21*m16*m22 + l13*l22*m16*m21 - l16*l21*m13*m22 + l16*l22*m13*m21;
	floatPrec b = 2*l11*l21*m14*m21 - 2*l11*l21*m11*m24 - 2*l11*l24*m11*m21 + 2*l14*l21*m11*m21 + 2*l11*l21*m12*m25 - 2*l11*l21*m15*m22 - 2*l11*l22*m11*m25 - 2*l11*l22*m12*m24 + 2*l11*l22*m14*m22 + 2*l11*l22*m15*m21 + 2*l11*l24*m12*m22 - 2*l11*l25*m11*m22 - 2*l11*l25*m12*m21 - 2*l12*l21*m11*m25 - 2*l12*l21*m12*m24 + 2*l12*l21*m14*m22 + 2*l12*l21*m15*m21 + 2*l12*l22*m11*m24 - 2*l12*l22*m14*m21 - 2*l12*l24*m11*m22 - 2*l12*l24*m12*m21 + 2*l12*l25*m11*m21 - 2*l14*l21*m12*m22 + 2*l14*l22*m11*m22 + 2*l14*l22*m12*m21 + 2*l15*l21*m11*m22 + 2*l15*l21*m12*m21 - 2*l15*l22*m11*m21 + 2*l11*l23*m11*m26 + 2*l11*l26*m11*m23 - 2*l12*l22*m12*m25 + 2*l12*l22*m15*m22 - 2*l12*l25*m12*m22 - 2*l13*l21*m16*m21 + 2*l15*l22*m12*m22 - 2*l16*l21*m13*m21 + 2*l12*l23*m12*m26 + 2*l12*l26*m12*m23 - 2*l13*l22*m16*m22 - 2*l16*l22*m13*m22;
	floatPrec c = 4*l11*l21*m11*m25 + 2*l11*l21*m12*m24 - 2*l11*l21*m14*m22 - 4*l11*l21*m15*m21 - 4*l11*l22*m11*m24 + 2*l11*l22*m14*m21 + 4*l11*l24*m11*m22 + 2*l11*l24*m12*m21 - 4*l11*l25*m11*m21 - 2*l12*l21*m11*m24 + 4*l12*l21*m14*m21 - 2*l12*l24*m11*m21 - 2*l14*l21*m11*m22 - 4*l14*l21*m12*m21 + 2*l14*l22*m11*m21 + 4*l15*l21*m11*m21 + 2*l11*l22*m12*m25 - 4*l11*l22*m15*m22 + 2*l11*l25*m12*m22 + 4*l12*l21*m12*m25 - 2*l12*l21*m15*m22 - 2*l12*l22*m11*m25 - 4*l12*l22*m12*m24 + 4*l12*l22*m14*m22 + 2*l12*l22*m15*m21 + 4*l12*l24*m12*m22 - 2*l12*l25*m11*m22 - 4*l12*l25*m12*m21 - 4*l14*l22*m12*m22 - 2*l15*l21*m12*m22 + 4*l15*l22*m11*m22 + 2*l15*l22*m12*m21; 
	floatPrec d = 2*l11*l21*m11*m24 - 2*l11*l21*m14*m21 + 2*l11*l24*m11*m21 - 2*l14*l21*m11*m21 - 2*l11*l21*m12*m25 + 2*l11*l21*m15*m22 + 2*l11*l22*m11*m25 + 2*l11*l22*m12*m24 - 2*l11*l22*m14*m22 - 2*l11*l22*m15*m21 - 2*l11*l24*m12*m22 + 2*l11*l25*m11*m22 + 2*l11*l25*m12*m21 + 2*l12*l21*m11*m25 + 2*l12*l21*m12*m24 - 2*l12*l21*m14*m22 - 2*l12*l21*m15*m21 - 2*l12*l22*m11*m24 + 2*l12*l22*m14*m21 + 2*l12*l24*m11*m22 + 2*l12*l24*m12*m21 - 2*l12*l25*m11*m21 + 2*l14*l21*m12*m22 - 2*l14*l22*m11*m22 - 2*l14*l22*m12*m21 - 2*l15*l21*m11*m22 - 2*l15*l21*m12*m21 + 2*l15*l22*m11*m21 + 2*l11*l23*m11*m26 + 2*l11*l26*m11*m23 + 2*l12*l22*m12*m25 - 2*l12*l22*m15*m22 + 2*l12*l25*m12*m22 - 2*l13*l21*m16*m21 - 2*l15*l22*m12*m22 - 2*l16*l21*m13*m21 + 2*l12*l23*m12*m26 + 2*l12*l26*m12*m23 - 2*l13*l22*m16*m22 - 2*l16*l22*m13*m22;
	floatPrec e = l11*l21*m14*m22 - l11*l21*m12*m24 - l11*l22*m14*m21 - l11*l24*m12*m21 + l12*l21*m11*m24 + l12*l24*m11*m21 + l14*l21*m11*m22 - l14*l22*m11*m21 - l11*l22*m12*m25 - l11*l25*m12*m22 + l12*l21*m15*m22 + l12*l22*m11*m25 - l12*l22*m15*m21 + l12*l25*m11*m22 + l15*l21*m12*m22 - l15*l22*m12*m21 - l11*l23*m12*m26 - l11*l26*m12*m23 + l12*l23*m11*m26 + l12*l26*m11*m23 + l13*l21*m16*m22 - l13*l22*m16*m21 + l16*l21*m13*m22 - l16*l22*m13*m21;

	// get the roots (will be sTheta)
	Eigen::Matrix<floatPrec, 5, 1> factors;
	factors(0, 0) = a;
	factors(1, 0) = b;
	factors(2, 0) = c;
	factors(3, 0) = d;
	factors(4, 0) = e;
	vector<floatPrec> realRoots = o4_roots(factors);
	for(size_t iter = 0; iter < realRoots.size(); iter++){
		floatPrec s = realRoots[iter];
		
		floatPrec tz = -(l11*m14 + l14*m11 + l12*m15 + l15*m12 + l13*m16 + l16*m13 - l11*m14*s*s - l14*m11*s*s - l12*m15*s*s - l15*m12*s*s + l13*m16*s*s + l16*m13*s*s + 2*l11*m15*s - 2*l12*m14*s + 2*l14*m12*s - 2*l15*m11*s)/(l12*m11 - l11*m12 + l11*m12*s*s - l12*m11*s*s + 2*l11*m11*s + 2*l12*m12*s);

		Eigen::Matrix<floatPrec,3,3> R;
		R << 1-s*s, -2*s, 0, 2*s, 1-s*s, 0, 0, 0, 1+s*s;
		R = 1/(1+s*s)*R;
		Eigen::Matrix<floatPrec,3,1> t;
		t << 0,0,tz;
		Eigen::Matrix<floatPrec,4,4> Tw = Eigen::Matrix<floatPrec,4,4>::Identity(), sol;
		Tw.topLeftCorner(3,3) = R;
		Tw.topRightCorner(3,1) = t;
		sol = TU2.inverse()*Tw*TU1;
		out.push_back(sol);
	}

	return out;
}

// Solver for the case of 1 line and 1 plane correspondence
template std::vector<Eigen::Matrix<float, 4, 4>> solver1M1P(
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>> lCPair);

template std::vector<Eigen::Matrix<double, 4, 4>> solver1M1P(
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lCPair);

template <typename floatPrec>
std::vector<Eigen::Matrix<floatPrec, 4, 4>> solver1M1P(
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> ptPair,
	std::vector<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> plPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lPair,
	std::vector<std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>> lCPair){

	// check input vectors size
	if(lCPair.size() < 1 && plPair.size() < 2){
		std::cerr << "Solver 1M1P requires at least 1 line and 1 plane correspondence!" << std::endl;
		exit(-1);
	}

	std::vector<Eigen::Matrix<floatPrec,4,4>> out;
	out.clear();

	// parse inputs
	Eigen::Matrix<floatPrec, 4, 1> Pi, Nu;
	Pi = plPair[0].first; Nu = plPair[0].second;
	// get lines in plucker coordinates
	std::vector<Eigen::Matrix<floatPrec,6,1>> lines1 = getPluckerCoord(lCPair[0]);

	// check if line and plane are parallel
	if(abs(Pi.head(3).dot(lines1[0].head(3))) < 1e-6){
		// std::cout << Pi.head(3).transpose() << "   " << lines1[0].head(3).transpose() << std::endl; 
		// std::cout << lCPair[0].first.second.head(3).transpose() << "  " << lCPair[0].first.first.head(3).transpose() << std::endl;
		// std::cerr << "Solver 1M1P: The line and plane cannot be parallel!" << std::endl;
		return out;
		// exit(-1);
	}

	// get the predefined transformation matrices
	vector<Matrix<floatPrec, 4, 4>> predTrans = getPredefinedTransformations1Plane3Line<floatPrec>(Pi, Nu);
	Matrix<floatPrec, 4, 4> TU2 = predTrans.back();
	predTrans.pop_back();
	Matrix<floatPrec, 4, 4> TU1 = predTrans.back();
	predTrans.pop_back();

	Eigen::Matrix<floatPrec,6,6> TUL1 = Eigen::Matrix<floatPrec,6,6>::Identity(), TUL2 = Eigen::Matrix<floatPrec,6,6>::Identity();
	TUL1.topLeftCorner(3,3) = TU1.topLeftCorner(3,3);
	TUL1.block(3,3,3,3) = TU1.topLeftCorner(3,3);
	TUL1.block(3,0,3,3) = -getSkew<floatPrec>(TU1.topRightCorner(3,1))*TU1.topLeftCorner(3,3);
	TUL2.topLeftCorner(3,3) = TU2.topLeftCorner(3,3);
	TUL2.block(3,3,3,3) = TU2.topLeftCorner(3,3);
	TUL2.block(3,0,3,3) = -getSkew<floatPrec>(TU2.topRightCorner(3,1))*TU2.topLeftCorner(3,3);

	// transform features
	Eigen::Matrix<floatPrec,6,1> lU1 = TUL1*lines1[0];
	Eigen::Matrix<floatPrec,6,1> lU2 = TUL2*lines1[1];
	Eigen::Matrix<floatPrec,4,1> PiU = TU1.transpose().inverse()*Pi;
	Eigen::Matrix<floatPrec,4,1> NuU = TU2.transpose().inverse()*Nu;
	
	// get direction and moment vectors
	Eigen::Matrix<floatPrec,3,1> dir1 = lU1.head(3), m1 = lU1.tail(3);
	Eigen::Matrix<floatPrec,3,1> dir2 = lU2.head(3), m2 = lU2.tail(3);
	Eigen::Matrix<floatPrec,3,1> n1 = PiU.head(3), n2 = NuU.head(3);

	// compute intersection line with plane
	Eigen::Matrix<floatPrec,3,1> p1 = (n1.cross(m1) - PiU(3)*dir1)/(n1.dot(dir1));
	Eigen::Matrix<floatPrec,3,1> p2 = (n2.cross(m2) - NuU(3)*dir2)/(n2.dot(dir2));

	Matrix<floatPrec, 4, 4> Tu1 = Matrix<floatPrec, 4, 4>::Identity();
	Tu1.topRightCorner(3,1) = p1;
	Matrix<floatPrec, 4, 4> Tu2 = Matrix<floatPrec, 4, 4>::Identity();
	Tu2.topRightCorner(3,1) = p2;
	Eigen::Matrix<floatPrec,6,6> TuL1 = Eigen::Matrix<floatPrec,6,6>::Identity(), TuL2 = Eigen::Matrix<floatPrec,6,6>::Identity();
	TuL1.block(3,0,3,3) = -getSkew<floatPrec>(p1);
	TuL2.block(3,0,3,3) = -getSkew<floatPrec>(p2);

	// transform features
	Eigen::Matrix<floatPrec,6,1> lu1 = TuL1*lU1;
	Eigen::Matrix<floatPrec,6,1> lu2 = TuL2*lU2;
	Eigen::Matrix<floatPrec,4,1> Piu = Tu1.transpose().inverse()*PiU;
	Eigen::Matrix<floatPrec,4,1> Nuu = Tu2.transpose().inverse()*NuU;

	// align x axis wiht projection of line direction to plane
	// project line direction to plane
	dir1 = lu1.head(3); m1 = lu1.tail(3);
	dir2 = lu2.head(3); m2 = lu2.tail(3);
	n1 = Piu.head(3); n2 = Nuu.head(3);
	// frame A
	Eigen::Matrix<floatPrec,3,1> w = (dir1 - dir1.dot(n1)*n1).normalized();
	floatPrec alpha1 = atan2(w(1),w(0));
	Eigen::Matrix<floatPrec,3,3> V1;
	V1 << cos(alpha1),sin(alpha1),0,-sin(alpha1),cos(alpha1),0,0,0,1;
	Eigen::Matrix<floatPrec,4,4> TV1 = Eigen::Matrix<floatPrec,4,4>::Identity();
	TV1.topLeftCorner(3,3) = V1;
	// frame B
	Eigen::Matrix<floatPrec,3,1> w2 = (dir2 - dir2.dot(n2)*n2).normalized();
	floatPrec alpha2 = atan2(w2(1),w2(0));
	Eigen::Matrix<floatPrec,3,3> V2;
	V2 << cos(alpha2),sin(alpha2),0,-sin(alpha2),cos(alpha2),0,0,0,1;
	Eigen::Matrix<floatPrec,4,4> TV2 = Eigen::Matrix<floatPrec,4,4>::Identity();
	TV2.topLeftCorner(3,3) = V2;

	Eigen::Matrix<floatPrec,4,4> sol = TU2.inverse()*Tu2.inverse()*TV2.inverse()*TV1*Tu1*TU1;
	out.push_back(sol);

	return out;
}



// #######################################################################################################
// Aux functions...
// get Plucker coordinates from pair of pair of points
template vector<Matrix<float, 6,1>> getPluckerCoord(std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>> lPair);

template vector<Matrix<double, 6,1>> getPluckerCoord(std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> lPair);

template <typename floatPrec>
vector<Matrix<floatPrec, 6,1>> getPluckerCoord(std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>> lPair){

	// get line normalized direction
	Eigen::Matrix<floatPrec,3,1> dir1 = lPair.first.second.head(3) - lPair.first.first.head(3);
	Eigen::Matrix<floatPrec,3,1> dir2 = lPair.second.second.head(3) - lPair.second.first.head(3);
	dir1.normalize();
	dir2.normalize();
	// get moment vectors
	Eigen::Matrix<floatPrec,3,1> p1 = lPair.first.first.head(3);
	Eigen::Matrix<floatPrec,3,1> m1 = dir1.cross(p1);
	Eigen::Matrix<floatPrec,3,1> p2 = lPair.second.first.head(3);
	Eigen::Matrix<floatPrec,3,1> m2 = dir2.cross(p2);
	// get plucker coordinates
	Eigen::Matrix<floatPrec,6,1> l1;
	l1.head(3) = dir1; l1.tail(3) = m1;
	Eigen::Matrix<floatPrec,6,1> l2;
	l2.head(3) = dir2; l2.tail(3) = m2;

	vector<Matrix<floatPrec, 6,1>> lines;
	lines.push_back(l1); lines.push_back(l2);

	return lines;
}

// Get the predifined transformations for the case 1 point and 1 line correspondence
template <typename floatPrec>
vector<Matrix<floatPrec, 4,4>> getPredefinedTransformations1M1Q(Matrix<floatPrec, 6,1> l1, Matrix<floatPrec, 6,1> l2){

	// get closest point --in the line-- to the origin
	Eigen::Matrix<floatPrec,3,1> dir1 = l1.head(3), m1 = l1.tail(3);
	Eigen::Matrix<floatPrec,3,1> dir2 = l2.head(3), m2 = l2.tail(3);
	Eigen::Matrix<floatPrec,3,1> x1 = dir1.cross(m1);
	Eigen::Matrix<floatPrec,3,1> x2 = dir2.cross(m2);
	
	// get transform to translate frames to the line
	Eigen::Matrix<floatPrec,3,1> u1 = x1 - x1.transpose()*dir1*dir1;
	Eigen::Matrix<floatPrec,3,1> u2 = x2 - x2.transpose()*dir2*dir2;
	Eigen::Matrix<floatPrec,4,4> Tu1 = Eigen::Matrix<floatPrec,4,4>::Identity(), Tu2 = Eigen::Matrix<floatPrec,4,4>::Identity();
	Tu1.topRightCorner(3,1) = u1;
	Tu2.topRightCorner(3,1) = u2;
	Eigen::Matrix<floatPrec,6,6> TuL1 = Eigen::Matrix<floatPrec,6,6>::Identity(), TuL2 = Eigen::Matrix<floatPrec,6,6>::Identity();
	TuL1.block(3,0,3,3) = -getSkew<floatPrec>(u1);
	TuL2.block(3,0,3,3) = -getSkew<floatPrec>(u2);

	// transform point and line to new ref
	Eigen::Matrix<floatPrec,6,1> lu1 = TuL1*l1;
	Eigen::Matrix<floatPrec,6,1> lu2 = TuL2*l2;

	//align z axis with line direction
	Eigen::Matrix<floatPrec,3,1> U1_3 = lu1.head(3), U2_3 = lu2.head(3);
	Eigen::Matrix<floatPrec,3,1> ex, ey;
	ex << 1,0,0; ey << 0,1,0;
	Eigen::Matrix<floatPrec,3,1> U1_1, U2_1;
	// frame A
	Eigen::Matrix<floatPrec,3,1> U1_3M = ey.cross(U1_3);
	Eigen::Matrix<floatPrec,3,1> U1_3P = ex.cross(U1_3);
	if(U1_3M.norm() > U1_3P.norm())
		U1_1 = U1_3M.normalized();
	else
		U1_1 = U1_3P.normalized();
	Eigen::Matrix<floatPrec,3,3> U1;
	U1.row(0) = U1_1.transpose();
	U1.row(1) = U1_3.cross(U1_1).transpose();
	U1.row(2) = U1_3.transpose();
	Eigen::Matrix<floatPrec,4,4> TU1 = Eigen::Matrix<floatPrec,4,4>::Identity();
	TU1.topLeftCorner(3,3) = U1;
	Eigen::Matrix<floatPrec,6,6> TUL1 = Eigen::Matrix<floatPrec,6,6>::Identity();
	TUL1.topLeftCorner(3,3) = U1; TUL1.block(3,3,3,3) = U1;
	// frame B
	Eigen::Matrix<floatPrec,3,1> U2_3M = ey.cross(U2_3);
	Eigen::Matrix<floatPrec,3,1> U2_3P = ex.cross(U2_3);
	if(U2_3M.norm() > U2_3P.norm())
		U2_1 = U2_3M.normalized();
	else
		U2_1 = U2_3P.normalized();
	Eigen::Matrix<floatPrec,3,3> U2;
	U2.row(0) = U2_1.transpose();
	U2.row(1) = U2_3.cross(U2_1).transpose();
	U2.row(2) = U2_3.transpose();
	Eigen::Matrix<floatPrec,4,4> TU2 = Eigen::Matrix<floatPrec,4,4>::Identity();
	TU2.topLeftCorner(3,3) = U2;
	Eigen::Matrix<floatPrec,6,6> TUL2 = Eigen::Matrix<floatPrec,6,6>::Identity();
	TUL2.topLeftCorner(3,3) = U2; TUL2.block(3,3,3,3) = U2;

	Eigen::Matrix<floatPrec,4,4> TU = TU1*Tu1;
	Eigen::Matrix<floatPrec,4,4> TU_B = TU2*Tu2;

	std::vector<Eigen::Matrix<floatPrec,4,4>> trans;
	trans.push_back(TU);
	trans.push_back(TU_B);

	return trans;

}

// Get the predifined transformations for the case 2 points and 1 line
template <typename floatPrec>
vector<Matrix<floatPrec, 4,4>> getPredefinedTransformations2Points1Line(Matrix<floatPrec, 4,1> P1, Matrix<floatPrec, 4,1> P2, Matrix<floatPrec, 4,1> P1_B, Matrix<floatPrec, 4,1> P2_B)
{

	// translate frames to P1 and P1_B
	Matrix<floatPrec, 4,4> transU, transU_B;
	transU.setIdentity(4,4);
	transU.topRightCorner(3, 1) = -P1.head(3);
	transU_B.setIdentity(4,4);
	transU_B.topRightCorner(3, 1) = -P1_B.head(3);

	// transform P2 and P2_B to the new frame
	Matrix<floatPrec, 4,1> P2_1, P2_B1;
	P2_1 = transU * P2;
	P2_B1 = transU_B * P2_B;

	// align Z axis to the direction of P1 -> P2
	Matrix<floatPrec, 4,4> transV, transV_B;
	transV.setIdentity(4,4);
	transV_B.setIdentity(4,4);

	// compute rotation ref A
	Matrix<floatPrec, 3,1> v3, v1, v1_, t1, t2;
	t1.setZero();
	t1(0) = 1;
	t2.setZero();
	t2(1) = 1;
	v3 = P2_1.head(3).normalized();
	v1 = t1.cross(v3);
	v1_ = t2.cross(v3);
	if (v1.norm() >= v1_.norm())
	{
		v1 = v1.normalized();
	}
	else
	{
		v1 = v1_.normalized();
	}
	transV.topLeftCorner(1, 3) = v1.transpose();
	transV.block(1, 0, 1, 3) = v3.cross(v1).transpose();
	transV.block(2, 0, 1, 3) = v3.transpose();

	// compute rotation ref B
	v3 = P2_B1.head(3).normalized();
	v1 = t1.cross(v3);
	v1_ = t2.cross(v3);
	if (v1.norm() >= v1_.norm())
	{
		v1 = v1.normalized();
	}
	else
	{
		v1 = v1_.normalized();
	}
	transV_B.topLeftCorner(1, 3) = v1.transpose();
	transV_B.block(1, 0, 1, 3) = v3.cross(v1).transpose();
	transV_B.block(2, 0, 1, 3) = v3.transpose();

	vector<Matrix<floatPrec, 4,4>> out;
	Matrix<floatPrec, 4,4> TP, TP_B;
	TP = transV * transU;
	out.push_back(TP);

	TP_B = transV_B * transU_B;
	out.push_back(TP_B);

	return out;
}

// Get the predifined transformations for the case 2 points and 1 line
template <typename floatPrec>
vector<Matrix<floatPrec, 4,4>> getPredefinedTransformations1L1Q1P(
	Matrix<floatPrec, 4,1> P1, Matrix<floatPrec, 4,1> P1_B,
	Matrix<floatPrec, 4,1> Pi, Matrix<floatPrec, 4,1> Pi_B)
{

	// project points P1 and P1_B to planes Pi and Pi_B
	Matrix<floatPrec, 4,1> P1_pi, P1_pi_B;
	P1_pi.head(3) = P1.head(3) - (P1.head(3).dot(Pi.head(3)) + Pi(3)) * Pi.head(3);
	P1_pi(3) = 1.0;
	P1_pi_B.head(3) = P1_B.head(3) - (P1_B.head(3).dot(Pi_B.head(3)) + Pi_B(3)) * Pi_B.head(3);
	P1_pi_B(3) = 1.0;

	// translate frames to P1_pi and P1_pi_B
	Matrix<floatPrec, 4,4> transU, transU_B;
	transU.setIdentity(4,4);
	transU.topRightCorner(3, 1) = -P1_pi.head(3);
	transU_B.setIdentity(4,4);
	transU_B.topRightCorner(3, 1) = -P1_pi_B.head(3);

	// transform P2 and P2_B to the new frame
	Matrix<floatPrec, 4,1> Pi_1, Pi_B1;
	Pi_1 = transU.transpose().inverse() * Pi;
	Pi_B1 = transU_B.transpose().inverse() * Pi_B;

	// aling planes Pi and Pi_B with XY planes of both frames
	Matrix<floatPrec, 4,4> transV, transV_B;
	transV.setIdentity(4,4);
	transV_B.setIdentity(4,4);

	// compute rotation ref A
	Matrix<floatPrec, 3,1> v3, v1, v1_, t1, t2;
	t1.setZero();
	t1(0) = 1;
	t2.setZero();
	t2(1) = 1;
	v3 = Pi_1.head(3).normalized();
	v1 = t1.cross(v3);
	v1_ = t2.cross(v3);
	if (v1.norm() >= v1_.norm())
	{
		v1 = v1.normalized();
	}
	else
	{
		v1 = v1_.normalized();
	}
	transV.topLeftCorner(1, 3) = v1.transpose();
	transV.block(1, 0, 1, 3) = v3.cross(v1).transpose();
	transV.block(2, 0, 1, 3) = v3.transpose();

	// compute rotation ref B
	v3 = Pi_B1.head(3).normalized();
	v1 = t1.cross(v3);
	v1_ = t2.cross(v3);
	if (v1.norm() >= v1_.norm())
	{
		v1 = v1.normalized();
	}
	else
	{
		v1 = v1_.normalized();
	}
	transV_B.topLeftCorner(1, 3) = v1.transpose();
	transV_B.block(1, 0, 1, 3) = v3.cross(v1).transpose();
	transV_B.block(2, 0, 1, 3) = v3.transpose();

	vector<Matrix<floatPrec, 4,4>> out;
	Matrix<floatPrec, 4,4> TP, TP_B;
	TP = transV * transU;
	out.push_back(TP);

	TP_B = transV_B * transU_B;
	out.push_back(TP_B);

	return out;
}

template <typename floatPrec>
vector<Matrix<floatPrec, 4,4>> getPredefinedTransformations3L1Q(
	Matrix<floatPrec, 4,1> P1, Matrix<floatPrec, 4,1> P1_B)
{

	// translate frames to P1 and P1_B
	Matrix<floatPrec, 4,4> transU, transU_B;
	transU.setIdentity(4,4);
	transU.topRightCorner(3, 1) = -P1.head(3);
	transU_B.setIdentity(4,4);
	transU_B.topRightCorner(3, 1) = -P1_B.head(3);

	vector<Matrix<floatPrec, 4,4>> out;
	out.push_back(transU);
	out.push_back(transU_B);

	return out;
}

// Get the predefined transformations for the 2 plane and 1 line case
template <typename floatPrec>
vector<Matrix<floatPrec, 4, 4>> getPredefinedTransformations2Plane1Line(
	Matrix<floatPrec, 4, 1> Pi1, Matrix<floatPrec, 4, 1> Pi2,
	Matrix<floatPrec, 4, 1> Nu1, Matrix<floatPrec, 4, 1> Nu2,
	bool verbose)
{

	vector<Matrix<floatPrec, 4, 4>> outTrans;
	Matrix<floatPrec, 4, 4> auxTrans;
	Matrix<floatPrec, 3, 3> auxRot;
	Matrix<floatPrec, 3, 1> auxT;

	floatPrec pi11, pi12, pi13, pi14;
	floatPrec pi21, pi22, pi23, pi24;
	floatPrec nu11, nu12, nu13, nu14;
	floatPrec nu21, nu22, nu23, nu24;

	pi11 = Pi1(0);
	pi12 = Pi1(1);
	pi13 = Pi1(2);
	pi14 = Pi1(3);
	pi21 = Pi2(0);
	pi22 = Pi2(1);
	pi23 = Pi2(2);
	pi24 = Pi2(3);
	nu11 = Nu1(0);
	nu12 = Nu1(1);
	nu13 = Nu1(2);
	nu14 = Nu1(3);
	nu21 = Nu2(0);
	nu22 = Nu2(1);
	nu23 = Nu2(2);
	nu24 = Nu2(3);

	// Compute U Matrix
	auxRot << 1.0 / sqrt((pi12 * pi12 + pi13 * pi13) * pow((pi12 * pi12) * pi21 + (pi13 * pi13) * pi21 - pi11 * pi12 * pi22 - pi11 * pi13 * pi23, 2.0) + pow(pi12 * pi23 - pi13 * pi22, 2.0) * (pi12 * pi12 + pi13 * pi13) * (pi11 * pi11 + pi12 * pi12 + pi13 * pi13)) * (pi12 * pi23 - pi13 * pi22) * (pi12 * pi12 + pi13 * pi13),
		-1.0 / sqrt((pi12 * pi12 + pi13 * pi13) * pow((pi12 * pi12) * pi21 + (pi13 * pi13) * pi21 - pi11 * pi12 * pi22 - pi11 * pi13 * pi23, 2.0) + pow(pi12 * pi23 - pi13 * pi22, 2.0) * (pi12 * pi12 + pi13 * pi13) * (pi11 * pi11 + pi12 * pi12 + pi13 * pi13)) * (pi11 * pi23 - pi13 * pi21) * (pi12 * pi12 + pi13 * pi13),
		1.0 / sqrt((pi12 * pi12 + pi13 * pi13) * pow((pi12 * pi12) * pi21 + (pi13 * pi13) * pi21 - pi11 * pi12 * pi22 - pi11 * pi13 * pi23, 2.0) + pow(pi12 * pi23 - pi13 * pi22, 2.0) * (pi12 * pi12 + pi13 * pi13) * (pi11 * pi11 + pi12 * pi12 + pi13 * pi13)) * (pi11 * pi22 - pi12 * pi21) * (pi12 * pi12 + pi13 * pi13),
		-1.0 / sqrt((pi12 * pi12 + pi13 * pi13) * pow((pi12 * pi12) * pi21 + (pi13 * pi13) * pi21 - pi11 * pi12 * pi22 - pi11 * pi13 * pi23, 2.0) + pow(pi12 * pi23 - pi13 * pi22, 2.0) * (pi12 * pi12 + pi13 * pi13) * (pi11 * pi11 + pi12 * pi12 + pi13 * pi13)) * (pi12 * pi12 + pi13 * pi13) * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13) * ((pi12 * pi12) * pi21 + (pi13 * pi13) * pi21 - pi11 * pi12 * pi22 - pi11 * pi13 * pi23),
		-1.0 / sqrt((pi12 * pi12 + pi13 * pi13) * pow((pi12 * pi12) * pi21 + (pi13 * pi13) * pi21 - pi11 * pi12 * pi22 - pi11 * pi13 * pi23, 2.0) + pow(pi12 * pi23 - pi13 * pi22, 2.0) * (pi12 * pi12 + pi13 * pi13) * (pi11 * pi11 + pi12 * pi12 + pi13 * pi13)) * (pi12 * pi12 + pi13 * pi13) * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13) * ((pi11 * pi11) * pi22 + (pi13 * pi13) * pi22 - pi11 * pi12 * pi21 - pi12 * pi13 * pi23),
		-1.0 / sqrt((pi12 * pi12 + pi13 * pi13) * pow((pi12 * pi12) * pi21 + (pi13 * pi13) * pi21 - pi11 * pi12 * pi22 - pi11 * pi13 * pi23, 2.0) + pow(pi12 * pi23 - pi13 * pi22, 2.0) * (pi12 * pi12 + pi13 * pi13) * (pi11 * pi11 + pi12 * pi12 + pi13 * pi13)) * (pi12 * pi12 + pi13 * pi13) * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13) * ((pi11 * pi11) * pi23 + (pi12 * pi12) * pi23 - pi11 * pi13 * pi21 - pi12 * pi13 * pi22),
		pi11 * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13),
		pi12 * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13),
		pi13 * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13);

	// Compute u vector
	auxT << 0.0,
		-1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13) * ((pi11 * pi11) * pi24 + (pi12 * pi12) * pi24 + (pi13 * pi13) * pi24 - pi11 * pi14 * pi21 - pi12 * pi14 * pi22 - pi13 * pi14 * pi23) * 1.0 / sqrt((pi11 * pi11) * (pi22 * pi22) + (pi12 * pi12) * (pi21 * pi21) + (pi11 * pi11) * (pi23 * pi23) + (pi13 * pi13) * (pi21 * pi21) + (pi12 * pi12) * (pi23 * pi23) + (pi13 * pi13) * (pi22 * pi22) - pi11 * pi12 * pi21 * pi22 * 2.0 - pi11 * pi13 * pi21 * pi23 * 2.0 - pi12 * pi13 * pi22 * pi23 * 2.0),
		pi14 * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13);

	auxTrans = Matrix<floatPrec, 4, 4>::Identity(4, 4);
	auxTrans.topLeftCorner(3, 3) = auxRot;
	auxTrans.topRightCorner(3, 1) = auxT;

	if (verbose)
		cout << "[VERBOSE] Matrix [U,u;0,0,0,1]" << endl
			 << auxTrans << endl;
	if (verbose)
		cout << "[VERBOSE] inv(T)*T = eye(3)" << endl
			 << auxTrans.inverse() * auxTrans << endl;

	// add U,u to the output vector
	outTrans.push_back(auxTrans);

	// Compute V Matrix
	auxRot << 1.0 / sqrt((nu12 * nu12 + nu13 * nu13) * pow((nu12 * nu12) * nu21 + (nu13 * nu13) * nu21 - nu11 * nu12 * nu22 - nu11 * nu13 * nu23, 2.0) + pow(nu12 * nu23 - nu13 * nu22, 2.0) * (nu12 * nu12 + nu13 * nu13) * (nu11 * nu11 + nu12 * nu12 + nu13 * nu13)) * (nu12 * nu23 - nu13 * nu22) * (nu12 * nu12 + nu13 * nu13),
		-1.0 / sqrt((nu12 * nu12 + nu13 * nu13) * pow((nu12 * nu12) * nu21 + (nu13 * nu13) * nu21 - nu11 * nu12 * nu22 - nu11 * nu13 * nu23, 2.0) + pow(nu12 * nu23 - nu13 * nu22, 2.0) * (nu12 * nu12 + nu13 * nu13) * (nu11 * nu11 + nu12 * nu12 + nu13 * nu13)) * (nu11 * nu23 - nu13 * nu21) * (nu12 * nu12 + nu13 * nu13),
		1.0 / sqrt((nu12 * nu12 + nu13 * nu13) * pow((nu12 * nu12) * nu21 + (nu13 * nu13) * nu21 - nu11 * nu12 * nu22 - nu11 * nu13 * nu23, 2.0) + pow(nu12 * nu23 - nu13 * nu22, 2.0) * (nu12 * nu12 + nu13 * nu13) * (nu11 * nu11 + nu12 * nu12 + nu13 * nu13)) * (nu11 * nu22 - nu12 * nu21) * (nu12 * nu12 + nu13 * nu13),
		-1.0 / sqrt((nu12 * nu12 + nu13 * nu13) * pow((nu12 * nu12) * nu21 + (nu13 * nu13) * nu21 - nu11 * nu12 * nu22 - nu11 * nu13 * nu23, 2.0) + pow(nu12 * nu23 - nu13 * nu22, 2.0) * (nu12 * nu12 + nu13 * nu13) * (nu11 * nu11 + nu12 * nu12 + nu13 * nu13)) * (nu12 * nu12 + nu13 * nu13) * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13) * ((nu12 * nu12) * nu21 + (nu13 * nu13) * nu21 - nu11 * nu12 * nu22 - nu11 * nu13 * nu23),
		-1.0 / sqrt((nu12 * nu12 + nu13 * nu13) * pow((nu12 * nu12) * nu21 + (nu13 * nu13) * nu21 - nu11 * nu12 * nu22 - nu11 * nu13 * nu23, 2.0) + pow(nu12 * nu23 - nu13 * nu22, 2.0) * (nu12 * nu12 + nu13 * nu13) * (nu11 * nu11 + nu12 * nu12 + nu13 * nu13)) * (nu12 * nu12 + nu13 * nu13) * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13) * ((nu11 * nu11) * nu22 + (nu13 * nu13) * nu22 - nu11 * nu12 * nu21 - nu12 * nu13 * nu23),
		-1.0 / sqrt((nu12 * nu12 + nu13 * nu13) * pow((nu12 * nu12) * nu21 + (nu13 * nu13) * nu21 - nu11 * nu12 * nu22 - nu11 * nu13 * nu23, 2.0) + pow(nu12 * nu23 - nu13 * nu22, 2.0) * (nu12 * nu12 + nu13 * nu13) * (nu11 * nu11 + nu12 * nu12 + nu13 * nu13)) * (nu12 * nu12 + nu13 * nu13) * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13) * ((nu11 * nu11) * nu23 + (nu12 * nu12) * nu23 - nu11 * nu13 * nu21 - nu12 * nu13 * nu22),
		nu11 * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13),
		nu12 * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13),
		nu13 * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13);

	// Compute v vector
	auxT << 0.0,
		-1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13) * ((nu11 * nu11) * nu24 + (nu12 * nu12) * nu24 + (nu13 * nu13) * nu24 - nu11 * nu14 * nu21 - nu12 * nu14 * nu22 - nu13 * nu14 * nu23) * 1.0 / sqrt((nu11 * nu11) * (nu22 * nu22) + (nu12 * nu12) * (nu21 * nu21) + (nu11 * nu11) * (nu23 * nu23) + (nu13 * nu13) * (nu21 * nu21) + (nu12 * nu12) * (nu23 * nu23) + (nu13 * nu13) * (nu22 * nu22) - nu11 * nu12 * nu21 * nu22 * 2.0 - nu11 * nu13 * nu21 * nu23 * 2.0 - nu12 * nu13 * nu22 * nu23 * 2.0),
		nu14 * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13);

	auxTrans = Matrix<floatPrec, 4, 4>::Identity(4, 4);
	auxTrans.topLeftCorner(3, 3) = auxRot;
	auxTrans.topRightCorner(3, 1) = auxT;

	if (verbose)
		cout << "[VERBOSE] Matrix [V,v;0,0,0,1]" << endl
			 << auxTrans << endl;
	if (verbose)
		cout << "[VERBOSE] inv(T)*T = eye(3)" << endl
			 << auxTrans.inverse() * auxTrans << endl;

	// add V,v to the output vector
	outTrans.push_back(auxTrans);

	return outTrans;
}

// Get the predefined transformations for the 2 plane and 1 line case

template <typename floatPrec>
vector<Matrix<floatPrec, 4, 4>> getPredefinedTransformations1Plane3Line(
	Matrix<floatPrec, 4, 1> Pi, Matrix<floatPrec, 4, 1> Nu,
	bool verbose)
{

	vector<Matrix<floatPrec, 4, 4>> outTrans;
	Matrix<floatPrec, 4, 4> auxTrans;
	Matrix<floatPrec, 3, 3> auxRot;
	Matrix<floatPrec, 3, 1> auxT;

	floatPrec pi11, pi12, pi13, pi14;
	floatPrec nu11, nu12, nu13, nu14;

	pi11 = Pi(0);
	pi12 = Pi(1);
	pi13 = Pi(2);
	pi14 = Pi(3);
	nu11 = Nu(0);
	nu12 = Nu(1);
	nu13 = Nu(2);
	nu14 = Nu(3);

	// Compute U Matrix
	auxRot << 0.0,
		-pi13 * 1.0 / sqrt(pi12 * pi12 + pi13 * pi13),
		pi12 * 1.0 / sqrt(pi12 * pi12 + pi13 * pi13),
		sqrt(pi12 * pi12 + pi13 * pi13) * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13),
		-pi11 * pi12 * 1.0 / sqrt(pi12 * pi12 + pi13 * pi13) * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13),
		-pi11 * pi13 * 1.0 / sqrt(pi12 * pi12 + pi13 * pi13) * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13),
		pi11 * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13),
		pi12 * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13),
		pi13 * 1.0 / sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13);

	// Compute u vector
	auxT << 0.0, 0.0, pi14 * sqrt(pi11 * pi11 + pi12 * pi12 + pi13 * pi13);

	auxTrans = Matrix<floatPrec, 4, 4>::Identity(4, 4);
	auxTrans.topLeftCorner(3, 3) = auxRot;
	auxTrans.topRightCorner(3, 1) = auxT;

	if (verbose)
		cout << "[VERBOSE] Matrix [U,u;0,0,0,1]" << endl
			 << auxTrans << endl;
	if (verbose)
		cout << "[VERBOSE] inv(T)*T = eye(3)" << endl
			 << auxTrans.inverse() * auxTrans << endl;

	// add U,u to the output vector
	outTrans.push_back(auxTrans);

	// Compute V Matrix
	auxRot << 0.0,
		-nu13 * 1.0 / sqrt(nu12 * nu12 + nu13 * nu13),
		nu12 * 1.0 / sqrt(nu12 * nu12 + nu13 * nu13),
		sqrt(nu12 * nu12 + nu13 * nu13) * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13),
		-nu11 * nu12 * 1.0 / sqrt(nu12 * nu12 + nu13 * nu13) * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13),
		-nu11 * nu13 * 1.0 / sqrt(nu12 * nu12 + nu13 * nu13) * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13),
		nu11 * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13),
		nu12 * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13),
		nu13 * 1.0 / sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13);

	// Compute v vector
	auxT << 0.0, 0.0, nu14 * sqrt(nu11 * nu11 + nu12 * nu12 + nu13 * nu13);

	auxTrans = Matrix<floatPrec, 4, 4>::Identity(4, 4);
	auxTrans.topLeftCorner(3, 3) = auxRot;
	auxTrans.topRightCorner(3, 1) = auxT;

	if (verbose)
		cout << "[VERBOSE] Matrix [V,v;0,0,0,1]" << endl
			 << auxTrans << endl;
	if (verbose)
		cout << "[VERBOSE] inv(T)*T = eye(3)" << endl
			 << auxTrans.inverse() * auxTrans << endl;

	// add V,v to the output vector
	outTrans.push_back(auxTrans);

	return outTrans;
}

// Solver for line intersection equations with orthogonality constraints
// based on the MATLAB code from
// J. Ventura, C. Arth, G. Reitmayr, and D. Schmalstieg,
// "A Minimal Solution to the Generalized Pose-and-Scale Problem", CVPR, 2014,
// generated with the code from
// Kukelova Z., Bujnak M., Pajdla T., Automatic Generator of Minimal Problem Solvers,
// ECCV 2008, Marseille, France, October 12-18, 2008
template <typename floatPrec>
vector<Matrix<floatPrec, Dynamic,1>> genposeandscale_solvecoeffs(Matrix<floatPrec, Dynamic,Dynamic> B)
{

	vector<Matrix<floatPrec, Dynamic,1>> b;

	floatPrec B11, B12, B13, B14, B15, B16, B17, B18, B19;
	floatPrec B21, B22, B23, B24, B25, B26, B27, B28, B29;
	floatPrec B31, B32, B33, B34, B35, B36, B37, B38, B39;
	floatPrec B41, B42, B43, B44, B45, B46, B47, B48, B49;
	floatPrec B51, B52, B53, B54, B55, B56, B57, B58, B59;
	floatPrec B61, B62, B63, B64, B65, B66, B67, B68, B69;

	B11 = B(0, 0);
	B12 = B(1, 0);
	B13 = B(2, 0);
	B14 = B(3, 0);
	B15 = B(4, 0);
	B16 = B(5, 0);
	B17 = B(6, 0);
	B18 = B(7, 0);
	B19 = B(8, 0);
	B21 = B(0, 1);
	B22 = B(1, 1);
	B23 = B(2, 1);
	B24 = B(3, 1);
	B25 = B(4, 1);
	B26 = B(5, 1);
	B27 = B(6, 1);
	B28 = B(7, 1);
	B29 = B(8, 1);
	B31 = B(0, 2);
	B32 = B(1, 2);
	B33 = B(2, 2);
	B34 = B(3, 2);
	B35 = B(4, 2);
	B36 = B(5, 2);
	B37 = B(6, 2);
	B38 = B(7, 2);
	B39 = B(8, 2);
	B41 = B(0, 3);
	B42 = B(1, 3);
	B43 = B(2, 3);
	B44 = B(3, 3);
	B45 = B(4, 3);
	B46 = B(5, 3);
	B47 = B(6, 3);
	B48 = B(7, 3);
	B49 = B(8, 3);
	B51 = B(0, 4);
	B52 = B(1, 4);
	B53 = B(2, 4);
	B54 = B(3, 4);
	B55 = B(4, 4);
	B56 = B(5, 4);
	B57 = B(6, 4);
	B58 = B(7, 4);
	B59 = B(8, 4);
	B61 = B(0, 5);
	B62 = B(1, 5);
	B63 = B(2, 5);
	B64 = B(3, 5);
	B65 = B(4, 5);
	B66 = B(5, 5);
	B67 = B(6, 5);
	B68 = B(7, 5);
	B69 = B(8, 5);

	Matrix<floatPrec, 210,1> c;
	c.setZero();
	c[0] = B11 * B11 - B12 * B12 + B14 * B14 - B15 * B15 + B17 * B17 - B18 * B18;
	c[1] = B11 * B21 * 2.0 - B12 * B22 * 2.0 + B14 * B24 * 2.0 - B15 * B25 * 2.0 + B17 * B27 * 2.0 - B18 * B28 * 2.0;
	c[2] = B21 * B21 - B22 * B22 + B24 * B24 - B25 * B25 + B27 * B27 - B28 * B28;
	c[3] = B11 * B31 * 2.0 - B12 * B32 * 2.0 + B14 * B34 * 2.0 - B15 * B35 * 2.0 + B17 * B37 * 2.0 - B18 * B38 * 2.0;
	c[4] = B21 * B31 * 2.0 - B22 * B32 * 2.0 + B24 * B34 * 2.0 - B25 * B35 * 2.0 + B27 * B37 * 2.0 - B28 * B38 * 2.0;
	c[5] = B31 * B31 - B32 * B32 + B34 * B34 - B35 * B35 + B37 * B37 - B38 * B38;
	c[6] = B11 * B41 * 2.0 - B12 * B42 * 2.0 + B14 * B44 * 2.0 - B15 * B45 * 2.0 + B17 * B47 * 2.0 - B18 * B48 * 2.0;
	c[7] = B21 * B41 * 2.0 - B22 * B42 * 2.0 + B24 * B44 * 2.0 - B25 * B45 * 2.0 + B27 * B47 * 2.0 - B28 * B48 * 2.0;
	c[8] = B31 * B41 * 2.0 - B32 * B42 * 2.0 + B34 * B44 * 2.0 - B35 * B45 * 2.0 + B37 * B47 * 2.0 - B38 * B48 * 2.0;
	c[9] = B41 * B41 - B42 * B42 + B44 * B44 - B45 * B45 + B47 * B47 - B48 * B48;
	c[10] = B11 * B51 * 2.0 - B12 * B52 * 2.0 + B14 * B54 * 2.0 - B15 * B55 * 2.0 + B17 * B57 * 2.0 - B18 * B58 * 2.0;
	c[11] = B21 * B51 * 2.0 - B22 * B52 * 2.0 + B24 * B54 * 2.0 - B25 * B55 * 2.0 + B27 * B57 * 2.0 - B28 * B58 * 2.0;
	c[12] = B31 * B51 * 2.0 - B32 * B52 * 2.0 + B34 * B54 * 2.0 - B35 * B55 * 2.0 + B37 * B57 * 2.0 - B38 * B58 * 2.0;
	c[13] = B41 * B51 * 2.0 - B42 * B52 * 2.0 + B44 * B54 * 2.0 - B45 * B55 * 2.0 + B47 * B57 * 2.0 - B48 * B58 * 2.0;
	c[14] = B51 * B51 - B52 * B52 + B54 * B54 - B55 * B55 + B57 * B57 - B58 * B58;
	c[15] = B11 * B61 * 2.0 - B12 * B62 * 2.0 + B14 * B64 * 2.0 - B15 * B65 * 2.0 + B17 * B67 * 2.0 - B18 * B68 * 2.0;
	c[16] = B21 * B61 * 2.0 - B22 * B62 * 2.0 + B24 * B64 * 2.0 - B25 * B65 * 2.0 + B27 * B67 * 2.0 - B28 * B68 * 2.0;
	c[17] = B31 * B61 * 2.0 - B32 * B62 * 2.0 + B34 * B64 * 2.0 - B35 * B65 * 2.0 + B37 * B67 * 2.0 - B38 * B68 * 2.0;
	c[18] = B41 * B61 * 2.0 - B42 * B62 * 2.0 + B44 * B64 * 2.0 - B45 * B65 * 2.0 + B47 * B67 * 2.0 - B48 * B68 * 2.0;
	c[19] = B51 * B61 * 2.0 - B52 * B62 * 2.0 + B54 * B64 * 2.0 - B55 * B65 * 2.0 + B57 * B67 * 2.0 - B58 * B68 * 2.0;
	c[20] = B61 * B61 - B62 * B62 + B64 * B64 - B65 * B65 + B67 * B67 - B68 * B68;
	c[21] = B11 * B11 - B13 * B13 + B14 * B14 - B16 * B16 + B17 * B17 - B19 * B19;
	c[22] = B11 * B21 * 2.0 - B13 * B23 * 2.0 + B14 * B24 * 2.0 - B16 * B26 * 2.0 + B17 * B27 * 2.0 - B19 * B29 * 2.0;
	c[23] = B21 * B21 - B23 * B23 + B24 * B24 - B26 * B26 + B27 * B27 - B29 * B29;
	c[24] = B11 * B31 * 2.0 - B13 * B33 * 2.0 + B14 * B34 * 2.0 - B16 * B36 * 2.0 + B17 * B37 * 2.0 - B19 * B39 * 2.0;
	c[25] = B21 * B31 * 2.0 - B23 * B33 * 2.0 + B24 * B34 * 2.0 - B26 * B36 * 2.0 + B27 * B37 * 2.0 - B29 * B39 * 2.0;
	c[26] = B31 * B31 - B33 * B33 + B34 * B34 - B36 * B36 + B37 * B37 - B39 * B39;
	c[27] = B11 * B41 * 2.0 - B13 * B43 * 2.0 + B14 * B44 * 2.0 - B16 * B46 * 2.0 + B17 * B47 * 2.0 - B19 * B49 * 2.0;
	c[28] = B21 * B41 * 2.0 - B23 * B43 * 2.0 + B24 * B44 * 2.0 - B26 * B46 * 2.0 + B27 * B47 * 2.0 - B29 * B49 * 2.0;
	c[29] = B31 * B41 * 2.0 - B33 * B43 * 2.0 + B34 * B44 * 2.0 - B36 * B46 * 2.0 + B37 * B47 * 2.0 - B39 * B49 * 2.0;
	c[30] = B41 * B41 - B43 * B43 + B44 * B44 - B46 * B46 + B47 * B47 - B49 * B49;
	c[31] = B11 * B51 * 2.0 - B13 * B53 * 2.0 + B14 * B54 * 2.0 - B16 * B56 * 2.0 + B17 * B57 * 2.0 - B19 * B59 * 2.0;
	c[32] = B21 * B51 * 2.0 - B23 * B53 * 2.0 + B24 * B54 * 2.0 - B26 * B56 * 2.0 + B27 * B57 * 2.0 - B29 * B59 * 2.0;
	c[33] = B31 * B51 * 2.0 - B33 * B53 * 2.0 + B34 * B54 * 2.0 - B36 * B56 * 2.0 + B37 * B57 * 2.0 - B39 * B59 * 2.0;
	c[34] = B41 * B51 * 2.0 - B43 * B53 * 2.0 + B44 * B54 * 2.0 - B46 * B56 * 2.0 + B47 * B57 * 2.0 - B49 * B59 * 2.0;
	c[35] = B51 * B51 - B53 * B53 + B54 * B54 - B56 * B56 + B57 * B57 - B59 * B59;
	c[36] = B11 * B61 * 2.0 - B13 * B63 * 2.0 + B14 * B64 * 2.0 - B16 * B66 * 2.0 + B17 * B67 * 2.0 - B19 * B69 * 2.0;
	c[37] = B21 * B61 * 2.0 - B23 * B63 * 2.0 + B24 * B64 * 2.0 - B26 * B66 * 2.0 + B27 * B67 * 2.0 - B29 * B69 * 2.0;
	c[38] = B31 * B61 * 2.0 - B33 * B63 * 2.0 + B34 * B64 * 2.0 - B36 * B66 * 2.0 + B37 * B67 * 2.0 - B39 * B69 * 2.0;
	c[39] = B41 * B61 * 2.0 - B43 * B63 * 2.0 + B44 * B64 * 2.0 - B46 * B66 * 2.0 + B47 * B67 * 2.0 - B49 * B69 * 2.0;
	c[40] = B51 * B61 * 2.0 - B53 * B63 * 2.0 + B54 * B64 * 2.0 - B56 * B66 * 2.0 + B57 * B67 * 2.0 - B59 * B69 * 2.0;
	c[41] = B61 * B61 - B63 * B63 + B64 * B64 - B66 * B66 + B67 * B67 - B69 * B69;
	c[42] = B11 * B11 + B12 * B12 + B13 * B13 - B14 * B14 - B15 * B15 - B16 * B16;
	c[43] = B11 * B21 * 2.0 + B12 * B22 * 2.0 + B13 * B23 * 2.0 - B14 * B24 * 2.0 - B15 * B25 * 2.0 - B16 * B26 * 2.0;
	c[44] = B21 * B21 + B22 * B22 + B23 * B23 - B24 * B24 - B25 * B25 - B26 * B26;
	c[45] = B11 * B31 * 2.0 + B12 * B32 * 2.0 + B13 * B33 * 2.0 - B14 * B34 * 2.0 - B15 * B35 * 2.0 - B16 * B36 * 2.0;
	c[46] = B21 * B31 * 2.0 + B22 * B32 * 2.0 + B23 * B33 * 2.0 - B24 * B34 * 2.0 - B25 * B35 * 2.0 - B26 * B36 * 2.0;
	c[47] = B31 * B31 + B32 * B32 + B33 * B33 - B34 * B34 - B35 * B35 - B36 * B36;
	c[48] = B11 * B41 * 2.0 + B12 * B42 * 2.0 + B13 * B43 * 2.0 - B14 * B44 * 2.0 - B15 * B45 * 2.0 - B16 * B46 * 2.0;
	c[49] = B21 * B41 * 2.0 + B22 * B42 * 2.0 + B23 * B43 * 2.0 - B24 * B44 * 2.0 - B25 * B45 * 2.0 - B26 * B46 * 2.0;
	c[50] = B31 * B41 * 2.0 + B32 * B42 * 2.0 + B33 * B43 * 2.0 - B34 * B44 * 2.0 - B35 * B45 * 2.0 - B36 * B46 * 2.0;
	c[51] = B41 * B41 + B42 * B42 + B43 * B43 - B44 * B44 - B45 * B45 - B46 * B46;
	c[52] = B11 * B51 * 2.0 + B12 * B52 * 2.0 + B13 * B53 * 2.0 - B14 * B54 * 2.0 - B15 * B55 * 2.0 - B16 * B56 * 2.0;
	c[53] = B21 * B51 * 2.0 + B22 * B52 * 2.0 + B23 * B53 * 2.0 - B24 * B54 * 2.0 - B25 * B55 * 2.0 - B26 * B56 * 2.0;
	c[54] = B31 * B51 * 2.0 + B32 * B52 * 2.0 + B33 * B53 * 2.0 - B34 * B54 * 2.0 - B35 * B55 * 2.0 - B36 * B56 * 2.0;
	c[55] = B41 * B51 * 2.0 + B42 * B52 * 2.0 + B43 * B53 * 2.0 - B44 * B54 * 2.0 - B45 * B55 * 2.0 - B46 * B56 * 2.0;
	c[56] = B51 * B51 + B52 * B52 + B53 * B53 - B54 * B54 - B55 * B55 - B56 * B56;
	c[57] = B11 * B61 * 2.0 + B12 * B62 * 2.0 + B13 * B63 * 2.0 - B14 * B64 * 2.0 - B15 * B65 * 2.0 - B16 * B66 * 2.0;
	c[58] = B21 * B61 * 2.0 + B22 * B62 * 2.0 + B23 * B63 * 2.0 - B24 * B64 * 2.0 - B25 * B65 * 2.0 - B26 * B66 * 2.0;
	c[59] = B31 * B61 * 2.0 + B32 * B62 * 2.0 + B33 * B63 * 2.0 - B34 * B64 * 2.0 - B35 * B65 * 2.0 - B36 * B66 * 2.0;
	c[60] = B41 * B61 * 2.0 + B42 * B62 * 2.0 + B43 * B63 * 2.0 - B44 * B64 * 2.0 - B45 * B65 * 2.0 - B46 * B66 * 2.0;
	c[61] = B51 * B61 * 2.0 + B52 * B62 * 2.0 + B53 * B63 * 2.0 - B54 * B64 * 2.0 - B55 * B65 * 2.0 - B56 * B66 * 2.0;
	c[62] = B61 * B61 + B62 * B62 + B63 * B63 - B64 * B64 - B65 * B65 - B66 * B66;
	c[63] = B11 * B11 + B12 * B12 + B13 * B13 - B17 * B17 - B18 * B18 - B19 * B19;
	c[64] = B11 * B21 * 2.0 + B12 * B22 * 2.0 + B13 * B23 * 2.0 - B17 * B27 * 2.0 - B18 * B28 * 2.0 - B19 * B29 * 2.0;
	c[65] = B21 * B21 + B22 * B22 + B23 * B23 - B27 * B27 - B28 * B28 - B29 * B29;
	c[66] = B11 * B31 * 2.0 + B12 * B32 * 2.0 + B13 * B33 * 2.0 - B17 * B37 * 2.0 - B18 * B38 * 2.0 - B19 * B39 * 2.0;
	c[67] = B21 * B31 * 2.0 + B22 * B32 * 2.0 + B23 * B33 * 2.0 - B27 * B37 * 2.0 - B28 * B38 * 2.0 - B29 * B39 * 2.0;
	c[68] = B31 * B31 + B32 * B32 + B33 * B33 - B37 * B37 - B38 * B38 - B39 * B39;
	c[69] = B11 * B41 * 2.0 + B12 * B42 * 2.0 + B13 * B43 * 2.0 - B17 * B47 * 2.0 - B18 * B48 * 2.0 - B19 * B49 * 2.0;
	c[70] = B21 * B41 * 2.0 + B22 * B42 * 2.0 + B23 * B43 * 2.0 - B27 * B47 * 2.0 - B28 * B48 * 2.0 - B29 * B49 * 2.0;
	c[71] = B31 * B41 * 2.0 + B32 * B42 * 2.0 + B33 * B43 * 2.0 - B37 * B47 * 2.0 - B38 * B48 * 2.0 - B39 * B49 * 2.0;
	c[72] = B41 * B41 + B42 * B42 + B43 * B43 - B47 * B47 - B48 * B48 - B49 * B49;
	c[73] = B11 * B51 * 2.0 + B12 * B52 * 2.0 + B13 * B53 * 2.0 - B17 * B57 * 2.0 - B18 * B58 * 2.0 - B19 * B59 * 2.0;
	c[74] = B21 * B51 * 2.0 + B22 * B52 * 2.0 + B23 * B53 * 2.0 - B27 * B57 * 2.0 - B28 * B58 * 2.0 - B29 * B59 * 2.0;
	c[75] = B31 * B51 * 2.0 + B32 * B52 * 2.0 + B33 * B53 * 2.0 - B37 * B57 * 2.0 - B38 * B58 * 2.0 - B39 * B59 * 2.0;
	c[76] = B41 * B51 * 2.0 + B42 * B52 * 2.0 + B43 * B53 * 2.0 - B47 * B57 * 2.0 - B48 * B58 * 2.0 - B49 * B59 * 2.0;
	c[77] = B51 * B51 + B52 * B52 + B53 * B53 - B57 * B57 - B58 * B58 - B59 * B59;
	c[78] = B11 * B61 * 2.0 + B12 * B62 * 2.0 + B13 * B63 * 2.0 - B17 * B67 * 2.0 - B18 * B68 * 2.0 - B19 * B69 * 2.0;
	c[79] = B21 * B61 * 2.0 + B22 * B62 * 2.0 + B23 * B63 * 2.0 - B27 * B67 * 2.0 - B28 * B68 * 2.0 - B29 * B69 * 2.0;
	c[80] = B31 * B61 * 2.0 + B32 * B62 * 2.0 + B33 * B63 * 2.0 - B37 * B67 * 2.0 - B38 * B68 * 2.0 - B39 * B69 * 2.0;
	c[81] = B41 * B61 * 2.0 + B42 * B62 * 2.0 + B43 * B63 * 2.0 - B47 * B67 * 2.0 - B48 * B68 * 2.0 - B49 * B69 * 2.0;
	c[82] = B51 * B61 * 2.0 + B52 * B62 * 2.0 + B53 * B63 * 2.0 - B57 * B67 * 2.0 - B58 * B68 * 2.0 - B59 * B69 * 2.0;
	c[83] = B61 * B61 + B62 * B62 + B63 * B63 - B67 * B67 - B68 * B68 - B69 * B69;
	c[84] = B11 * B12 + B14 * B15 + B17 * B18;
	c[85] = B11 * B22 + B12 * B21 + B14 * B25 + B15 * B24 + B17 * B28 + B18 * B27;
	c[86] = B21 * B22 + B24 * B25 + B27 * B28;
	c[87] = B11 * B32 + B12 * B31 + B14 * B35 + B15 * B34 + B17 * B38 + B18 * B37;
	c[88] = B21 * B32 + B22 * B31 + B24 * B35 + B25 * B34 + B27 * B38 + B28 * B37;
	c[89] = B31 * B32 + B34 * B35 + B37 * B38;
	c[90] = B11 * B42 + B12 * B41 + B14 * B45 + B15 * B44 + B17 * B48 + B18 * B47;
	c[91] = B21 * B42 + B22 * B41 + B24 * B45 + B25 * B44 + B27 * B48 + B28 * B47;
	c[92] = B31 * B42 + B32 * B41 + B34 * B45 + B35 * B44 + B37 * B48 + B38 * B47;
	c[93] = B41 * B42 + B44 * B45 + B47 * B48;
	c[94] = B11 * B52 + B12 * B51 + B14 * B55 + B15 * B54 + B17 * B58 + B18 * B57;
	c[95] = B21 * B52 + B22 * B51 + B24 * B55 + B25 * B54 + B27 * B58 + B28 * B57;
	c[96] = B31 * B52 + B32 * B51 + B34 * B55 + B35 * B54 + B37 * B58 + B38 * B57;
	c[97] = B41 * B52 + B42 * B51 + B44 * B55 + B45 * B54 + B47 * B58 + B48 * B57;
	c[98] = B51 * B52 + B54 * B55 + B57 * B58;
	c[99] = B11 * B62 + B12 * B61 + B14 * B65 + B15 * B64 + B17 * B68 + B18 * B67;
	c[100] = B21 * B62 + B22 * B61 + B24 * B65 + B25 * B64 + B27 * B68 + B28 * B67;
	c[101] = B31 * B62 + B32 * B61 + B34 * B65 + B35 * B64 + B37 * B68 + B38 * B67;
	c[102] = B41 * B62 + B42 * B61 + B44 * B65 + B45 * B64 + B47 * B68 + B48 * B67;
	c[103] = B51 * B62 + B52 * B61 + B54 * B65 + B55 * B64 + B57 * B68 + B58 * B67;
	c[104] = B61 * B62 + B64 * B65 + B67 * B68;
	c[105] = B11 * B13 + B14 * B16 + B17 * B19;
	c[106] = B11 * B23 + B13 * B21 + B14 * B26 + B16 * B24 + B17 * B29 + B19 * B27;
	c[107] = B21 * B23 + B24 * B26 + B27 * B29;
	c[108] = B11 * B33 + B13 * B31 + B14 * B36 + B16 * B34 + B17 * B39 + B19 * B37;
	c[109] = B21 * B33 + B23 * B31 + B24 * B36 + B26 * B34 + B27 * B39 + B29 * B37;
	c[110] = B31 * B33 + B34 * B36 + B37 * B39;
	c[111] = B11 * B43 + B13 * B41 + B14 * B46 + B16 * B44 + B17 * B49 + B19 * B47;
	c[112] = B21 * B43 + B23 * B41 + B24 * B46 + B26 * B44 + B27 * B49 + B29 * B47;
	c[113] = B31 * B43 + B33 * B41 + B34 * B46 + B36 * B44 + B37 * B49 + B39 * B47;
	c[114] = B41 * B43 + B44 * B46 + B47 * B49;
	c[115] = B11 * B53 + B13 * B51 + B14 * B56 + B16 * B54 + B17 * B59 + B19 * B57;
	c[116] = B21 * B53 + B23 * B51 + B24 * B56 + B26 * B54 + B27 * B59 + B29 * B57;
	c[117] = B31 * B53 + B33 * B51 + B34 * B56 + B36 * B54 + B37 * B59 + B39 * B57;
	c[118] = B41 * B53 + B43 * B51 + B44 * B56 + B46 * B54 + B47 * B59 + B49 * B57;
	c[119] = B51 * B53 + B54 * B56 + B57 * B59;
	c[120] = B11 * B63 + B13 * B61 + B14 * B66 + B16 * B64 + B17 * B69 + B19 * B67;
	c[121] = B21 * B63 + B23 * B61 + B24 * B66 + B26 * B64 + B27 * B69 + B29 * B67;
	c[122] = B31 * B63 + B33 * B61 + B34 * B66 + B36 * B64 + B37 * B69 + B39 * B67;
	c[123] = B41 * B63 + B43 * B61 + B44 * B66 + B46 * B64 + B47 * B69 + B49 * B67;
	c[124] = B51 * B63 + B53 * B61 + B54 * B66 + B56 * B64 + B57 * B69 + B59 * B67;
	c[125] = B61 * B63 + B64 * B66 + B67 * B69;
	c[126] = B12 * B13 + B15 * B16 + B18 * B19;
	c[127] = B12 * B23 + B13 * B22 + B15 * B26 + B16 * B25 + B18 * B29 + B19 * B28;
	c[128] = B22 * B23 + B25 * B26 + B28 * B29;
	c[129] = B12 * B33 + B13 * B32 + B15 * B36 + B16 * B35 + B18 * B39 + B19 * B38;
	c[130] = B22 * B33 + B23 * B32 + B25 * B36 + B26 * B35 + B28 * B39 + B29 * B38;
	c[131] = B32 * B33 + B35 * B36 + B38 * B39;
	c[132] = B12 * B43 + B13 * B42 + B15 * B46 + B16 * B45 + B18 * B49 + B19 * B48;
	c[133] = B22 * B43 + B23 * B42 + B25 * B46 + B26 * B45 + B28 * B49 + B29 * B48;
	c[134] = B32 * B43 + B33 * B42 + B35 * B46 + B36 * B45 + B38 * B49 + B39 * B48;
	c[135] = B42 * B43 + B45 * B46 + B48 * B49;
	c[136] = B12 * B53 + B13 * B52 + B15 * B56 + B16 * B55 + B18 * B59 + B19 * B58;
	c[137] = B22 * B53 + B23 * B52 + B25 * B56 + B26 * B55 + B28 * B59 + B29 * B58;
	c[138] = B32 * B53 + B33 * B52 + B35 * B56 + B36 * B55 + B38 * B59 + B39 * B58;
	c[139] = B42 * B53 + B43 * B52 + B45 * B56 + B46 * B55 + B48 * B59 + B49 * B58;
	c[140] = B52 * B53 + B55 * B56 + B58 * B59;
	c[141] = B12 * B63 + B13 * B62 + B15 * B66 + B16 * B65 + B18 * B69 + B19 * B68;
	c[142] = B22 * B63 + B23 * B62 + B25 * B66 + B26 * B65 + B28 * B69 + B29 * B68;
	c[143] = B32 * B63 + B33 * B62 + B35 * B66 + B36 * B65 + B38 * B69 + B39 * B68;
	c[144] = B42 * B63 + B43 * B62 + B45 * B66 + B46 * B65 + B48 * B69 + B49 * B68;
	c[145] = B52 * B63 + B53 * B62 + B55 * B66 + B56 * B65 + B58 * B69 + B59 * B68;
	c[146] = B62 * B63 + B65 * B66 + B68 * B69;
	c[147] = B11 * B14 + B12 * B15 + B13 * B16;
	c[148] = B11 * B24 + B14 * B21 + B12 * B25 + B15 * B22 + B13 * B26 + B16 * B23;
	c[149] = B21 * B24 + B22 * B25 + B23 * B26;
	c[150] = B11 * B34 + B14 * B31 + B12 * B35 + B15 * B32 + B13 * B36 + B16 * B33;
	c[151] = B21 * B34 + B24 * B31 + B22 * B35 + B25 * B32 + B23 * B36 + B26 * B33;
	c[152] = B31 * B34 + B32 * B35 + B33 * B36;
	c[153] = B11 * B44 + B14 * B41 + B12 * B45 + B15 * B42 + B13 * B46 + B16 * B43;
	c[154] = B21 * B44 + B24 * B41 + B22 * B45 + B25 * B42 + B23 * B46 + B26 * B43;
	c[155] = B31 * B44 + B34 * B41 + B32 * B45 + B35 * B42 + B33 * B46 + B36 * B43;
	c[156] = B41 * B44 + B42 * B45 + B43 * B46;
	c[157] = B11 * B54 + B14 * B51 + B12 * B55 + B15 * B52 + B13 * B56 + B16 * B53;
	c[158] = B21 * B54 + B24 * B51 + B22 * B55 + B25 * B52 + B23 * B56 + B26 * B53;
	c[159] = B31 * B54 + B34 * B51 + B32 * B55 + B35 * B52 + B33 * B56 + B36 * B53;
	c[160] = B41 * B54 + B44 * B51 + B42 * B55 + B45 * B52 + B43 * B56 + B46 * B53;
	c[161] = B51 * B54 + B52 * B55 + B53 * B56;
	c[162] = B11 * B64 + B14 * B61 + B12 * B65 + B15 * B62 + B13 * B66 + B16 * B63;
	c[163] = B21 * B64 + B24 * B61 + B22 * B65 + B25 * B62 + B23 * B66 + B26 * B63;
	c[164] = B31 * B64 + B34 * B61 + B32 * B65 + B35 * B62 + B33 * B66 + B36 * B63;
	c[165] = B41 * B64 + B44 * B61 + B42 * B65 + B45 * B62 + B43 * B66 + B46 * B63;
	c[166] = B51 * B64 + B54 * B61 + B52 * B65 + B55 * B62 + B53 * B66 + B56 * B63;
	c[167] = B61 * B64 + B62 * B65 + B63 * B66;
	c[168] = B11 * B17 + B12 * B18 + B13 * B19;
	c[169] = B11 * B27 + B17 * B21 + B12 * B28 + B18 * B22 + B13 * B29 + B19 * B23;
	c[170] = B21 * B27 + B22 * B28 + B23 * B29;
	c[171] = B11 * B37 + B17 * B31 + B12 * B38 + B18 * B32 + B13 * B39 + B19 * B33;
	c[172] = B21 * B37 + B27 * B31 + B22 * B38 + B28 * B32 + B23 * B39 + B29 * B33;
	c[173] = B31 * B37 + B32 * B38 + B33 * B39;
	c[174] = B11 * B47 + B17 * B41 + B12 * B48 + B18 * B42 + B13 * B49 + B19 * B43;
	c[175] = B21 * B47 + B27 * B41 + B22 * B48 + B28 * B42 + B23 * B49 + B29 * B43;
	c[176] = B31 * B47 + B37 * B41 + B32 * B48 + B38 * B42 + B33 * B49 + B39 * B43;
	c[177] = B41 * B47 + B42 * B48 + B43 * B49;
	c[178] = B11 * B57 + B17 * B51 + B12 * B58 + B18 * B52 + B13 * B59 + B19 * B53;
	c[179] = B21 * B57 + B27 * B51 + B22 * B58 + B28 * B52 + B23 * B59 + B29 * B53;
	c[180] = B31 * B57 + B37 * B51 + B32 * B58 + B38 * B52 + B33 * B59 + B39 * B53;
	c[181] = B41 * B57 + B47 * B51 + B42 * B58 + B48 * B52 + B43 * B59 + B49 * B53;
	c[182] = B51 * B57 + B52 * B58 + B53 * B59;
	c[183] = B11 * B67 + B17 * B61 + B12 * B68 + B18 * B62 + B13 * B69 + B19 * B63;
	c[184] = B21 * B67 + B27 * B61 + B22 * B68 + B28 * B62 + B23 * B69 + B29 * B63;
	c[185] = B31 * B67 + B37 * B61 + B32 * B68 + B38 * B62 + B33 * B69 + B39 * B63;
	c[186] = B41 * B67 + B47 * B61 + B42 * B68 + B48 * B62 + B43 * B69 + B49 * B63;
	c[187] = B51 * B67 + B57 * B61 + B52 * B68 + B58 * B62 + B53 * B69 + B59 * B63;
	c[188] = B61 * B67 + B62 * B68 + B63 * B69;
	c[189] = B14 * B17 + B15 * B18 + B16 * B19;
	c[190] = B14 * B27 + B17 * B24 + B15 * B28 + B18 * B25 + B16 * B29 + B19 * B26;
	c[191] = B24 * B27 + B25 * B28 + B26 * B29;
	c[192] = B14 * B37 + B17 * B34 + B15 * B38 + B18 * B35 + B16 * B39 + B19 * B36;
	c[193] = B24 * B37 + B27 * B34 + B25 * B38 + B28 * B35 + B26 * B39 + B29 * B36;
	c[194] = B34 * B37 + B35 * B38 + B36 * B39;
	c[195] = B14 * B47 + B17 * B44 + B15 * B48 + B18 * B45 + B16 * B49 + B19 * B46;
	c[196] = B24 * B47 + B27 * B44 + B25 * B48 + B28 * B45 + B26 * B49 + B29 * B46;
	c[197] = B34 * B47 + B37 * B44 + B35 * B48 + B38 * B45 + B36 * B49 + B39 * B46;
	c[198] = B44 * B47 + B45 * B48 + B46 * B49;
	c[199] = B14 * B57 + B17 * B54 + B15 * B58 + B18 * B55 + B16 * B59 + B19 * B56;
	c[200] = B24 * B57 + B27 * B54 + B25 * B58 + B28 * B55 + B26 * B59 + B29 * B56;
	c[201] = B34 * B57 + B37 * B54 + B35 * B58 + B38 * B55 + B36 * B59 + B39 * B56;
	c[202] = B44 * B57 + B47 * B54 + B45 * B58 + B48 * B55 + B46 * B59 + B49 * B56;
	c[203] = B54 * B57 + B55 * B58 + B56 * B59;
	c[204] = B14 * B67 + B17 * B64 + B15 * B68 + B18 * B65 + B16 * B69 + B19 * B66;
	c[205] = B24 * B67 + B27 * B64 + B25 * B68 + B28 * B65 + B26 * B69 + B29 * B66;
	c[206] = B34 * B67 + B37 * B64 + B35 * B68 + B38 * B65 + B36 * B69 + B39 * B66;
	c[207] = B44 * B67 + B47 * B64 + B45 * B68 + B48 * B65 + B46 * B69 + B49 * B66;
	c[208] = B54 * B67 + B57 * B64 + B55 * B68 + B58 * B65 + B56 * B69 + B59 * B66;
	c[209] = B64 * B67 + B65 * B68 + B66 * B69;

	//VectorXf M(2688);
	Matrix<floatPrec, Dynamic,Dynamic> M(2688,1);
	M.setZero();
	M(1688) = c(168);
	M(1736) = c(169);
	M(1784) = c(170);
	M(1832) = c(171);
	M(1880) = c(172);
	M(1928) = c(173);
	M(1976) = c(174);
	M(2024) = c(175);
	M(2072) = c(176);
	M(2120) = c(177);
	M(2168) = c(178);
	M(2216) = c(179);
	M(2264) = c(180);
	M(2312) = c(181);
	M(2360) = c(182);
	M(2408) = c(183);
	M(2456) = c(184);
	M(2504) = c(185);
	M(2552) = c(186);
	M(2600) = c(187);
	M(2648) = c(188);
	M(1689) = c(189);
	M(1737) = c(190);
	M(1785) = c(191);
	M(1833) = c(192);
	M(1881) = c(193);
	M(1929) = c(194);
	M(1977) = c(195);
	M(2025) = c(196);
	M(2073) = c(197);
	M(2121) = c(198);
	M(2169) = c(199);
	M(2217) = c(200);
	M(2265) = c(201);
	M(2313) = c(202);
	M(2361) = c(203);
	M(2409) = c(204);
	M(2457) = c(205);
	M(2505) = c(206);
	M(2553) = c(207);
	M(2601) = c(208);
	M(2649) = c(209);

	std::vector<int> ind{14, 61, 204, 491, 970, 1680};
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[0];
	}
	ind.assign({62, 109, 252, 539, 1018, 1728});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[1];
	}
	ind.assign({110, 157, 300, 587, 1066, 1776});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[2];
	}
	ind.assign({206, 253, 348, 635, 1114, 1824});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[3];
	}
	ind.assign({254, 301, 396, 683, 1162, 1872});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[4];
	}
	ind.assign({350, 397, 444, 731, 1210, 1920});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[5];
	}
	ind.assign({494, 541, 636, 779, 1258, 1968});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[6];
	}
	ind.assign({542, 589, 684, 827, 1306, 2016});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[7];
	}
	ind.assign({638, 685, 732, 875, 1354, 2064});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[8];
	}
	ind.assign({782, 829, 876, 923, 1402, 2112});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[9];
	}
	ind.assign({974, 1021, 1116, 1259, 1450, 2160});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[10];
	}
	ind.assign({1022, 1069, 1164, 1307, 1498, 2208});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[11];
	}
	ind.assign({1118, 1165, 1212, 1355, 1546, 2256});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[12];
	}
	ind.assign({1262, 1309, 1356, 1403, 1594, 2304});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[13];
	}
	ind.assign({1454, 1501, 1548, 1595, 1642, 2352});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[14];
	}
	ind.assign({1694, 1741, 1836, 1979, 2170, 2400});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[15];
	}
	ind.assign({1742, 1789, 1884, 2027, 2218, 2448});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[16];
	}
	ind.assign({1838, 1885, 1932, 2075, 2266, 2496});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[17];
	}
	ind.assign({1982, 2029, 2076, 2123, 2314, 2544});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[18];
	}
	ind.assign({2174, 2221, 2268, 2315, 2362, 2592});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[19];
	}
	ind.assign({2414, 2461, 2508, 2555, 2602, 2640});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[20];
	}
	ind.assign({19, 66, 209, 496, 975, 1681});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[21];
	}
	ind.assign({67, 114, 257, 544, 1023, 1729});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[22];
	}
	ind.assign({115, 162, 305, 592, 1071, 1777});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[23];
	}
	ind.assign({211, 258, 353, 640, 1119, 1825});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[24];
	}
	ind.assign({259, 306, 401, 688, 1167, 1873});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[25];
	}
	ind.assign({355, 402, 449, 736, 1215, 1921});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[26];
	}
	ind.assign({499, 546, 641, 784, 1263, 1969});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[27];
	}
	ind.assign({547, 594, 689, 832, 1311, 2017});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[28];
	}
	ind.assign({643, 690, 737, 880, 1359, 2065});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[29];
	}
	ind.assign({787, 834, 881, 928, 1407, 2113});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[30];
	}
	ind.assign({979, 1026, 1121, 1264, 1455, 2161});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[31];
	}
	ind.assign({1027, 1074, 1169, 1312, 1503, 2209});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[32];
	}
	ind.assign({1123, 1170, 1217, 1360, 1551, 2257});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[33];
	}
	ind.assign({1267, 1314, 1361, 1408, 1599, 2305});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[34];
	}
	ind.assign({1459, 1506, 1553, 1600, 1647, 2353});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[35];
	}
	ind.assign({1699, 1746, 1841, 1984, 2175, 2401});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[36];
	}
	ind.assign({1747, 1794, 1889, 2032, 2223, 2449});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[37];
	}
	ind.assign({1843, 1890, 1937, 2080, 2271, 2497});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[38];
	}
	ind.assign({1987, 2034, 2081, 2128, 2319, 2545});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[39];
	}
	ind.assign({2179, 2226, 2273, 2320, 2367, 2593});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[40];
	}
	ind.assign({2419, 2466, 2513, 2560, 2607, 2641});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[41];
	}
	ind.assign({24, 71, 214, 501, 980, 1682});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[42];
	}
	ind.assign({72, 119, 262, 549, 1028, 1730});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[43];
	}
	ind.assign({120, 167, 310, 597, 1076, 1778});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[44];
	}
	ind.assign({216, 263, 358, 645, 1124, 1826});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[45];
	}
	ind.assign({264, 311, 406, 693, 1172, 1874});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[46];
	}
	ind.assign({360, 407, 454, 741, 1220, 1922});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[47];
	}
	ind.assign({504, 551, 646, 789, 1268, 1970});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[48];
	}
	ind.assign({552, 599, 694, 837, 1316, 2018});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[49];
	}
	ind.assign({648, 695, 742, 885, 1364, 2066});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[50];
	}
	ind.assign({792, 839, 886, 933, 1412, 2114});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[51];
	}
	ind.assign({984, 1031, 1126, 1269, 1460, 2162});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[52];
	}
	ind.assign({1032, 1079, 1174, 1317, 1508, 2210});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[53];
	}
	ind.assign({1128, 1175, 1222, 1365, 1556, 2258});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[54];
	}
	ind.assign({1272, 1319, 1366, 1413, 1604, 2306});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[55];
	}
	ind.assign({1464, 1511, 1558, 1605, 1652, 2354});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[56];
	}
	ind.assign({1704, 1751, 1846, 1989, 2180, 2402});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[57];
	}
	ind.assign({1752, 1799, 1894, 2037, 2228, 2450});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[58];
	}
	ind.assign({1848, 1895, 1942, 2085, 2276, 2498});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[59];
	}
	ind.assign({1992, 2039, 2086, 2133, 2324, 2546});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[60];
	}
	ind.assign({2184, 2231, 2278, 2325, 2372, 2594});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[61];
	}
	ind.assign({2424, 2471, 2518, 2565, 2612, 2642});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[62];
	}
	ind.assign({29, 76, 219, 506, 985, 1683});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[63];
	}
	ind.assign({77, 124, 267, 554, 1033, 1731});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[64];
	}
	ind.assign({125, 172, 315, 602, 1081, 1779});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[65];
	}
	ind.assign({221, 268, 363, 650, 1129, 1827});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[66];
	}
	ind.assign({269, 316, 411, 698, 1177, 1875});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[67];
	}
	ind.assign({365, 412, 459, 746, 1225, 1923});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[68];
	}
	ind.assign({509, 556, 651, 794, 1273, 1971});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[69];
	}
	ind.assign({557, 604, 699, 842, 1321, 2019});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[70];
	}
	ind.assign({653, 700, 747, 890, 1369, 2067});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[71];
	}
	ind.assign({797, 844, 891, 938, 1417, 2115});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[72];
	}
	ind.assign({989, 1036, 1131, 1274, 1465, 2163});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[73];
	}
	ind.assign({1037, 1084, 1179, 1322, 1513, 2211});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[74];
	}
	ind.assign({1133, 1180, 1227, 1370, 1561, 2259});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[75];
	}
	ind.assign({1277, 1324, 1371, 1418, 1609, 2307});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[76];
	}
	ind.assign({1469, 1516, 1563, 1610, 1657, 2355});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[77];
	}
	ind.assign({1709, 1756, 1851, 1994, 2185, 2403});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[78];
	}
	ind.assign({1757, 1804, 1899, 2042, 2233, 2451});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[79];
	}
	ind.assign({1853, 1900, 1947, 2090, 2281, 2499});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[80];
	}
	ind.assign({1997, 2044, 2091, 2138, 2329, 2547});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[81];
	}
	ind.assign({2189, 2236, 2283, 2330, 2377, 2595});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[82];
	}
	ind.assign({2429, 2476, 2523, 2570, 2617, 2643});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[83];
	}
	ind.assign({34, 81, 224, 511, 990, 1684});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[84];
	}
	ind.assign({82, 129, 272, 559, 1038, 1732});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[85];
	}
	ind.assign({130, 177, 320, 607, 1086, 1780});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[86];
	}
	ind.assign({226, 273, 368, 655, 1134, 1828});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[87];
	}
	ind.assign({274, 321, 416, 703, 1182, 1876});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[88];
	}
	ind.assign({370, 417, 464, 751, 1230, 1924});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[89];
	}
	ind.assign({514, 561, 656, 799, 1278, 1972});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[90];
	}
	ind.assign({562, 609, 704, 847, 1326, 2020});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[91];
	}
	ind.assign({658, 705, 752, 895, 1374, 2068});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[92];
	}
	ind.assign({802, 849, 896, 943, 1422, 2116});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[93];
	}
	ind.assign({994, 1041, 1136, 1279, 1470, 2164});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[94];
	}
	ind.assign({1042, 1089, 1184, 1327, 1518, 2212});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[95];
	}
	ind.assign({1138, 1185, 1232, 1375, 1566, 2260});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[96];
	}
	ind.assign({1282, 1329, 1376, 1423, 1614, 2308});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[97];
	}
	ind.assign({1474, 1521, 1568, 1615, 1662, 2356});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[98];
	}
	ind.assign({1714, 1761, 1856, 1999, 2190, 2404});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[99];
	}
	ind.assign({1762, 1809, 1904, 2047, 2238, 2452});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[100];
	}
	ind.assign({1858, 1905, 1952, 2095, 2286, 2500});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[101];
	}
	ind.assign({2002, 2049, 2096, 2143, 2334, 2548});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[102];
	}
	ind.assign({2194, 2241, 2288, 2335, 2382, 2596});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[103];
	}
	ind.assign({2434, 2481, 2528, 2575, 2622, 2644});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[104];
	}
	ind.assign({39, 86, 229, 516, 995, 1685});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[105];
	}
	ind.assign({87, 134, 277, 564, 1043, 1733});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[106];
	}
	ind.assign({135, 182, 325, 612, 1091, 1781});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[107];
	}
	ind.assign({231, 278, 373, 660, 1139, 1829});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[108];
	}
	ind.assign({279, 326, 421, 708, 1187, 1877});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[109];
	}
	ind.assign({375, 422, 469, 756, 1235, 1925});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[110];
	}
	ind.assign({519, 566, 661, 804, 1283, 1973});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[111];
	}
	ind.assign({567, 614, 709, 852, 1331, 2021});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[112];
	}
	ind.assign({663, 710, 757, 900, 1379, 2069});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[113];
	}
	ind.assign({807, 854, 901, 948, 1427, 2117});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[114];
	}
	ind.assign({999, 1046, 1141, 1284, 1475, 2165});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[115];
	}
	ind.assign({1047, 1094, 1189, 1332, 1523, 2213});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[116];
	}
	ind.assign({1143, 1190, 1237, 1380, 1571, 2261});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[117];
	}
	ind.assign({1287, 1334, 1381, 1428, 1619, 2309});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[118];
	}
	ind.assign({1479, 1526, 1573, 1620, 1667, 2357});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[119];
	}
	ind.assign({1719, 1766, 1861, 2004, 2195, 2405});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[120];
	}
	ind.assign({1767, 1814, 1909, 2052, 2243, 2453});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[121];
	}
	ind.assign({1863, 1910, 1957, 2100, 2291, 2501});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[122];
	}
	ind.assign({2007, 2054, 2101, 2148, 2339, 2549});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[123];
	}
	ind.assign({2199, 2246, 2293, 2340, 2387, 2597});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[124];
	}
	ind.assign({2439, 2486, 2533, 2580, 2627, 2645});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[125];
	}
	ind.assign({44, 91, 234, 521, 1000, 1686});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[126];
	}
	ind.assign({92, 139, 282, 569, 1048, 1734});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[127];
	}
	ind.assign({140, 187, 330, 617, 1096, 1782});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[128];
	}
	ind.assign({236, 283, 378, 665, 1144, 1830});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[129];
	}
	ind.assign({284, 331, 426, 713, 1192, 1878});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[130];
	}
	ind.assign({380, 427, 474, 761, 1240, 1926});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[131];
	}
	ind.assign({524, 571, 666, 809, 1288, 1974});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[132];
	}
	ind.assign({572, 619, 714, 857, 1336, 2022});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[133];
	}
	ind.assign({668, 715, 762, 905, 1384, 2070});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[134];
	}
	ind.assign({812, 859, 906, 953, 1432, 2118});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[135];
	}
	ind.assign({1004, 1051, 1146, 1289, 1480, 2166});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[136];
	}
	ind.assign({1052, 1099, 1194, 1337, 1528, 2214});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[137];
	}
	ind.assign({1148, 1195, 1242, 1385, 1576, 2262});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[138];
	}
	ind.assign({1292, 1339, 1386, 1433, 1624, 2310});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[139];
	}
	ind.assign({1484, 1531, 1578, 1625, 1672, 2358});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[140];
	}
	ind.assign({1724, 1771, 1866, 2009, 2200, 2406});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[141];
	}
	ind.assign({1772, 1819, 1914, 2057, 2248, 2454});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[142];
	}
	ind.assign({1868, 1915, 1962, 2105, 2296, 2502});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[143];
	}
	ind.assign({2012, 2059, 2106, 2153, 2344, 2550});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[144];
	}
	ind.assign({2204, 2251, 2298, 2345, 2392, 2598});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[145];
	}
	ind.assign({2444, 2491, 2538, 2585, 2632, 2646});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[146];
	}
	ind.assign({239, 526, 1005, 1687});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[147];
	}
	ind.assign({287, 574, 1053, 1735});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[148];
	}
	ind.assign({335, 622, 1101, 1783});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[149];
	}
	ind.assign({383, 670, 1149, 1831});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[150];
	}
	ind.assign({431, 718, 1197, 1879});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[151];
	}
	ind.assign({479, 766, 1245, 1927});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[152];
	}
	ind.assign({671, 814, 1293, 1975});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[153];
	}
	ind.assign({719, 862, 1341, 2023});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[154];
	}
	ind.assign({767, 910, 1389, 2071});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[155];
	}
	ind.assign({911, 958, 1437, 2119});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[156];
	}
	ind.assign({1151, 1294, 1485, 2167});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[157];
	}
	ind.assign({1199, 1342, 1533, 2215});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[158];
	}
	ind.assign({1247, 1390, 1581, 2263});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[159];
	}
	ind.assign({1391, 1438, 1629, 2311});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[160];
	}
	ind.assign({1583, 1630, 1677, 2359});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[161];
	}
	ind.assign({1871, 2014, 2205, 2407});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[162];
	}
	ind.assign({1919, 2062, 2253, 2455});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[163];
	}
	ind.assign({1967, 2110, 2301, 2503});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[164];
	}
	ind.assign({2111, 2158, 2349, 2551});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[165];
	}
	ind.assign({2303, 2350, 2397, 2599});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[166];
	}
	ind.assign({2543, 2590, 2637, 2647});
	for (int i = 0; i < ind.size(); i++)
	{
		M(ind[i]) = c[167];
	}

	M.resize(48, 56);

	Matrix<floatPrec, Dynamic,Dynamic> Mr = M.topLeftCorner(48, 48).inverse() * M.topRightCorner(48, 8);

	Matrix<floatPrec, 8,8> A;
	A.setZero();
	A(0, 5) = 1;
	A.row(1) = -Mr.row(45).rowwise().reverse();
	A.row(2) = -Mr.row(41).rowwise().reverse();
	A.row(3) = -Mr.row(38).rowwise().reverse();
	A.row(4) = -Mr.row(36).rowwise().reverse();
	A.row(5) = -Mr.row(35).rowwise().reverse();
	A.row(6) = -Mr.row(30).rowwise().reverse();
	A.row(7) = -Mr.row(26).rowwise().reverse();

	EigenSolver<Matrix<floatPrec, Dynamic,Dynamic>> es(A);
	Matrix<complex<floatPrec>, Dynamic,Dynamic> V = es.eigenvectors();

	Matrix<complex<floatPrec>, Dynamic,Dynamic> Vr = V.topRows(6).colwise().reverse().colwise().hnormalized();

	Matrix<complex<floatPrec>, Dynamic,Dynamic> sol = Vr;
	ind.clear();
	for (int i = 0; i < sol.cols(); i++)
	{
		if (isnan(sol(0, i).real()) || imag(sol(0, i)) != 0)
		{
			ind.push_back(i);
		}
	}

	for (int j = 0; j < ind.size(); j++)
	{
		removeColumn<floatPrec>(sol, ind[j] - j);
	}

	Matrix<floatPrec, Dynamic,Dynamic> sol_real = sol.real();

	Matrix<floatPrec, Dynamic,1> b1, b2, b3, b4, b5;
	b1 = sol_real.row(0);
	b.push_back(b1);
	b2 = sol_real.row(1);
	b.push_back(b2);
	b3 = sol_real.row(2);
	b.push_back(b3);
	b4 = sol_real.row(3);
	b.push_back(b4);
	b5 = sol_real.row(4);
	b.push_back(b5);

	return b;
}

template <typename floatPrec>
void removeColumn(Eigen::Matrix<complex<floatPrec>, Dynamic,Dynamic> &matrix, int colToRemove)
{
	unsigned int numRows = matrix.rows();
	unsigned int numCols = matrix.cols() - 1;

	if (colToRemove < numCols)
		matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.block(0, colToRemove + 1, numRows, numCols - colToRemove);

	matrix.conservativeResize(numRows, numCols);
}