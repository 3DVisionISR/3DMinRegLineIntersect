#include <iostream>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

#include <cmath>

#include <vector>
#include <complex>
#include <solversPoints.hpp>
#include <utils.hpp>

using namespace Eigen;
using namespace std;

// solver for the three points 3D registration solver
// instanciate float/double options
template vector<Matrix<float,4,4>> solver3Q<float>(
    vector<pair<Matrix<float,4,1>, Matrix<float,4,1>>> ptPair,
	vector<pair<Matrix<float,4,1>, Matrix<float,4,1>>> plPair,
	vector<pair<pair<Matrix<float,4,1>, Matrix<float,4,1>>,pair<Matrix<float,4,1>, Matrix<float,4,1>>>> lPair
);
template vector<Matrix<double,4,4>> solver3Q<double>(
    vector<pair<Matrix<double,4,1>, Matrix<double,4,1>>> ptPair,
	vector<pair<Matrix<double,4,1>, Matrix<double,4,1>>> plPair,
	vector<pair<pair<Matrix<double,4,1>, Matrix<double,4,1>>,pair<Matrix<double,4,1>, Matrix<double,4,1>>>> lPair
);

template <typename floatPrec>
vector<Matrix<floatPrec,4,4>> solver3Q(
    vector<pair<Matrix<floatPrec,4,1>, Matrix<floatPrec,4,1>>> ptPair,
	vector<pair<Matrix<floatPrec,4,1>, Matrix<floatPrec,4,1>>> plPair,
	vector<pair<pair<Matrix<floatPrec,4,1>, Matrix<floatPrec,4,1>>,pair<Matrix<floatPrec,4,1>, Matrix<floatPrec,4,1>>>> lPair
){

    vector<Matrix<floatPrec, 4,4>> estSolutions;

    // checks if we get the required number of input
    // point correspondences
    if (ptPair.size() < 3){
        cerr << "[3Q ERR]: Not enough input data" << endl;
        return estSolutions;
    }

    Matrix<floatPrec, 4,1> p11 = ptPair[0].first;
    Matrix<floatPrec, 4,1> p21 = ptPair[0].second; 
    Matrix<floatPrec, 4,1> p12 = ptPair[1].first; 
    Matrix<floatPrec, 4,1> p22 = ptPair[1].second; 
    Matrix<floatPrec, 4,1> p13 = ptPair[2].first; 
    Matrix<floatPrec, 4,1> p23 = ptPair[2].second;

    // gets the predefined transformations
    // first and second cameras
    Matrix<floatPrec, 4,4> H1, H2;
 
    // H1 = [eye(3) -p11; 0 0 0 1];
    // H2 = [eye(3) -p21; 0 0 0 1];
    H1.setIdentity(4,4);
    H2.setIdentity(4,4);
    H1.topRightCorner(3,1) = -p11.head(3);
    H2.topRightCorner(3,1) = -p21.head(3);


    // apply the predefined transformations
    // predefined rotations. We want a rotation that aligns bar(p1,p2) with the z-axis
    Matrix<floatPrec, 3,1> r1, r2, r3; // columns of the rotation matrix
    Matrix<floatPrec, 3,1> ey(0,1,0), ez(0,0,1);
    Matrix<floatPrec, 3,3> Rez, Rfz;
    Matrix<floatPrec, 4,4> G1, G2;

    G1.setIdentity(4,4);
    G2.setIdentity(4,4);

    r3 = p12.head(3)-p11.head(3); r3 /= r3.norm();
    if ((ez.cross(r3)).norm() > (ey.cross(r3)).norm())
        r2 = ez.cross(r3);
    else r2 = ey.cross(r3);
    r2 /= r2.norm();
    r1 = r3.cross(r2);
    Rez.col(0) = r1;
    Rez.col(1) = r2;
    Rez.col(2) = r3;
    Rez *= Rez.determinant();
    G1.topLeftCorner(3,3) = Rez.transpose();

    r3 = p22.head(3)-p21.head(3); r3 /= r3.norm();
    if (ez.cross(r3).norm() > ey.cross(r3).norm()) r2 = ez.cross(r3);
    else r2 = ey.cross(r3);
    r2 /= r2.norm();
    r1 = r3.cross(r2);
    Rfz.col(0) = r1;
    Rfz.col(1) = r2;
    Rfz.col(2) = r3;
    Rfz *= Rfz.determinant();
    G2.topLeftCorner(3,3) = Rfz.transpose();

    // compute the predefined transformations
    Matrix<floatPrec, 4,4> aT1 = G1*H1;
    Matrix<floatPrec, 4,4> aT2 = G2*H2;

    // compute points in the new coordinate systems
    Matrix<floatPrec, 4,1> tp13 = aT1*p13;
    Matrix<floatPrec, 4,1> tp23 = aT2*p23;

    // Solve the pose
    floatPrec q21, q22, q23, q31, q32, q33;
    q21 = tp13(0); q22 = tp13(1); q23 = tp13(2);
    q31 = tp23(0); q32 = tp23(1); q33 = tp23(2);
    // solutions
    floatPrec a1, a2, b1, b2;
    a1 = (q32 - (q21*(q21*q32 + q22*pow(q21*q21 + q22*q22 - q32*q32, 0.5)))/(q21*q21 + q22*q22))/q22;
    a2 = (q32 - (q21*(q21*q32 - q22*pow(q21*q21 + q22*q22 - q32*q32, 0.5)))/(q21*q21 + q22*q22))/q22;
    b1 = (q21*q32 + q22*pow(q21*q21 + q22*q22 - q32*q32, 0.5))/(q21*q21 + q22*q22);
    b2 = (q21*q32 - q22*pow(q21*q21 + q22*q22 - q32*q32, 0.5))/(q21*q21 + q22*q22);

    Matrix<floatPrec, 4,4> L1, L2;
    L1.setIdentity(4,4);
    L1(0,0) = a1; L1(0,1) = -b1;    
    L1(1,0) = b1; L1(1,1) =  a1;    
    
    L2.setIdentity(4,4);
    L2(0,0) = a2; L2(0,1) = -b2;    
    L2(1,0) = b2; L2(1,1) =  a2;

    estSolutions.push_back(aT2.inverse()*L1*aT1);
    estSolutions.push_back(aT2.inverse()*L2*aT1);

    return estSolutions;
}