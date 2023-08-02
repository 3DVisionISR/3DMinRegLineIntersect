// #include <iostream>
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

#include <iostream>
#include <vector>
#include <complex>
#include <utils.hpp>

using namespace Eigen;
using namespace std;

// Instantiate functions for the supported template type parameters
template Matrix<float, 4, 4> getGTTransformations<float>(bool verbose);
template Matrix<double, 4, 4> getGTTransformations<double>(bool verbose);
//
template Matrix<float, 3, 3> getSkew<float>(Matrix<float, 3, 1> v);
template Matrix<double, 3, 3> getSkew<double>(Matrix<double, 3, 1> v);
//
template std::vector<float> o4_roots<float>(const Eigen::Matrix<float, 5, 1> &p);
template std::vector<double> o4_roots<double>(const Eigen::Matrix<double, 5, 1> &p);

// -------------------------------------------------------------
// get the antisymetric matrixfrom v
// -------------------------------------------------------------
template <typename floatPrec>
Matrix<floatPrec, 3, 3> getSkew(Matrix<floatPrec, 3, 1> v)
{

  Matrix<floatPrec, 3, 3> sV;
  sV << 0, -v(2), v(1),
      v(2), 0, -v(0),
      -v(1), v(0), 0;

  return sV;
}

// -------------------------------------------------------------
// get a randomly generated transformation matrix
// -------------------------------------------------------------
template <typename floatPrec>
Matrix<floatPrec, 4, 4> getGTTransformations(bool verbose)
{

  Matrix<floatPrec, 4, 4> transOut;

  // init the output
  transOut = Matrix<floatPrec, 4, 4>::Identity(4, 4);

  floatPrec angX, angY, angZ;
  Matrix<floatPrec, 3, 3> rotGT;
  Matrix<floatPrec, 3, 1> tranGT;

  // generate random angles
  angX = (rand() % 200) / 5.0;
  angY = (rand() % 200) / 5.0;
  angZ = (rand() % 200) / 5.0;
  if (verbose)
    cout << "Rand() angles = (" << angX << "," << angY << "," << angZ << ")" << endl;

  // create a rotation matrix
  rotGT = AngleAxis<floatPrec>(angZ * M_PI / 360, Matrix<floatPrec, 3, 1>::UnitZ()) *
          AngleAxis<floatPrec>(angY * M_PI / 360, Matrix<floatPrec, 3, 1>::UnitY()) *
          AngleAxis<floatPrec>(angX * M_PI / 360, Matrix<floatPrec, 3, 1>::UnitX());
  if (verbose)
    cout << "GT Rotation Matrix: " << endl
         << rotGT << endl;
  // cout << rotGT.transpose()*rotGT << endl;

  // create a translation vector
  tranGT(0) = rand() % 200 / 10.0 - 10.0;
  tranGT(1) = rand() % 200 / 10.0 - 10.0;
  tranGT(2) = rand() % 200 / 10.0 - 10.0;
  if (verbose)
    cout << "GT Translation Vector: " << endl
         << tranGT.transpose() << endl;

  transOut.topLeftCorner(3, 3) = rotGT;
  transOut.topRightCorner(3, 1) = tranGT;

  return transOut;
}

//
// -------------------------------------------------------------
// Compute the roots of a four deg polynomial equation (must use double)
// -------------------------------------------------------------
template <typename floatPrec>
vector<floatPrec> o4_roots(const Matrix<floatPrec, 5, 1> &p)
{

  vector<floatPrec> realRoots;
  complex<double> temp;

  double A = (double)p(0, 0);
  double B = (double)p(1, 0);
  double C = (double)p(2, 0);
  double D = (double)p(3, 0);
  double E = (double)p(4, 0);

  double A_pw2 = A * A;
  double B_pw2 = B * B;
  double A_pw3 = A_pw2 * A;
  double B_pw3 = B_pw2 * B;
  double A_pw4 = A_pw3 * A;
  double B_pw4 = B_pw3 * B;

  double alpha = -3 * B_pw2 / (8 * A_pw2) + C / A;
  double beta = B_pw3 / (8 * A_pw3) - B * C / (2 * A_pw2) + D / A;
  double gamma = -3 * B_pw4 / (256 * A_pw4) + B_pw2 * C / (16 * A_pw3) - B * D / (4 * A_pw2) + E / A;

  double alpha_pw2 = alpha * alpha;
  double alpha_pw3 = alpha_pw2 * alpha;

  complex<double> P(-alpha_pw2 / 12.0 - gamma, 0);
  complex<double> Q(-alpha_pw3 / 108.0 + alpha * gamma / 3 - pow(beta, 2) / 8, 0);
  complex<double> R = -Q / 2.0 + sqrt(pow(Q, 2.0) / 4.0 + pow(P, 3.0) / 27.0);

  complex<double> U = pow(R, (1.0 / 3.0));
  complex<double> y;

  if (U.real() == 0)
    y = -5.0 * alpha / 6.0 - pow(Q, (1.0 / 3.0));
  else
    y = -5.0 * alpha / 6.0 - P / (3.0 * U) + U;

  complex<double> w = sqrt(alpha + 2.0 * y);

  temp = (floatPrec)-B / (4.0 * A) + 0.5 * (w + sqrt(-(3.0 * alpha + 2.0 * y + 2.0 * beta / w)));
  if (abs(temp.imag()) < 0.000001)
    realRoots.push_back((float)temp.real());

  temp = (floatPrec)-B / (4.0 * A) + 0.5 * (w - sqrt(-(3.0 * alpha + 2.0 * y + 2.0 * beta / w)));
  if (abs(temp.imag()) < 0.000001)
    realRoots.push_back((float)temp.real());

  temp = (floatPrec)-B / (4.0 * A) + 0.5 * (-w + sqrt(-(3.0 * alpha + 2.0 * y - 2.0 * beta / w)));
  if (abs(temp.imag()) < 0.000001)
    realRoots.push_back((float)temp.real());

  temp = (floatPrec)-B / (4.0 * A) + 0.5 * (-w - sqrt(-(3.0 * alpha + 2.0 * y - 2.0 * beta / w)));
  if (abs(temp.imag()) < 0.000001)
    realRoots.push_back((float)temp.real());

  return realRoots;
}
