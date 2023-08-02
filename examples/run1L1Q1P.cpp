#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdlib>
#include <chrono>

#include <solvers.hpp>
#include <utils.hpp>
#include <genData.hpp>


// solvers...
using namespace Eigen;
using namespace std;
using namespace chrono;

int main(){
    bool verbose = true;
    high_resolution_clock::time_point t1, t2;

    // Intializes random number generator
    time_t t;
    srand((unsigned) time(&t));
    vector<double> timeVector;

    int powIter = 0;
	
    if (powIter > 1 && verbose)
	{
		std::cerr << "ERR: check the number of iterations and the verbose option..." << std::endl;
		return 0;
	}

    vector<unsigned int> numSols;
    vector<double> transErrors;
    vector<double> rotErrors;
    Matrix<double, 4,4> transGT;
    std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> ptPairs;
	std::vector<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>> plPairs;
	std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lPairs;
    std::vector<std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>> lCPairs;
    for (int iter = 0 ; iter < pow(10,powIter) ; iter++){ 
	    transGT = getGTTransformations<double>();

        lPairs.push_back(genLineIntPair<double>(transGT));
        ptPairs.push_back(genPointPair<double>(transGT));
        plPairs.push_back(genPlanePair<double>(transGT));

        if(verbose){
            cout << "GT transformation: " << endl;
            cout << transGT << endl;
        }

        // test solver
	    t1 = high_resolution_clock::now();
	    // run solver
        vector<Matrix<double, 4,4>> estTrans = solver1L1Q1P<double>(ptPairs,plPairs,lPairs,lCPairs);
	    t2 = high_resolution_clock::now();
	    auto duration = duration_cast<microseconds>( t2 - t1 ).count();

        Matrix<double, 4,4> estT;
        Matrix<double, 3,3> R;
        double transError = 1000;
        double rotError = 1000;
        AngleAxis<double> axAng;
        for(int iiter= 0; iiter < estTrans.size(); iiter++){
                estT = estTrans[iiter];
                if( (estT.topRightCorner(3,1) - transGT.topRightCorner(3,1)).norm() < transError)
                    transError = (estT.topRightCorner(3,1) - transGT.topRightCorner(3,1)).norm();
                R = estT.topLeftCorner(3,3)*transGT.topLeftCorner(3,3).transpose();
                axAng.fromRotationMatrix(R);
                if( abs(axAng.angle()) < rotError)
                    rotError = abs(axAng.angle());
                if (verbose) cout << "Estimated Transformation:" << endl << estT << endl;
	            if (verbose) cout << "Error (frob norm): " << (transGT-estT).norm() << ", time: " << duration << endl;        
        }         

    }

    return 0;
}