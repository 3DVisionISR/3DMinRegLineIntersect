#ifndef GENDATA_HPP
#define GENDATA_HPP

#define POINTSWITHINCUBE 40
#define SETMAXSHOW 5

#include <Eigen/Core>

template <typename floatPrec>
std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>> genPointPair(Eigen::Matrix<floatPrec,4,4> T){
    
    // get a random 3D point (in homogenous coordinates)
    Eigen::Matrix<floatPrec,3,1> P = Eigen::Matrix<floatPrec, 3, 1>::Random(3) * int(POINTSWITHINCUBE);
    // requires as input the pose between the two 3D scans,
    // to apply the respective transformation to the
    // point in RefB
    
    Eigen::Matrix<floatPrec,4,1> Q = T * P.homogeneous();

    return std::make_pair(P.homogeneous(),Q);
}

template std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>> genPointPair<float>(Eigen::Matrix<float,4,4> T);
template std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>> genPointPair<double>(Eigen::Matrix<double,4,4> T);

template <typename floatPrec>
std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,
    std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>
    genLineIntPair(Eigen::Matrix<floatPrec,4,4> T){

    Eigen::Matrix<floatPrec, 3, 1> dir;

    // first set of lines in the refA
    Eigen::Matrix<floatPrec,3,1> P1 = Eigen::Matrix<floatPrec, 3, 1>::Random(3) * int(POINTSWITHINCUBE);
    Eigen::Matrix<floatPrec,3,1> P2 = Eigen::Matrix<floatPrec, 3, 1>::Random(3) * int(POINTSWITHINCUBE);

    // second set of lines in the refB
    Eigen::Matrix<floatPrec,3,1> Q1 = Eigen::Matrix<floatPrec, 3, 1>::Random(3) * int(POINTSWITHINCUBE);
    dir = P2 - Q1;
    dir /= dir.norm();
    Eigen::Matrix<floatPrec,3,1>Q2 = Q1 + (rand() % int(POINTSWITHINCUBE)) * dir;

    // requires as input the pose between the two 3D scans,
    // in order to define the coordinates in two different refs
    return std::make_pair(
            std::make_pair(P1.homogeneous(),P2.homogeneous()),
            std::make_pair(T*Q1.homogeneous(),T*Q2.homogeneous()));
}

template std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,
    std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>
    genLineIntPair<float>(Eigen::Matrix<float,4,4> T);
template std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,
    std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>
    genLineIntPair<double>(Eigen::Matrix<double,4,4> T);

template <typename floatPrec>
std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>> 
    genPlanePair(Eigen::Matrix<floatPrec,4,4> T){
    
    Eigen::Matrix<floatPrec, 3, 4> nullMatrix;
    Eigen::FullPivLU<Eigen::Matrix<floatPrec, Eigen::Dynamic, Eigen::Dynamic>> lu;

    // the null space of the matrix built from threee
    // randomly generated 3D points in homogenous coordinates
    nullMatrix = Eigen::Matrix<floatPrec, 3, 4>::Ones(3, 4);
    
    nullMatrix.topLeftCorner(3, 3) =
        Eigen::Matrix<floatPrec, 3, 3>::Random(3, 3) * int(POINTSWITHINCUBE);

    lu.compute(nullMatrix);
    Eigen::Matrix<floatPrec, 4,1> Pi = lu.kernel();
    Pi /= Pi.head(3).norm();

    // requires the pose from the two scans, to compute
    // the coordinates of the plane in RefB
    Eigen::Matrix<floatPrec, 4,1> Nu = T.transpose().inverse() * Pi;

    return std::make_pair(Pi,Nu);
}

template std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>> 
    genPlanePair<float>(Eigen::Matrix<float,4,4> T);
template std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>> 
    genPlanePair<double>(Eigen::Matrix<double,4,4> T);


template <typename floatPrec>
std::pair<std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>,
    std::pair<Eigen::Matrix<floatPrec,4,1>, Eigen::Matrix<floatPrec,4,1>>>
    genLineCorrPair(Eigen::Matrix<floatPrec,4,4> T){

    // first set of lines in the refA
    Eigen::Matrix<floatPrec,3,1> P1 = Eigen::Matrix<floatPrec, 3, 1>::Random(3) * int(POINTSWITHINCUBE);
    Eigen::Matrix<floatPrec,3,1> P2 = Eigen::Matrix<floatPrec, 3, 1>::Random(3) * int(POINTSWITHINCUBE);
    
    // requires as input the pose between the two 3D scans,
    // in order to define the coordinates in two different refs
    return std::make_pair(
            std::make_pair(P1.homogeneous(),P2.homogeneous()),
            std::make_pair(T*P1.homogeneous(),T*P2.homogeneous()));
}

template std::pair<std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>,
    std::pair<Eigen::Matrix<float,4,1>, Eigen::Matrix<float,4,1>>>
    genLineCorrPair<float>(Eigen::Matrix<float,4,4> T);
template std::pair<std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>,
    std::pair<Eigen::Matrix<double,4,1>, Eigen::Matrix<double,4,1>>>
    genLineCorrPair<double>(Eigen::Matrix<double,4,4> T);


#endif