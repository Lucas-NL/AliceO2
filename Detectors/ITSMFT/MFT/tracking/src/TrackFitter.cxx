// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFitter.cxx
/// \brief Implementation of a class to fit a track to a set of clusters
///
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#include "MFTBase/Constants.h"
#include "MFTTracking/TrackFitter.h"
#include "MFTTracking/TrackCA.h"
#include "MFTTracking/TrackExtrap.h"
#include "MFTTracking/Cluster.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "ITSMFTReconstruction/ChipMappingMFT.h"
#include <stdexcept>
#include <TMath.h>
#include <TMatrixD.h>
#include <TF1.h>
#include <TF2.h>
#include "CommonConstants/MathConstants.h"
#include "MathUtils/MathBase.h"
#include "MathUtils/Utils.h"

using o2::math_utils::math_base::fitGaus;

namespace o2
{
namespace mft
{

//_________________________________________________________________________________________________
void TrackFitter::setBz(float bZ)
{
  auto& mftTrackingParam = MFTTrackingParam::Instance();

  /// Set the magnetic field for the MFT
  mBZField = bZ;
  mTrackExtrap.setBz(bZ);

  if (mftTrackingParam.verbose) {
    LOG(INFO) << "Setting Fitter field = " << bZ;
  }
}

//_________________________________________________________________________________________________
bool TrackFitter::fit(FitterTrackMFT& track, bool smooth, bool finalize,
                      std::list<TrackParamMFT>::reverse_iterator* itStartingParam)
{
  /// Fit a track to its attached clusters
  /// Smooth the track if requested and the smoother enabled
  /// If finalize = true: copy the smoothed parameters, if any, into the regular ones
  /// Fit the entire track or only the part upstream itStartingParam
  /// Returns false in case of failure

  auto& mftTrackingParam = MFTTrackingParam::Instance();

  if (mftTrackingParam.verbose) {
    std::cout << "\n ***************************** Start Fitting new track ***************************** \n";
    std::cout << "N Clusters = " << track.getNPoints() << std::endl;
  }

  // initialize the starting track parameters and cluster
  double chi2invqptquad;
  double invpqtseed;

  auto itParam(track.rbegin());
  if (itStartingParam != nullptr) {
    // use the ones pointed to by itStartingParam
    if (*itStartingParam == track.rend()) {
      LOG(INFO) << "MFT Fit ERROR: invalid track starting parameters";
      track.removable();
      return false;
    }
    itParam = *itStartingParam;
  } else {
    // or start from the last cluster and compute the track parameters from its position
    // and the one of the first previous cluster found on a different layer
    auto itPreviousParam(itParam);
    ++itPreviousParam;
    invpqtseed = invQPtFromFCF(track, mBZField, chi2invqptquad);//mftTrackingParam.seedQPt ? invQPtFromFCF(track, mBZField, chi2invqptquad) : invQPtFromParabola(track, mBZField, chi2invqptquad);
    track.setInvQPtSeed(invpqtseed);
    track.setChi2QPtQuadtratic(chi2invqptquad);
    (*itParam).setInvQPt(track.getInvQPtSeed()); // Initial momentum estimate
    initTrack(*itParam->getClusterPtr(), *itParam ,*track.first().getClusterPtr());
  }

  if (mftTrackingParam.verbose) {
    std::cout << "Seed covariances:";
    itParam->getCovariances().Print();
  }

  // recusively add the upstream clusters and update the track parameters
  TrackParamMFT* startingParam = &*itParam;
  while (++itParam != track.rend()) {
    if (!addCluster(*startingParam, *itParam->getClusterPtr(), *itParam)) {
      track.removable();
      return false;
    }
    startingParam = &*itParam;
  }

  itParam--;
  if (mftTrackingParam.verbose) {
    std::cout << "Track covariances:";
    itParam->getCovariances().Print();
    std::cout << "Track Chi2 = " << itParam->getTrackChi2() << std::endl;
    std::cout << " ***************************** Done fitting *****************************\n";
  }

  // smooth the track if requested and the smoother enabled
  if (smooth && mSmooth) {
    if (!smoothTrack(track, finalize)) {
      track.removable();
      return false;
    }
  }
  return true;
}

//_________________________________________________________________________________________________
void TrackFitter::initTrack(const Cluster& cl, TrackParamMFT& param, const Cluster& firstCl)
{

  /// Compute the initial track parameters at the z position of the last cluster (cl)
  /// The covariance matrix is computed such that the last cluster is the only constraint
  /// (by assigning an infinite dispersion to the other cluster)
  /// These parameters are the seed for the Kalman filter

  auto& mftTrackingParam = MFTTrackingParam::Instance();

  double x0 = cl.xCoordinate;
  double y0 = cl.yCoordinate;
  double z0 = cl.zCoordinate;
  double x1 = firstCl.xCoordinate;	
  double y1 = firstCl.yCoordinate;	
  double z1 = firstCl.zCoordinate;	 

  double k = 0.3 * TMath::Abs(mBZField);
  double Hz = mBZField / TMath::Abs(mBZField);
  double dx = x0 - x1;
  double dy = y0 - y1;
  double dz = z0 - z1;
  double dx2 = dx * dx;
  double dy2 = dy * dy; 
  double dr = TMath::Sqrt(dx2 + dy2);
  double dr2 = dx2 + dy2;
  double invdr = 1./dr;
  double invdr2 = 1. / dr2;
  double invdr3 = invdr2*invdr;
  double HzKdr = Hz * k * dr;

  double invqpt = param.getInvQPt();
  double Qr = invqpt * dr * k;
  double Qrsq = Qr * Qr;
  double Qi = TMath::Sqrt(Qrsq+1);
  double Q1 = Qi + 1;
  double Q1sq = Q1 * Q1;
  double sqrtQ1 = TMath::Sqrt(Q1);

  double invQi = 1. / Qi; 
  double invQisq = invQi * invQi;
  double invQ1 = 1. / Q1; 
  double invsqrtQ1 = 1. / sqrtQ1;
  double invQ1sq = invQ1 * invQ1;
  double invQ1cu = invQ1sq * invQ1;
  double QQ = 2 * Qi * Q1 - Qrsq; 
  double sqrt2 = TMath::Sqrt(2);
  // compute the track parameters at the last cluster


  double pt = TMath::Sqrt(x0 * x0 + y0 * y0);
  double pz = z0;
  double tanl = 0.5 * TMath::Sqrt(2) * (dz / dr) * TMath::Sqrt(Qi + 1);
  double phi0 = TMath::ATan2(dy, dx) - invqpt * Hz * dz * k * 0.5 / tanl;
//  double r0sq = x0 * x0 + y0 * y0;
// double r0cu = r0sq * TMath::Sqrt(r0sq);
//  double invr0sq = 1.0 / r0sq;
//  double invr0cu = 1.0 / r0cu;
  double sigmax0sq = cl.sigmaX2;
  double sigmay0sq = cl.sigmaY2;
  double sigmaDeltaZsq = 5.0;         // Primary vertex distribution: beam interaction diamond
  double sigmaboost = mftTrackingParam.sigmaboost; // Boost q/pt seed covariances
  double sigmainvqpt = sigmaboost;
  double seedH_k = mftTrackingParam.seedH_k;       // SeedH constant

  param.setX(x0);
  param.setY(y0);
  param.setZ(z0);
  param.setPhi(phi0);
  param.setTanl(tanl);

  // Configure the track seed
  switch (mftTrackingParam.seed) {
    case AB:
      if (mftTrackingParam.verbose)
        std::cout << " Init track with Seed A / B; sigmaboost = " << sigmaboost << ".\n";
      param.setInvQPt(1.0 / pt); // Seeds A & B
      break;
    case CE:
      if (mftTrackingParam.verbose)
        std::cout << " Init track with Seed C / E; sigmaboost = " << sigmaboost << ".\n";
      param.setInvQPt(std::copysign(1.0, param.getInvQPt()) / pt); // Seeds C & E
      break;
    case DH:
      if (mftTrackingParam.verbose)
        std::cout << " Init track with Seed H; (k = " << seedH_k << "); sigmaboost = " << sigmaboost << ".\n";
      param.setInvQPt(param.getInvQPt() / seedH_k); // SeedH
      break;
    default:
      if (mftTrackingParam.verbose)
        std::cout << " Init track with Seed D.\n";
      break;
  }
  if (mftTrackingParam.verbose) {
    auto model = (mftTrackingParam.trackmodel == Helix) ? "Helix" : (mftTrackingParam.trackmodel == Quadratic) ? "Quadratic" : "Linear";
    std::cout << "Track Model: " << model << std::endl;
    std::cout << "  initTrack: X = " << x0 << " Y = " << y0 << " Z = " << z0 << " Tgl = " << param.getTanl() << "  Phi = " << param.getPhi() << " pz = " << param.getPz() << " qpt = " << 1.0 / param.getInvQPt() << std::endl;
  }

  // compute the track parameter covariances at the last cluster (as if the other clusters did not exist)
  TMatrixD lastParamCov(5, 5);
  lastParamCov.Zero();
  lastParamCov(0, 0) = sigmax0sq;                   // <X,X>
  lastParamCov(0, 1) = 0;                           // <Y,X>
  lastParamCov(0, 2) = -sigmax0sq * 0.25 * invQi * invdr2 * invQ1sq * ( Hz * dx * Qr * sqrt2 * Q1 * (2 * Qi * sqrtQ1 * Q1sq - Qrsq) + 4 * Qi * dy * Q1sq);	// <PHI,X>
  lastParamCov(0, 3) = sigmax0sq * sqrt2 * dz * 0.25 * invQi * invdr3 * invsqrtQ1 * dx * (-2 * Qi *  Q1 + Qrsq); 	// <TANL,X>estudar
  lastParamCov(0, 4) = 0;    	// <INVQPT,X>

  lastParamCov(1, 1) = sigmay0sq;	// <Y,Y>
  lastParamCov(1, 2) = -sigmay0sq * 0.25 * invQi * invdr2 * invQ1sq * ( Hz * dy * Qr * sqrt2 * Q1 * (2 * Qi * sqrtQ1 * Q1sq - Qrsq) + 4 * Qi * dx * Q1sq);;       // <PHI,Y>
  lastParamCov(1, 3) = sigmay0sq * sqrt2 * dz * 0.25 * invQi * invdr3 * invsqrtQ1 * dy * (-2 * Qi * Q1 + Qrsq); 	// <TANL,Y>
  lastParamCov(1, 4) = 0;       // <INVQPT,Y>

  lastParamCov(2, 2) = sigmainvqpt * HzKdr * aux * 0.125 * invQisq * invQ1cu * QQ * QQ; // <PHI,PHI>
  lastParamCov(2, 3) = sigmainvqpt * -HzKdr * Qr * dz * k * 0.125 * invQisq * invQ1sq * QQ;      //  <TANL,PHI>
  lastParamCov(2, 4) = sigmainvqpt * -HzKdr * sqrt2 * 0.25 * invQi * invsqrtQ1 * invQ1cu * QQ;           //  <INVQPT,PHI>

  lastParamCov(3, 3) = sigmainvqpt * Qr * Qr * dz * dz * k * k * 0.125 * invQisq * invQ1;; // <TANL,TANL>
  lastParamCov(3, 4) = sigmainvqpt * Qr * sqrt2 * dz * k * 0.125 * invQi * invsqrtQ1;                                // <INVQPT,TANL>

  lastParamCov(4, 4) = sigmainvqpt; // <INVQPT,INVQPT>

  lastParamCov(1, 0) = lastParamCov(0, 1); //
  lastParamCov(2, 0) = lastParamCov(0, 2); //
  lastParamCov(2, 1) = lastParamCov(1, 2); //
  lastParamCov(3, 0) = lastParamCov(0, 3); //
  lastParamCov(3, 1) = lastParamCov(1, 3); //
  lastParamCov(3, 2) = lastParamCov(2, 3); //
  lastParamCov(4, 0) = lastParamCov(0, 4); //
  lastParamCov(4, 1) = lastParamCov(1, 4); //
  lastParamCov(4, 2) = lastParamCov(2, 4); //
  lastParamCov(4, 3) = lastParamCov(3, 4); //

  param.setCovariances(lastParamCov);

  // set other parameters
  param.setClusterPtr(&cl);
  param.setTrackChi2(0.);
}

//_________________________________________________________________________________________________
bool TrackFitter::addCluster(const TrackParamMFT& startingParam, const Cluster& cl, TrackParamMFT& param)
{
  /// Extrapolate the starting track parameters to the z position of the new cluster
  /// accounting for MCS dispersion in the current layer and the other(s) crossed
  /// Recompute the parameters adding the cluster constraint with the Kalman filter
  /// Returns false in case of failure

  auto& mftTrackingParam = MFTTrackingParam::Instance();

  if (cl.zCoordinate <= startingParam.getZ()) {
    LOG(INFO) << "AddCluster ERROR: The new cluster must be upstream! Bug on TrackFinder. ";
    return false;
  }
  if (mftTrackingParam.verbose)
    std::cout << "addCluster:     X = " << cl.xCoordinate << " Y = " << cl.yCoordinate << " Z = " << cl.zCoordinate << std::endl;
  // copy the current parameters into the new ones
  param.setParameters(startingParam.getParameters());
  param.setZ(startingParam.getZ());
  param.setCovariances(startingParam.getCovariances());
  param.setTrackChi2(startingParam.getTrackChi2());

  // add MCS effects for the new cluster
  using o2::mft::constants::LayerZPosition;
  int startingLayerID, newLayerID;

  double dZ = TMath::Abs(cl.zCoordinate - startingParam.getZ());
  //LayerID of each cluster from ZPosition // TODO: Use ChipMapping
  for (auto layer = 10; layer--;)
    if (startingParam.getZ() < LayerZPosition[layer] + .3 & startingParam.getZ() > LayerZPosition[layer] - .3)
      startingLayerID = layer;
  for (auto layer = 10; layer--;)
    if (cl.zCoordinate<LayerZPosition[layer] + .3 & cl.zCoordinate> LayerZPosition[layer] - .3)
      newLayerID = layer;
  // Number of disks crossed by this tracklet
  int NDisksMS = (startingLayerID % 2 == 0) ? (startingLayerID - newLayerID) / 2 : (startingLayerID - newLayerID + 1) / 2;

  double MFTDiskThicknessInX0 = mftTrackingParam.MFTRadLenghts / 5.0;
  if (mftTrackingParam.verbose) {
    std::cout << "startingLayerID = " << startingLayerID << " ; "
              << "newLayerID = " << newLayerID << " ; ";
    std::cout << "cl.zCoordinate = " << cl.zCoordinate << " ; ";
    std::cout << "startingParam.getZ() = " << startingParam.getZ() << " ; ";
    std::cout << "NDisksMS = " << NDisksMS << std::endl;
  }

  // Add MCS effects
  if ((NDisksMS * MFTDiskThicknessInX0) != 0)
    mTrackExtrap.addMCSEffect(&param, -1, NDisksMS * MFTDiskThicknessInX0);

  // reset propagator for smoother
  if (mSmooth) {
    param.resetPropagator();
  }

  if (mftTrackingParam.verbose)
    std::cout << "  BeforeExtrap: X = " << param.getX() << " Y = " << param.getY() << " Z = " << param.getZ() << " Tgl = " << param.getTanl() << "  Phi = " << param.getPhi() << " pz = " << param.getPz() << " qpt = " << 1.0 / param.getInvQPt() << std::endl;

  // extrapolate to the z position of the new cluster
  mTrackExtrap.extrapToZCov(&param, cl.zCoordinate, mSmooth);

  if (mftTrackingParam.verbose)
    std::cout << "   AfterExtrap: X = " << param.getX() << " Y = " << param.getY() << " Z = " << param.getZ() << " Tgl = " << param.getTanl() << "  Phi = " << param.getPhi() << " pz = " << param.getPz() << " qpt = " << 1.0 / param.getInvQPt() << std::endl;

  // save extrapolated parameters and covariances for smoother
  if (mSmooth) {
    param.setExtrapParameters(param.getParameters());
    param.setExtrapCovariances(param.getCovariances());
  }

  // recompute the parameters
  param.setClusterPtr(&cl);
  if (runKalmanFilter(param)) {
    if (mftTrackingParam.verbose) {
      std::cout << "   New Cluster: X = " << cl.xCoordinate << " Y = " << cl.yCoordinate << " Z = " << cl.zCoordinate << std::endl;
      std::cout << "   AfterKalman: X = " << param.getX() << " Y = " << param.getY() << " Z = " << param.getZ() << " Tgl = " << param.getTanl() << "  Phi = " << param.getPhi() << " pz = " << param.getPz() << " qpt = " << 1.0 / param.getInvQPt() << std::endl;
      std::cout << std::endl;
      // Outputs track covariance matrix:
      // param.getCovariances().Print();
    }
    return true;
  } else
    return false;
}

//_________________________________________________________________________________________________
bool TrackFitter::smoothTrack(FitterTrackMFT& track, bool finalize)
{
  /// Recompute the track parameters at each cluster using the Smoother
  /// Smoothed parameters are stored in dedicated data members
  /// If finalize, they are copied in the regular parameters in case of success
  /// Returns false in case of failure

  auto itCurrentParam(track.begin());
  auto itPreviousParam(itCurrentParam);
  ++itCurrentParam;

  // smoothed parameters and covariances at first cluster = filtered parameters and covariances
  itPreviousParam->setSmoothParameters(itPreviousParam->getParameters());
  itPreviousParam->setSmoothCovariances(itPreviousParam->getCovariances());

  // local chi2 at first cluster = last additional chi2 provided by Kalman
  itPreviousParam->setLocalChi2(itPreviousParam->getTrackChi2() - itCurrentParam->getTrackChi2());

  // recursively smooth the next parameters and covariances
  do {
    if (!runSmoother(*itPreviousParam, *itCurrentParam)) {
      return false;
    }
    ++itPreviousParam;
  } while (++itCurrentParam != track.end());

  // update the regular parameters and covariances if requested
  if (finalize) {
    for (auto& param : track) {
      param.setParameters(param.getSmoothParameters());
      param.setCovariances(param.getSmoothCovariances());
    }
  }
  return true;
}

//_________________________________________________________________________________________________
bool TrackFitter::runKalmanFilter(TrackParamMFT& trackParam)
{
  /// Compute the new track parameters including the attached cluster with the Kalman filter
  /// The current parameters are supposed to have been extrapolated to the cluster z position
  /// Retruns false in case of failure

  // get actual track parameters (p)
  TMatrixD param(trackParam.getParameters());

  // get new cluster parameters (m)
  const Cluster* cluster = trackParam.getClusterPtr();
  TMatrixD clusterParam(5, 1);
  clusterParam.Zero();
  clusterParam(0, 0) = cluster->xCoordinate;
  clusterParam(1, 0) = cluster->yCoordinate;

  // compute the actual parameter weight (W)
  TMatrixD paramWeight(trackParam.getCovariances());
  if (paramWeight.Determinant() != 0) {
    paramWeight.Invert();
  } else {
    LOG(INFO) << "runKalmanFilter ERROR: Determinant = 0";
    return false;
  }

  // compute the new cluster weight (U)
  TMatrixD clusterWeight(5, 5);
  clusterWeight.Zero();
  clusterWeight(0, 0) = 1. / cluster->sigmaX2; // 1. / cluster->getEx2();
  clusterWeight(1, 1) = 1. / cluster->sigmaY2; //  1. / cluster->getEy2();

  // compute the new parameters covariance matrix ((W+U)^-1)
  TMatrixD newParamCov(paramWeight, TMatrixD::kPlus, clusterWeight);
  if (newParamCov.Determinant() != 0) {
    newParamCov.Invert();
  } else {
    LOG(INFO) << "runKalmanFilter ERROR: Determinant = 0";
    return false;
  }
  trackParam.setCovariances(newParamCov);

  // compute the new parameters (p' = ((W+U)^-1)U(m-p) + p)
  TMatrixD tmp(clusterParam, TMatrixD::kMinus, param);   // m-p
  TMatrixD tmp2(clusterWeight, TMatrixD::kMult, tmp);    // U(m-p)
  TMatrixD newParam(newParamCov, TMatrixD::kMult, tmp2); // ((W+U)^-1)U(m-p)
  newParam += param;                                     // ((W+U)^-1)U(m-p) + p
  trackParam.setParameters(newParam);

  // compute the additional chi2 (= ((p'-p)^-1)W(p'-p) + ((p'-m)^-1)U(p'-m))
  tmp = newParam;                                                // p'
  tmp -= param;                                                  // (p'-p)
  TMatrixD tmp3(paramWeight, TMatrixD::kMult, tmp);              // W(p'-p)
  TMatrixD addChi2Track(tmp, TMatrixD::kTransposeMult, tmp3);    // ((p'-p)^-1)W(p'-p)
  tmp = newParam;                                                // p'
  tmp -= clusterParam;                                           // (p'-m)
  TMatrixD tmp4(clusterWeight, TMatrixD::kMult, tmp);            // U(p'-m)
  addChi2Track += TMatrixD(tmp, TMatrixD::kTransposeMult, tmp4); // ((p'-p)^-1)W(p'-p) + ((p'-m)^-1)U(p'-m)
  trackParam.setTrackChi2(trackParam.getTrackChi2() + addChi2Track(0, 0));

  return true;
}

//_________________________________________________________________________________________________
bool TrackFitter::runSmoother(const TrackParamMFT& previousParam, TrackParamMFT& param)
{
  /// Recompute the track parameters starting from the previous ones
  /// Returns false in case of failure

  // get variables
  const TMatrixD& extrapParameters = previousParam.getExtrapParameters();           // X(k+1 k)
  const TMatrixD& filteredParameters = param.getParameters();                       // X(k k)
  const TMatrixD& previousSmoothParameters = previousParam.getSmoothParameters();   // X(k+1 n)
  const TMatrixD& propagator = previousParam.getPropagator();                       // F(k)
  const TMatrixD& extrapCovariances = previousParam.getExtrapCovariances();         // C(k+1 k)
  const TMatrixD& filteredCovariances = param.getCovariances();                     // C(k k)
  const TMatrixD& previousSmoothCovariances = previousParam.getSmoothCovariances(); // C(k+1 n)

  // compute smoother gain: A(k) = C(kk) * F(k)^t * (C(k+1 k))^-1
  TMatrixD extrapWeight(extrapCovariances);
  if (extrapWeight.Determinant() != 0) {
    extrapWeight.Invert(); // (C(k+1 k))^-1
  } else {
    LOG(INFO) << "Smoother ERROR: Determinant = 0";
    return false;
  }
  TMatrixD smootherGain(filteredCovariances, TMatrixD::kMultTranspose, propagator); // C(kk) * F(k)^t
  smootherGain *= extrapWeight;                                                     // C(kk) * F(k)^t * (C(k+1 k))^-1

  // compute smoothed parameters: X(k n) = X(k k) + A(k) * (X(k+1 n) - X(k+1 k))
  TMatrixD tmpParam(previousSmoothParameters, TMatrixD::kMinus, extrapParameters); // X(k+1 n) - X(k+1 k)
  TMatrixD smoothParameters(smootherGain, TMatrixD::kMult, tmpParam);              // A(k) * (X(k+1 n) - X(k+1 k))
  smoothParameters += filteredParameters;                                          // X(k k) + A(k) * (X(k+1 n) - X(k+1 k))
  param.setSmoothParameters(smoothParameters);

  // compute smoothed covariances: C(k n) = C(k k) + A(k) * (C(k+1 n) - C(k+1 k)) * (A(k))^t
  TMatrixD tmpCov(previousSmoothCovariances, TMatrixD::kMinus, extrapCovariances); // C(k+1 n) - C(k+1 k)
  TMatrixD tmpCov2(tmpCov, TMatrixD::kMultTranspose, smootherGain);                // (C(k+1 n) - C(k+1 k)) * (A(k))^t
  TMatrixD smoothCovariances(smootherGain, TMatrixD::kMult, tmpCov2);              // A(k) * (C(k+1 n) - C(k+1 k)) * (A(k))^t
  smoothCovariances += filteredCovariances;                                        // C(k k) + A(k) * (C(k+1 n) - C(k+1 k)) * (A(k))^t
  param.setSmoothCovariances(smoothCovariances);

  // compute smoothed residual: r(k n) = cluster - X(k n)
  const Cluster* cluster = param.getClusterPtr();
  TMatrixD smoothResidual(2, 1);
  smoothResidual.Zero();
  smoothResidual(0, 0) = cluster->xCoordinate - smoothParameters(0, 0);
  smoothResidual(1, 0) = cluster->yCoordinate - smoothParameters(1, 0);

  // compute weight of smoothed residual: W(k n) = (clusterCov - C(k n))^-1
  TMatrixD smoothResidualWeight(2, 2);
  smoothResidualWeight(0, 0) = cluster->sigmaX2 - smoothCovariances(0, 0); // cluster->getEx2() - smoothCovariances(0, 0);
  smoothResidualWeight(0, 1) = -smoothCovariances(0, 2);
  smoothResidualWeight(1, 0) = -smoothCovariances(2, 0);
  smoothResidualWeight(1, 1) = cluster->sigmaY2 - smoothCovariances(2, 2); // cluster->getEy2() - smoothCovariances(2, 2);
  if (smoothResidualWeight.Determinant() != 0) {
    smoothResidualWeight.Invert();
  } else {
    LOG(INFO) << "Smoother ERROR: Determinant = 0";
    return false;
  }

  // compute local chi2 = (r(k n))^t * W(k n) * r(k n)
  TMatrixD tmpChi2(smoothResidual, TMatrixD::kTransposeMult, smoothResidualWeight); // (r(k n))^t * W(k n)
  TMatrixD localChi2(tmpChi2, TMatrixD::kMult, smoothResidual);                     // (r(k n))^t * W(k n) * r(k n)
  param.setLocalChi2(localChi2(0, 0));
  return true;
}

//__________________________________________________________________________
Double_t invQPtFromParabola(const FitterTrackMFT& track, Double_t bFieldZ, Double_t& chi2)
{
  //rotate track to stabilize quadratic fitting
  auto deltax = track.rbegin()->getX() - track.first().getX();
  auto deltay = track.rbegin()->getY() - track.first().getY();
  auto x_m = (track.rbegin()->getX() + track.first().getX()) / 2;
  auto y_m = (track.rbegin()->getY() + track.first().getY()) / 2;
  auto theta = -TMath::ATan2(deltay, deltax);
  auto costheta = TMath::Cos(theta), sintheta = TMath::Sin(theta);

  bool verbose = false;
  if (verbose) {
    std::cout << "First and last cluster X,Y => " << track.first().getX() << " , " << track.first().getY() << "     /  " << track.rbegin()->getX() << " , " << track.rbegin()->getY() << std::endl;
    std::cout << " Angle to rotate: " << theta << " ( " << theta * TMath::RadToDeg() << " deg ) " << std::endl;
  }

  auto nPoints = track.getNClusters();
  Double_t* x = new Double_t[nPoints];
  Double_t* y = new Double_t[nPoints];
  int n = 0;
  for (auto trackparam = track.begin(); trackparam != track.end(); trackparam++) {
    auto x_0 = trackparam->getClusterPtr()->xCoordinate - x_m;
    auto y_0 = trackparam->getClusterPtr()->yCoordinate - y_m;
    x[n] = x_0 * costheta - y_0 * sintheta;
    y[n] = x_0 * sintheta + y_0 * costheta;
    //std::cout << "    adding rotated point to fit at z = " << trackparam->getClusterPtr()->getZ() << " (" << x[n] <<  "," << y[n]  <<  ") "<< std::endl;
    n++;
  }

  Double_t q0, q1, q2;
  chi2 = QuadraticRegression(nPoints, x, y, q0, q1, q2);
  Double_t radiusParabola = 0.5 / q2;
  auto invqpt_parabola = q2 / (o2::constants::math::B2C * bFieldZ * 0.5); // radiusParabola; // radius = 0.5/q2

  if (verbose) {
    std::cout << "--------------------------------------------" << std::endl;
    std::cout << " Fit QuadraticRegression: " << std::endl;
    std::cout << " Fit Parameters [0] = " << q0 << " [1] =  " << q1 << " [2] = " << q2 << std::endl;
    std::cout << " Radius from QuadraticRegression = " << 0.5 / q2 << std::endl;
    std::cout << " Seed qpt = " << 1.0 / invqpt_parabola << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
  }

  return invqpt_parabola;
}

//__________________________________________________________________________
Double_t QuadraticRegression(Int_t nVal, Double_t* xVal, Double_t* yVal, Double_t& p0, Double_t& p1, Double_t& p2)
{
  /// Perform a Quadratic Regression
  /// Assume same error on all clusters = 1
  /// Return ~ Chi2

  TMatrixD y(nVal, 1);
  TMatrixD x(nVal, 3);
  TMatrixD xtrans(3, nVal);

  for (int i = 0; i < nVal; i++) {
    y(i, 0) = yVal[i];
    x(i, 0) = 1.;
    x(i, 1) = xVal[i];
    x(i, 2) = xVal[i] * xVal[i];
    xtrans(0, i) = 1.;
    xtrans(1, i) = xVal[i];
    xtrans(2, i) = xVal[i] * xVal[i];
  }
  TMatrixD tmp(xtrans, TMatrixD::kMult, x);
  tmp.Invert();

  TMatrixD tmp2(xtrans, TMatrixD::kMult, y);
  TMatrixD b(tmp, TMatrixD::kMult, tmp2);

  p0 = b(0, 0);
  p1 = b(1, 0);
  p2 = b(2, 0);

  // chi2 = (y-xb)^t . W . (y-xb)
  TMatrixD tmp3(x, TMatrixD::kMult, b);
  TMatrixD tmp4(y, TMatrixD::kMinus, tmp3);
  TMatrixD chi2(tmp4, TMatrixD::kTransposeMult, tmp4);

  return chi2(0, 0);
}


//__________________________________________________________________________
Double_t invQPtFromFCF(const FitterTrackMFT& track, Double_t bFieldZ, Double_t& chi2)
{
  // Fast Circle Fit (Hansroul, Jeremie, Savard, 1987)
  auto nPoints = track.getNClusters();
  Double_t* xVal = new Double_t[nPoints];
  Double_t* yVal = new Double_t[nPoints];
  Double_t* zVal = new Double_t[nPoints];
  Double_t* xErr = new Double_t[nPoints];
  Double_t* yErr = new Double_t[nPoints];
  Double_t* uVal = new Double_t[nPoints - 1];
  Double_t* vVal = new Double_t[nPoints - 1];
  Double_t* vErr = new Double_t[nPoints - 1];
  Double_t a, ae, b, be, x2, y2, invx2y2, rx, ry, r;

  int np = 0;
  for (auto trackparam = track.begin(); trackparam != track.end(); trackparam++) {
    //printf("BV track %d  %f  %f  %f \n", np, trackparam->getClusterPtr()->getX(), trackparam->getClusterPtr()->getY(), trackparam->getClusterPtr()->getZ());
    xErr[np] = trackparam->getClusterPtr()->sigmaX2;
    yErr[np] = trackparam->getClusterPtr()->sigmaY2;
    if (np > 0) {
      xVal[np] = trackparam->getClusterPtr()->xCoordinate - xVal[0];
      yVal[np] = trackparam->getClusterPtr()->yCoordinate - yVal[0];
      xErr[np] *= std::sqrt(2.);
      yErr[np] *= std::sqrt(2.);
    } else {
      xVal[np] = 0.;
      yVal[np] = 0.;
    }
    zVal[np] = trackparam->getClusterPtr()->zCoordinate;
    np++;
  }
  for (int i = 0; i < (np - 1); i++) {
    x2 = xVal[i + 1] * xVal[i + 1];
    y2 = yVal[i + 1] * yVal[i + 1];
    invx2y2 = 1. / (x2 + y2);
    uVal[i] = xVal[i + 1] * invx2y2;
    vVal[i] = yVal[i + 1] * invx2y2;
    vErr[i] = std::sqrt(8. * xErr[i + 1] * xErr[i + 1] * x2 * y2 + 2. * yErr[i + 1] * yErr[i + 1] * (x2 - y2)) * invx2y2 * invx2y2;
  }

  Double_t invqpt_fcf;
  Int_t qfcf;
  chi2 = 0.;
  if (LinearRegression((np - 1), uVal, vVal, yErr, a, ae, b, be)) {
    // v = a * u + b
    // circle passing through (0,0):
    // (x - rx)^2 + (y - ry)^2 = r^2
    // ---> a = - rx / ry;
    // ---> b = 1 / (2 * ry)
    ry =  1. / (2. * b);
    rx = -a * ry;
    r = std::sqrt(rx * rx + ry * ry);

    // pt --->
    Double_t invpt = 1. / (o2::constants::math::B2C * bFieldZ * r);

    // sign(q) --->
    // rotate around the first point (0,0) to bring the last point
    // on the x axis (y = 0) and check the y sign of the rotated
    // center of the circle
    Double_t x = xVal[np - 1], y = yVal[np - 1], z = zVal[np - 1];
    Double_t slope = TMath::ATan2(y, x);
    Double_t cosSlope = TMath::Cos(slope);
    Double_t sinSlope = TMath::Sin(slope);
    Double_t rxRot = rx * cosSlope + ry * sinSlope;
    Double_t ryRot = rx * sinSlope - ry * cosSlope;
    qfcf = (ryRot > 0.) ? -1 : +1;
    //printf("BV r-quad %f ,  r-fcf %f  q-fcf %f \n", 0.5 / q2, r, qfcf);

    //Double_t xRot = x * cosSlope + y * sinSlope;
    //printf("BV check %f %f \n", xRot, 2.0 * rxRot);

    Double_t alpha = 2.0 * std::abs(TMath::ATan2(rxRot, ryRot));
    Double_t x0 = xVal[0], y0 = yVal[0], z0 = zVal[0];
    Double_t dxyz2 = (x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0);
    Double_t cst = 1000.;
    Double_t c_alpha = cst * alpha;
    Double_t p, pt, pz;
    pt = 1. / invpt;
    p = std::sqrt(dxyz2) / c_alpha;
    pz = std::sqrt(p * p - pt * pt);
//    tanl = pz / pt;
    invqpt_fcf = qfcf * invpt;
  } else { // the linear regression failed...
    printf("BV LinearRegression failed!\n");
    invqpt_fcf = 1. / 100.;
  }

  return invqpt_fcf;
}

//__________________________________________________________________________
Bool_t LinearRegression(Int_t nVal, Double_t *xVal, Double_t *yVal, Double_t *yErr, Double_t &a, Double_t &ae, Double_t &b, Double_t &be)
{
  // linear regression y = a * x + b

  Double_t S1, SXY, SX, SY, SXX, SsXY, SsXX, SsYY, Xm, Ym, s, delta, difx;
  Double_t invYErr2;

  S1 = SXY = SX = SY = SXX = 0.0;
  SsXX = SsYY = SsXY = Xm = Ym = 0.;
  difx = 0.;
  for (Int_t i = 0; i < nVal; i++) {
    //printf("BV LinFit %d  %f  %f  %f  \n", i, xVal[i], yVal[i], yErr[i]);
    invYErr2 = 1. / (yErr[i] * yErr[i]);
    S1  += invYErr2;
    SXY += xVal[i] * yVal[i] * invYErr2;
    SX  += xVal[i] * invYErr2;
    SY  += yVal[i] * invYErr2;
    SXX += xVal[i] * xVal[i] * invYErr2;
    if (i > 0) difx += TMath::Abs(xVal[i] - xVal[i - 1]);
    Xm  += xVal[i];
    Ym  += yVal[i];
    SsXX += xVal[i] * xVal[i];
    SsYY += yVal[i] * yVal[i];
    SsXY += xVal[i] * yVal[i];
  }
  delta = SXX * S1 - SX * SX;
  if (delta == 0.) {
    return kFALSE;
  }
  a = (SXY * S1 - SX * SY) / delta;
  b = (SY * SXX - SX * SXY) / delta;

  Ym /= (Double_t)nVal;
  Xm /= (Double_t)nVal;
  SsYY -= (Double_t)nVal * (Ym*Ym);
  SsXX -= (Double_t)nVal * (Xm*Xm);
  SsXY -= (Double_t)nVal * (Ym*Xm);
  Double_t eps = 1.E-24;
  if ((nVal > 2) && (TMath::Abs(difx) > eps) && ((SsYY -(SsXY * SsXY) / SsXX) > 0.)) {
    s = TMath::Sqrt((SsYY - (SsXY * SsXY) / SsXX)/(nVal - 2));
    be = s * TMath::Sqrt(1. / (Double_t)nVal + (Xm * Xm) / SsXX);
    ae = s / TMath::Sqrt(SsXX);
  } else {
    be = 0.;
    ae = 0.;
  }
  return kTRUE;
}

} // namespace mft
} // namespace o2
