/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_CAPLER_SIMULATION_HPP_
#define NOESIS_GYM_ENVS_RAISIM_CAPLER_SIMULATION_HPP_

// C/C++
#include <unistd.h>
#include <cmath>
#include <iostream>

// Noesis
#include "noesis/framework/system/process.hpp"
#include "noesis/framework/system/time.hpp"
#include "noesis/framework/utils/macros.hpp"
#include "noesis/framework/math/random.hpp"
#include "noesis/framework/math/ops.hpp"
#include "noesis/gym/envs/raisim/common/math.hpp"
#include "noesis/gym/envs/raisim/common/world.hpp"

namespace noesis {
namespace gym {

class CaplerSimulation
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  // Constants
  static constexpr size_t Nb=1;
  static constexpr size_t Nj=2;
  static constexpr size_t Nq=3;
  static constexpr size_t Nu=3;
  
  // Internals types
  struct Parameters {
    double mu;
    double m_B;
    double m_T;
    double m_S;
    double l_S;
  };
  
  enum class World {
    Empty = 0,
    Grid
  };
  
  /*
   * Instantiation
   */
  
  explicit CaplerSimulation(World world=World::Empty, bool use_pid_controller=true):
    worldType_(world),
    usePidController_(use_pid_controller)
  {
    // Create and configure the simulation world
    world_.create();
    world_->setGravity(::raisim::Vec<3>{{0, 0, -9.81}});
    world_->setTimeStep(timeStep_);
    world_->setERP(10.0, 0.0);
    // Add capler
    std::string urdf = noesis::rootpath() + "/utils/models/capler/urdf/capler.urdf";
    capler_ = world_->addArticulatedSystem(urdf);
    capler_->setName("capler");
    capler_->getCollisionBodies().back().setMaterial("foot");
    // Add terrain
    switch (worldType_) {
      case World::Empty: {
        // Set zero gravity
        world_->setGravity(::raisim::Vec<3>{{0, 0, 0}});
      } break;
      case World::Grid: {
        // Add infinite flat-floor world
        floor_ = world_->addGround(0.0, "world");
      } break;
      default:
      NFATAL("[CaplerSimulation]: Invalid world type: Please use either of {Empty, Grid}.");
    }
    // Set physics parameters
    setFriction(mu_);
    // Set nominal shoulder positions to directly beneath the hips
    B_r_BN_ << 0.05, -0.0478, -0.35;
    // Set nominal joint configuration to define neutral joint positions of joint-space PD controller
    qjNominal_ << -M_PI_4, M_PI_2;
    // Configure the initial state
    qj_ = qjNominal_;
    q_ << 0.6, qj_;
    u_.setZero();
    tau_.setZero();
    capler_->setGeneralizedCoordinate(q_);
    capler_->setGeneralizedVelocity(u_);
    capler_->setGeneralizedForce(tau_);
    // Configure frame transformations
    // Base-to-Hip
    B_r_BH_ << 0.01755, -0.025, 0.0;
    R_BH_.setIdentity();
    // Hip-to-Thigh
    H_r_HT_ << 0.0, 0.0, 0.0;
    R_HT_.setIdentity();
    // Thigh-to-Shank
    T_r_TS_ << 0.0, 0.0, -0.24;
    R_TS_.setIdentity();
    // Shank-to-Foot
    S_r_SF_ << 0.0, -0.0228, -0.225;
    R_SF_.setIdentity();
    // Store default model parameters
    defaults_.mu = mu_;
    defaults_.m_B = capler_->getMass()[1];
    defaults_.m_T = capler_->getMass()[2];
    defaults_.m_S = capler_->getMass()[3];
    defaults_.l_S = std::abs(capler_->getCollisionBodies().back().posOffset[2]);
  }
  
  ~CaplerSimulation() = default;
  
  /*
   * Configurations
   */

  void setTorqueLpf(double alpha) { alpha_ = alpha; }
  
  void setMaxTorque(double tau) { maxTorque_ = tau; }
  
  void setFriction(double mu) {
    mu_ = mu;
    world_->setMaterialPairProp("foot", "terrain", mu_, cr_, cth_);
  }
  
  /*
   * Properties
   */
  
  noesis::gym::RaiSimWorld& world() {
    return world_;
  }
  
  ::raisim::ArticulatedSystem* capler() {
    return capler_;
  }
  
  ::raisim::Ground* floor() {
    return floor_;
  }
  
  World getWorldType() const {
    return worldType_;
  }
  
  double getTimeStep() const {
    return timeStep_;
  }
  
  double getMaxTorque() const {
    return maxTorque_;
  }
  
  double getFriction() const {
    return mu_;
  }
  
  Eigen::VectorXd getNominalJointConfiguration() const {
    return qjNominal_;
  }
  
  const Position& getPositionBaseToHipInBaseFrame() const {
    return B_r_BH_;
  }
  
  const Position& getPositionHipToThighInHipFrame() const {
    return H_r_HT_;
  }
  
  const Position& getPositionThighToShankInThighFrame() const {
    return T_r_TS_;
  }
  
  const Position& getPositionShankToFootInShankFrame() const {
    return S_r_SF_;
  }
  
  Position getNominalFootPositionInBaseFrame() const {
    return B_r_BN_;
  }
  
  const Eigen::VectorXd& getGeneralizedCoordinates() const {
    return q_;
  }
  
  const Eigen::VectorXd& getGeneralizedVelocities() const {
    return u_;
  }
  
  const Position& getPositionWorldToBaseInWorldFrame() const {
    return W_r_WB_;
  }
  
  const RotationMatrix& getOrientationBaseToWorld() const {
    return R_WB_;
  }
  
  const LinearVelocity& getLinearVelocityWorldToBaseInWorldFrame() const {
    return W_v_WB_;
  }
  
  const AngularVelocity& getAngularVelocityWorldToBaseInWorldFrame() const {
    return W_omega_WB_;
  }
  
  const Eigen::VectorXd& getJointPositions() const {
    return qj_;
  }
  
  const Eigen::VectorXd& getJointVelocities() const {
    return dqj_;
  }
  
  const Eigen::VectorXd& getJointTorques() const {
    return tauj_;
  }
  
  const Eigen::VectorXd& getDesiredJointPositions() const {
    return qjStar_;
  }
  
  const Eigen::VectorXd& getDesiredJointVelocities() const {
    return dqjStar_;
  }
  
  const Eigen::VectorXd& getDesiredJointTorques() const {
    return taujStar_;
  }
  
  Position getPositionWorldToComInWorldFrame() const {
    return capler_->getCompositeCOM().e();
  }
  
  Matrix3 getInertiaComInWorldFrame() const {
    return capler_->getCompositeInertia().e();
  }
  
  int getTotalContacts() const {
    return static_cast<int>(n_C_);
  }
  
  int getBaseContacts() const {
    return c_B_;
  }
  
  int getLegContacts() const {
    return c_L_;
  }
  
  int getShankContacts() const {
    return c_S_;
  }
  
  size_t getNumberOfLegContacts() const {
    return n_L_;
  }
  
  size_t getNumberOfShankContacts() const {
    return n_S_;
  }
  
  const Position& getPositionWorldToFootInWorldFrame() const {
    return W_r_WF_;
  }
  
  Position getPositionBaseToFootInBaseFrame() const {
    return Position(R_WB_.transpose() * (W_r_WF_ - W_r_WB_));
  }
  
  const LinearVelocity& getLinearVelocityWorldToFootInWorldFrame() const {
    return W_v_WF_;
  }
  
  LinearVelocity getLinearVelocityBaseToFootInBaseFrame() const {
    return LinearVelocity(R_WB_.transpose() * (W_v_WF_ - W_v_WB_));
  }
  
  const AngularVelocity& getAngularVelocityOfFootInWorldFrame() const {
    return W_omega_WF_;
  }
  
  const LinearForce& getFootForceInWorldFrame() const {
    return W_f_F_;
  }
  
  LinearForce getFootForceInBaseFrame() const {
    return LinearForce(R_WB_.transpose() * W_f_F_);
  }
  
  int getFootContact() const {
    return c_F_;
  }
  
  size_t getNumberOfFootContacts() const {
    return n_F_;
  }
  
  const LinearImpulse& getNetLinearVelocityOfFootContacts() const {
    return W_v_F_;
  }
  
  const LinearImpulse& getNetImpulseOfFootContacts() const {
    return W_p_F_;
  }
  
  /*
   * Operations
   */
  
  void randomize(unsigned int seed=0, double factor=1.0) {
    ::noesis::math::RandomNumberGenerator<double> prng(seed);
    // 1. Randomize foot-terrain contact friction
    setFriction(defaults_.mu * (1.0 + factor * prng.sampleUniform(-0.1, 0.5)));
    // 2. Randomize body masses
    auto& m_B = capler_->getMass()[1];
    m_B = defaults_.m_B * (1.0 + factor * prng.sampleUniform(-0.2, 0.2));
    auto& m_T = capler_->getMass()[2];
    m_T = defaults_.m_T * (1.0 + factor * prng.sampleUniform(-0.2, 0.2));
    auto& m_S = capler_->getMass()[3];
    m_S = defaults_.m_S * (1.0 + factor * prng.sampleUniform(-0.2, 0.2));
    // 3. Randomize link lengths
    auto& foot = capler_->getCollisionBodies().back();
    foot.posOffset[2] = - defaults_.l_S * (1.0 + factor * prng.sampleUniform(-0.1, 0.1));
    // TODO: capler_->getLinkCOM()[2]; // Thigh --> change z-element to move CoM up/down.
    // TODO: capler_->getLinkCOM()[3]; // Shank+Foot --> change z-element to move CoM up/down.
    // TODO: capler_->getInertia()[2]; // Thigh
    // TODO: capler_->getInertia()[3]; // Shank+Foot
    // Update physics engine internals
    capler_->updateMassInfo();
    NINFO_IF(factor > 0.0, "[CaplerSim]: New randomized model parameters:"
      << "\n  mu: " << mu_
      << "\n  m_B: " << m_B
      << "\n  m_T: " << m_T
      << "\n  m_S: " << m_S
      << "\n  l_S: " << std::abs(foot.posOffset[2])
    );
  }
  
  void reset(const Eigen::VectorXd& q, const Eigen::VectorXd& u) {
    q_ = q;
    u_ = u;
    tau_.setZero();
    W_f_F_.setZero();
    capler_->setGeneralizedCoordinate(q_);
    capler_->setGeneralizedVelocity(u_);
    capler_->setGeneralizedForce(tau_);
    step(q_.tail(Nj));
  }
  
  bool step(const Eigen::VectorXd& joint_commands) {
    bool stepIsValid = true;
    // Actuator dynamics
    if (usePidController_) { pidTorque(joint_commands); } else { pddTorque(joint_commands); }
    // Integrate physics
    world_.integrate1();
    capler_->setGeneralizedForce(tau_);
    world_.integrate2();
    // Update state
    q_ = capler_->getGeneralizedCoordinate().e();
    u_ = capler_->getGeneralizedVelocity().e();
    updateBase();
    updateJoints();
    updateContacts();
    // Data validity checks
    if (!u_.allFinite()) { stepIsValid = false; }
    if (!q_.allFinite()) { stepIsValid = false; }
    if (q_.norm() > 1000.0) { stepIsValid = false; }
    return stepIsValid;
  }
  
  inline void foot_forward_kinematics(const Vector2& q, Position& B_r_BF) {
    RotationMatrix R_BH = R_BH_;
    RotationMatrix R_HT = R_HT_*math::rotation_y(q(0));
    RotationMatrix R_TS = R_TS_*math::rotation_y(q(1));
    B_r_BF = B_r_BH_ + R_BH*H_r_HT_ + R_BH*R_HT*T_r_TS_ + R_BH*R_HT*R_TS*S_r_SF_;
  }
  
  inline Position foot_forward_kinematics(const Vector2& q) {
    Position B_r_BF = Position::Zero();
    foot_forward_kinematics(q, B_r_BF);
    return B_r_BF;
  }
  
  inline void foot_position_jacobian(const Vector2& q, Matrix3& jacobian) {
    RotationMatrix R_BH = R_BH_;
    RotationMatrix R_HT = R_HT_*math::rotation_y(q(0));
    RotationMatrix R_TS = R_TS_*math::rotation_y(q(1));
    Matrix3 S_BH = Matrix3::Zero();
    Matrix3 S_HT = R_HT_*::noesis::gym::math::skew(Vector3(0,1,0))*R_HT_.transpose();
    Matrix3 S_TS = R_TS_*::noesis::gym::math::skew(Vector3(0,1,0))*R_TS_.transpose();
    jacobian.col(0) = S_BH*R_BH*(H_r_HT_ + R_HT*T_r_TS_ + R_HT*R_TS*S_r_SF_);
    jacobian.col(1) = R_BH*S_HT*R_HT*(T_r_TS_ + R_TS*S_r_SF_);
    jacobian.col(2) = R_BH*R_HT*S_TS*R_TS*S_r_SF_;
  }
  
  inline Vector2 foot_inverse_kinematics(const Position& B_r_BF, double lambda=0.1, double epsilon=0.001, size_t maxiter=2000) {
    Matrix3 jac, invJac;
    Position B_r_BF_approx = Position::Zero();
    Position delta = Position::Zero();
    Vector3 q = Vector3::Zero();
    q.tail(2) = qjNominal_;
    size_t iter = 0;
    do {
      foot_forward_kinematics(q.tail(2), B_r_BF_approx);
      delta = B_r_BF - B_r_BF_approx;
      foot_position_jacobian(q.tail(2), jac);
      ::noesis::math::matrix_pseudo_inverse(jac, invJac);
      q += lambda * invJac * delta;
      iter++;
    } while ((delta.norm() > epsilon) && (iter < maxiter));
    for (int k = 0; k < q.size(); ++k) { q(k) = ::noesis::math::angle_modulo(q(k)); }
    if (iter >= maxiter) {
      NWARNING("Inverse kinematics reached max iterations (" << std::to_string(maxiter) << ")!");
      NWARNING("B_r_BF: " << B_r_BF.transpose());
      NWARNING("B_r_BF_approx: " << B_r_BF_approx.transpose());
      NWARNING("qj: " << q.transpose());
    }
    return Vector2(q.tail(2));
  }
  
  std::string info() const {
    return raisim::articulated_system_info(*capler_);
  }
  
private:
  
  inline void pidTorque(const Eigen::VectorXd& joint_commands) {
    // Buffer previous joint torque vector
    Eigen::VectorXd tauj = tauj_;
    // Joint-space PD control
    qjStar_ = joint_commands.segment<Nj>(0);
    taujStar_ = (qjStar_ - qj_) * kp_ - dqj_ * kd_;
    // Process joint torque commands
    tauj_ = taujStar_;
    tauj_ = tauj_* (1.0-alpha_) + tauj * alpha_;
    tauj_ = tauj_.array().min(maxTorque_).max(-maxTorque_);
    tau_.head(Nb).setZero();
    tau_.tail(Nj) = tauj_;
  }
  
  inline void pddTorque(const Eigen::VectorXd& joint_commands) {
    // Buffer previous joint torque vector
    Eigen::VectorXd tauj = tauj_;
    // Direct joint torque control
    taujStar_ = joint_commands.segment<Nj>(0);
    // Process joint torque commands
    tauj_ = taujStar_;
    tauj_ = tauj_* (1.0-alpha_) + tauj * alpha_;
    tauj_ = tauj_.array().min(maxTorque_).max(-maxTorque_);
    tau_.head(Nb).setZero();
    tau_.tail(Nj) = tauj_;
  }
  
  inline void updateBase() {
    W_r_WB_.z() = q_(0);
    W_v_WB_.z() = u_(0);
  }
  
  inline void updateJoints() {
    qj_ = q_.template segment<Nj>(Nb);
    dqj_ = u_.template segment<Nj>(Nb);
  }
  
  inline void updateContacts() {
    ::raisim::Vec<3> vec3;
    // Update contact counters
    n_C_ = capler_->getContacts().size();
    // Initialize base contact properties
    c_B_ = 0;
    c_L_ = 0;
    c_S_ = 0;
    n_B_ = 0;
    n_S_ = 0;
    n_L_ = 0;
    // Initialize feet contact properties and feet positions
    W_f_F_.setZero();
    c_F_ = 0;
    n_F_ = 0;
    const size_t bid = 3;
    ::raisim::Vec<3> S_r_SF;
    S_r_SF.e() = S_r_SF_;
    capler_->getPosition(bid, S_r_SF, vec3);
    W_r_WF_ = vec3.e();
    capler_->getVelocity(bid, S_r_SF, vec3);
    W_v_WF_ = vec3.e();
    capler_->getAngularVelocity(bid, vec3);
    W_omega_WF_ = vec3.e();
    W_n_F_.setZero();
    W_v_F_.setZero();
    W_p_F_.setZero();
    // Check over active contacts
    if (n_C_ > 0) {
      for (int k = 0; k < static_cast<int>(n_C_); k++) {
        if (!capler_->getContacts()[k].skip()) {
          size_t bid = capler_->getContacts()[k].getlocalBodyIndex();
          if (bid == 0) {
            c_B_ = 1;
            n_B_++;
          } else if (bid == 3) {
            W_n_F_ += capler_->getContacts()[k].getNormal().e();
            W_p_F_ += capler_->getContacts()[k].getImpulse()->e();
            capler_->getContactPointVel(k, vec3);
            W_v_F_ += vec3.e();
            double err = (W_r_WF_ - capler_->getContacts()[k].getPosition().e()).norm();
            if (err < 0.04 && capler_->getContacts()[k].getNormal().e()[2] > 0.5){
              n_F_++;
              c_F_ = 1;
            } else {
              n_S_++;
              c_S_ = 1;
            }
          } else {
            n_L_++;
            c_L_ = 1;
          }
        }
      }
    }
    // Compute contact forces from step-wise impulses
    W_f_F_ = W_p_F_ * (1.0/timeStep_);
    // Compute mean contact properties over total active contacts on each foot
    if (n_F_ > 1) {
      W_n_F_ /= n_F_;
      W_v_F_ /= n_F_;
      W_p_F_ /= n_F_;
    }
  }
  
private:
  // Default parameters
  Parameters defaults_;
  // Simulation elements
  ::noesis::gym::RaiSimWorld world_;
  ::raisim::ArticulatedSystem* capler_{nullptr};
  ::raisim::Ground* floor_{nullptr};
  World worldType_{World::Empty};
  // Kinematics parameters
  Position B_r_BN_;
  Position B_r_BH_;
  Position H_r_HT_;
  Position T_r_TS_;
  Position S_r_SF_;
  RotationMatrix R_BH_;
  RotationMatrix R_HT_;
  RotationMatrix R_TS_;
  RotationMatrix R_SF_;
  // Dynamics state
  Eigen::VectorXd q_{Eigen::VectorXd::Zero(Nq)};
  Eigen::VectorXd u_{Eigen::VectorXd::Zero(Nu)};
  Eigen::VectorXd tau_{Eigen::VectorXd::Zero(Nu)};
  // Joints
  Eigen::VectorXd qj_{Eigen::VectorXd::Zero(Nj)};
  Eigen::VectorXd dqj_{Eigen::VectorXd::Zero(Nj)};
  Eigen::VectorXd tauj_{Eigen::VectorXd::Zero(Nj)};
  // PD controller
  Eigen::VectorXd qjStar_{Eigen::VectorXd::Zero(Nj)};
  Eigen::VectorXd dqjStar_{Eigen::VectorXd::Zero(Nj)};
  Eigen::VectorXd taujStar_{Eigen::VectorXd::Zero(Nj)};
  Eigen::VectorXd qjNominal_{Eigen::VectorXd::Zero(Nj)};
  // Base and limbs
  Quaternion q_IB_{Quaternion(1,0,0,0)};
  RotationMatrix R_WB_{RotationMatrix::Identity()};
  Position W_r_WB_{Position::Zero()};
  LinearVelocity W_v_WB_{LinearVelocity::Zero()};
  AngularVelocity W_omega_WB_{AngularVelocity::Zero()};
  // Foot
  Position W_r_WF_{Position::Zero()};
  LinearForce W_f_F_{LinearForce::Zero()};
  LinearVelocity W_v_WF_{LinearVelocity::Zero()};
  AngularVelocity W_omega_WF_{AngularVelocity::Zero()};
  Position W_n_F_;
  LinearVelocity W_v_F_;
  LinearImpulse W_p_F_;
  // Contacts
  int c_B_{0}; // Base
  int c_S_{0}; // Leg
  int c_L_{0}; // Shank
  int c_F_{0}; // Foot
  size_t n_C_{0}; // Total
  size_t n_B_{0}; // Base
  size_t n_S_{0}; // Leg
  size_t n_L_{0}; // Shank
  size_t n_F_{0}; // Foot
  // Configurations
  double mu_{0.6};
  double cr_{0.0};
  double cth_{0.0};
  double kp_{60.0};
  double kd_{3.0};
  double alpha_{0.0};
  double maxTorque_{65.0};
  double timeStep_{0.0025};
  bool usePidController_{true};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_CAPLER_SIMULATION_HPP_

/* EOF */
