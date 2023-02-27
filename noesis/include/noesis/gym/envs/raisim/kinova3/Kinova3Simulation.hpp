/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_KINOVA3_SIMULATION_HPP_
#define NOESIS_GYM_ENVS_RAISIM_KINOVA3_SIMULATION_HPP_

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
#include "noesis/gym/envs/raisim/common/raisim.hpp"
#include "noesis/gym/envs/raisim/common/world.hpp"
#include "noesis/gym/envs/raisim/common/math.hpp"

namespace noesis {
namespace gym {

class Kinova3Simulation
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  // System dimensions
  static constexpr size_t Nj=7; // Number of joints
  static constexpr size_t Nq=7; // Number of generalized coordinates
  static constexpr size_t Nu=7; // Number of generalized velocities
  
  // Arm (large) actuator specs
  const double ArmActuatorPeakTorque=74.0;
  const double ArmActuatorNominalTorque=32.0;
  const double ArmActuatorMaxVelocity=1.78;
  
  // Wrist (small) actuator specs
  const double WristActuatorPeakTorque=34.0;
  const double WristActuatorNominalTorque=13.0;
  const double WristActuatorMaxVelocity=2.61;
  
  // Maximum range about nominal joint position
  const double ArmMaxJointDeflection=M_PI_2;
  const double WristMaxJointDeflection=M_PI_2;
  
  // Internals types
  struct Parameters {
    double mu;
    // TODO: all add link masses
  };
  
  enum class World {
    NoGravity = 0,
    Empty,
    Grid
  };
  
  /*
   * Instantiation
   */
  
  explicit Kinova3Simulation(World world=World::Empty):
    worldType_(world)
  {
    // Create and configure the simulation world
    world_.create();
    world_->setGravity(::raisim::Vec<3>{{0, 0, -9.81}});
    world_->setTimeStep(timeStep_);
    world_->setERP(10.0, 0.0);
    // Add the Kinova3 articulated system
    const std::string urdf = noesis::rootpath() + "/utils/models/kinova3/urdf/kinova3.urdf";
    kinova3_ = world_->addArticulatedSystem(urdf);
    kinova3_->setName("kinova3");
    kinova3_->getCollisionBodies().back().setMaterial("end_effector");
    // Add world
    switch (worldType_) {
      case World::NoGravity: {
        // Set zero gravity
        world_->setGravity(::raisim::Vec<3>{{0, 0, 0}});
      } break;
      case World::Empty: {
        // No action
      } break;
      case World::Grid: {
        // Add infinite flat-floor world
        floor_ = world_->addGround(0.0, "world");
      } break;
      default:
        NFATAL("[Kinova3Simulation]: Invalid world type: Please use either of {NoGravity, Empty, Grid}.");
    }
    // Set torque limits
    dqMax_ << Eigen::VectorXd::Constant(4, ArmActuatorMaxVelocity), Eigen::VectorXd::Constant(3, WristActuatorMaxVelocity);
    tauMax_ << Eigen::VectorXd::Constant(4, ArmActuatorPeakTorque), Eigen::VectorXd::Constant(3, WristActuatorPeakTorque);
    // Set nominal joint configurations to define neutral joint positions of joint-space PD controller
    qjNominal_ << 0.0, 0.4, 0.0, M_PI_2+0.6, 0.0, -1.0, -M_PI_2;
    // Configure the initial state
    q_ << qjNominal_;
    u_.setZero();
    tau_.setZero();
    kinova3_->setState(q_, u_);
    kinova3_->setGeneralizedForce(tau_);
    kinova3_->setActuationLimits(tauMax_, -tauMax_);
    // PID gains
    const auto Kp_A = ArmActuatorPeakTorque/ArmMaxJointDeflection;
    const auto Kp_W = WristActuatorPeakTorque/WristMaxJointDeflection;
    kp_ << Eigen::VectorXd::Constant(4, Kp_A), 0.1*Kp_W, 0.1*Kp_W, 0.1*Kp_W;
    kd_ << Eigen::VectorXd::Constant(4, 1.0), 0.1, 0.1, 0.1;
    b_ << Eigen::VectorXd::Constant(4, 0.0), Eigen::VectorXd::Constant(3, 0.0);
    // Initialize references
    qjStar_ = qjNominal_;
    dqjStar_.setZero();
    taujStar_.setZero();
  }
  
  ~Kinova3Simulation() = default;
  
  /*
   * Configurations
   */

  void setTimeStep(double dt) {
    world_->setTimeStep(dt);
    timeStep_ = dt;
  }
  
  void setUsePidJointController(bool enable) {
    usePidController_ = enable;
  }
  
  void setUseSimulatorPid(bool enable) {
    useSimulatorPid_ = enable;
  }
  
  void setMaxJointVelocity(int joint, double velocity) {
    dqMax_(joint) = velocity;
  }
  
  void setMaxJointVelocities(double arm, double wrist) {
    dqMax_.segment<4>(0).setConstant(arm);
    dqMax_.segment<3>(4).setConstant(wrist);
  }
  
  void setMaxJointTorque(int joint, double torque) {
    tauMax_(joint) = torque;
  }
  
  void setMaxJointTorques(double arm, double wrist) {
    tauMax_.segment<4>(0).setConstant(arm);
    tauMax_.segment<3>(4).setConstant(wrist);
  }
  
  void setTorqueLowPassFilterFactor(double alpha) {
    alpha_ = alpha;
  }
  
  void setForceCompensationFactor(double beta) {
    beta_ = beta;
  }
  
  void setJointPidKp(double kpArm, double kpWrist) {
    kp_ << Eigen::VectorXd::Constant(4, kpArm), Eigen::VectorXd::Constant(3, kpWrist);
  }

  void setJointPidKd(double kdArm, double kdWrist) {
    kd_ << Eigen::VectorXd::Constant(4, kdArm), Eigen::VectorXd::Constant(3, kdWrist);
  }
  
  void setJointDamping(double bArm, double bWrist) {
    b_ << Eigen::VectorXd::Constant(4, bArm), Eigen::VectorXd::Constant(3, bWrist);
  }
  
  void setEndEffectorFriction(double mu) {
    world_->setMaterialPairProp("end_effector", "world", mu, cr_, cth_);
    mu_ = mu;
  }
  
  /*
   * Properties
   */
  
  noesis::gym::RaiSimWorld& world() {
    return world_;
  }
  
  ::raisim::ArticulatedSystem* kinova3() {
    return kinova3_;
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
  
  double getMaxJointVelocity(int joint) const {
    return dqMax_(joint);
  }
  
  const Eigen::VectorXd& getMaxJointVelocities() const {
    return dqMax_;
  }
  
  double getMaxJointTorque(int joint) const {
    return tauMax_(joint);
  }
  
  const Eigen::VectorXd& getMaxJointTorques() const {
    return tauMax_;
  }
  
  const Eigen::VectorXd& getJointPGains() const {
    return kp_;
  }
  
  const Eigen::VectorXd& getJointDGains() const {
    return kd_;
  }
  
  const Eigen::VectorXd& getJointDamping() const {
    return b_;
  }
  
  double getFriction() const {
    return mu_;
  }
  
  Eigen::VectorXd getNominalJointConfiguration() const {
    return qjNominal_;
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
  
  const Eigen::VectorXd& getJointPositions() const {
    return q_;
  }
  
  const Eigen::VectorXd& getJointVelocities() const {
    return u_;
  }
  
  const Eigen::VectorXd& getJointTorques() const {
    return tau_;
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
    return kinova3_->getCompositeCOM().e();
  }
  
  Matrix3 getInertiaComInWorldFrame() const {
    return kinova3_->getCompositeInertia().e();
  }
  
  int getTotalContacts() const {
    return static_cast<int>(n_C_);
  }
  
  int getArmContacts() const {
    return c_A_;
  }
  
  int getWristContacts() const {
    return c_W_;
  }
  
  size_t getNumberOfArmContacts() const {
    return n_A_;
  }
  
  size_t getNumberOfWristContacts() const {
    return n_W_;
  }
  
  const RotationMatrix& getOrientationWorldToEndEffector() const {
    return R_ee_;
  }
  
  const Position& getPositionWorldToEndEffectorInWorldFrame() const {
    return W_r_ee_;
  }
  
  Position getPositionWorldToEndEffectorInBaseFrame() const {
    return Position(R_WB_.transpose() * (W_r_ee_ - W_r_WB_));
  }
  
  const LinearVelocity& getLinearVelocityWorldToEndEffectorInWorldFrame() const {
    return W_v_ee_;
  }
  
  LinearVelocity getLinearVelocityBaseToEndEffectorInBaseFrame() const {
    return LinearVelocity(R_WB_.transpose() * (W_v_ee_ - W_v_WB_));
  }
  
  const AngularVelocity& getAngularVelocityOfEndEffectorInWorldFrame() const {
    return W_omega_ee_;
  }
  
  AngularVelocity getAngularVelocityBaseToEndEffectorInBaseFrame() const {
    return AngularVelocity(R_WB_.transpose() * (W_omega_ee_ - W_omega_WB_));
  }
  
  const LinearForce& getEndEffectorForceInWorldFrame() const {
    return W_f_ee_;
  }
  
  LinearForce getEndEffectorForceInBaseFrame() const {
    return LinearForce(R_WB_.transpose() * W_f_ee_);
  }

  int getEndEffectorContact() const {
    return c_ee_;
  }

  int getArmContact() const {
        return c_A_;
  }

  int getWristContact() const {
      return c_W_;
  }

  int getEEContact() const {
      return c_ee_;
  }

  size_t getNumberOfEndEffectorContacts() const {
    return n_ee_;
  }
  
  const LinearImpulse& getNetLinearVelocityOfEndEffectorContacts() const {
    return W_v_ee_;
  }
  
  const LinearImpulse& getNetImpulseOfEndEffectorContacts() const {
    return W_p_ee_;
  }
  
  bool isUsingPidController() const {
    return usePidController_;
  }
  
  /*
   * Operations
   */
  
  void randomize(unsigned int seed=0, double factor=1.0) {
    ::noesis::math::RandomNumberGenerator<double> prng(seed);
    
    // TODO: Search for actuator bandwidth in Kinova3 manual.
    // TODO: Craft LPF filter for actions
    // TODO: Try SS LTI filter on actions (action rate)
    // TODO: Torus EE position workspace --> ??? for EE orientation workspace
    // TODO: Random EE wrench disturbance
    // TODO: Craft app which tests policies under different system perturbations and computes perf metrics.
    // TODO: Implement model randomization
  
    // TODO: Randomize external wrenches applied to main links
    // TODO: Randomize action delay using FIFO.
    // TODO: Randomize link mass/inertia
    // TODO: Randomize b_
    // TODO: Randomize kp_
    // TODO: Randomize kd_
  }
  
  void reset(const Eigen::VectorXd& q, const Eigen::VectorXd& u) {
    q_ = q;
    u_ = u;
    tau_.setZero();
    kinova3_->setState(q_, u_);
    kinova3_->setGeneralizedForce(tau_);
    updateEndEffector();
    updateContacts();
  }

  bool step(const Eigen::VectorXd& commands) {
    // Actuator dynamics
    if (usePidController_) { pidTorque(commands); } else { pddTorque(commands); }
    // Integrate physics
    world_.integrate();
    // Update state
    q_ = kinova3_->getGeneralizedCoordinate().e();
    u_ = kinova3_->getGeneralizedVelocity().e();
    tau_ = kinova3_->getGeneralizedForce().e();
    updateEndEffector();
    updateContacts();
    // Data validity checks
    bool stepIsValid = true;
    if (!u_.allFinite()) { stepIsValid = false; }
    if (!q_.allFinite()) { stepIsValid = false; }
    if (q_.norm() > 1000.0) { stepIsValid = false; }
    return stepIsValid;
  }
  
  std::string info() const {
    return raisim::articulated_system_info(*kinova3_);
  }
  
  static inline RotationMatrix transformFrameFromWorldToEndEffector(const RotationMatrix& R) {
    return R * math::rotation_z(M_PI_2) * math::rotation_x(M_PI_2);
  }
  
private:
  
  inline void pidTorque(const Eigen::VectorXd& commands) {
    // Joint-space PD control
    qjStar_ = commands.segment<Nj>(0);
    if (commands.size() == 2*Nj) {
      dqjStar_ = commands.segment<Nj>(Nj);
      taujStar_.setZero();
    } else if (commands.size() == 3*Nj) {
      dqjStar_ = commands.segment<Nj>(Nj);
      taujStar_ = commands.segment<Nj>(2*Nj);
    } else {
      dqjStar_.setZero();
      taujStar_.setZero();
    }
    // Process joint torque commands
    taujStar_ = taujStar_.cwiseMin(tauMax_).cwiseMax(-tauMax_);
    // Set (optional) joint damping (viscous friction).
    kinova3_->setJointDamping(b_);
    // Set commands into the simulator
    if (useSimulatorPid_) {
      kinova3_->setPdGains(kp_, kd_);
      kinova3_->setPdTarget(qjStar_, dqjStar_); // Set-point references
      kinova3_->setGeneralizedForce(taujStar_); // Feed-forward reference
    } else {
      Eigen::VectorXd tauPid = (qjStar_ - q_).cwiseProduct(kp_) + (dqjStar_ - u_).cwiseProduct(kd_) + taujStar_;
      tauPid = tauPid.cwiseMin(tauMax_).cwiseMax(-tauMax_);
      kinova3_->setGeneralizedForce(tauPid);
    }
  }
  
  inline void pddTorque(const Eigen::VectorXd& commands) {
    // Buffer the previous command
    const Eigen::VectorXd tauStar = taujStar_;
    const Eigen::VectorXd h = kinova3_->getNonlinearities().e();
    // Direct joint torque control
    taujStar_ = commands.segment<Nj>(0) + beta_ * h;
    // Process joint torque commands
    taujStar_ = (1.0 - alpha_) * taujStar_ + alpha_ * tauStar;
    taujStar_ = taujStar_.cwiseMin(tauMax_).cwiseMax(-tauMax_);
    // Set commands into the simulator
    kinova3_->setJointDamping(b_);
    kinova3_->setGeneralizedForce(taujStar_);
  }
  
  inline void updateEndEffector() {
    ::raisim::Vec<3> vec3;
    ::raisim::Mat<3,3> mat3;
    // Retrieve the end-effector state
    const auto fid = kinova3_->getFrameIdxByName("kinova3_end_effector_link_to_end_effector");
    kinova3_->getFrameOrientation(fid, mat3);
    R_ee_ = mat3.e();
    kinova3_->getFramePosition(fid,vec3);
    W_r_ee_ = vec3.e();
    kinova3_->getFrameVelocity(fid, vec3);
    W_v_ee_ = vec3.e();
    kinova3_->getFrameAngularVelocity(fid, vec3);
    W_omega_ee_ = vec3.e();
  }

  inline void updateContacts() {
    ::raisim::Vec<3> vec3;
    // Update contact counters
    n_C_ = kinova3_->getContacts().size();
    // Initialize base contact properties
    c_A_ = 0;
    c_W_ = 0;
    c_ee_ = 0;
    n_A_ = 0;
    n_W_ = 0;
    n_ee_ = 0;
    // Initialize force/impulse members
    W_n_ee_.setZero();
    W_p_ee_.setZero();
    W_f_ee_.setZero();
    // Check over active contacts
    if (n_C_ > 0) {
      for (int k = 0; k < static_cast<int>(n_C_); k++) {
        if (!kinova3_->getContacts()[k].skip()) {
          size_t bid = kinova3_->getContacts()[k].getlocalBodyIndex();
          if (bid == 2) { // Arm
            n_A_ ++;
            c_A_ = 1;
          } else if (bid == 5) { // Wrist
            n_W_ ++;
            c_W_ = 1;
          } else if (bid == 7) {
            W_n_ee_ += kinova3_->getContacts()[k].getNormal().e();
            W_p_ee_ += kinova3_->getContacts()[k].getImpulse()->e();
            n_ee_++;
            c_ee_ = 1;
          }
        }
      }
    }
    // Compute mean contact properties over total active contacts on each foot
    if (n_ee_ > 1) {
      W_n_ee_ /= n_ee_;
      W_p_ee_ /= n_ee_;
    }
    // Compute contact forces from step-wise impulses
    W_f_ee_ = W_p_ee_ * (1.0/timeStep_);
  }

private:
  // Default parameters
  Parameters defaults_;
  // Simulation elements
  ::noesis::gym::RaiSimWorld world_;
  ::raisim::ArticulatedSystem* kinova3_{nullptr};
  ::raisim::Ground* floor_{nullptr};
  World worldType_{World::Empty};
  // Dynamics state
  Eigen::VectorXd q_{Eigen::VectorXd::Zero(Nq)};
  Eigen::VectorXd u_{Eigen::VectorXd::Zero(Nu)};
  Eigen::VectorXd tau_{Eigen::VectorXd::Zero(Nu)};
  Eigen::VectorXd dqMax_{Eigen::VectorXd::Zero(Nu)};
  Eigen::VectorXd tauMax_{Eigen::VectorXd::Zero(Nu)};
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
  // End-Effector
  RotationMatrix R_ee_{RotationMatrix::Identity()};
  Position W_r_ee_{Position::Zero()};
  LinearForce W_f_ee_{LinearForce::Zero()};
  LinearVelocity W_v_ee_{LinearVelocity::Zero()};
  AngularVelocity W_omega_ee_{AngularVelocity::Zero()};
  LinearImpulse W_p_ee_{LinearImpulse::Zero()};
  Vector3 W_n_ee_{Vector3::Zero()};
  // Gains
  Eigen::VectorXd kp_{Eigen::VectorXd::Zero(Nj)};
  Eigen::VectorXd kd_{Eigen::VectorXd::Zero(Nj)};
  Eigen::VectorXd b_{Eigen::VectorXd::Zero(Nj)};
  // Contacts
  int c_A_{0}; // Arm
  int c_W_{0}; // Wrist
  int c_ee_{0}; // Foot
  size_t n_C_{0}; // Total
  size_t n_A_{0}; // Arm
  size_t n_W_{0}; // Wrist
  size_t n_ee_{0}; // End-Effector
  // Configurations
  double mu_{0.6};
  double cr_{0.0};
  double cth_{0.0};
  double alpha_{0.0};
  double beta_{0.0};
  double timeStep_{0.0025};
  bool usePidController_{true};
  bool useSimulatorPid_{true};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_KINOVA3_SIMULATION_HPP_

/* EOF */
