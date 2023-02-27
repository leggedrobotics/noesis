/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2020 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_COMMON_FILTERS_HPP_
#define NOESIS_GYM_ENVS_RAISIM_COMMON_FILTERS_HPP_

// C/C++
#include <cmath>
#include <random>

// Noesis
#include <noesis/framework/log/message.hpp>

// DeepGait
#include <noesis/gym/envs/raisim/common/math.hpp>

namespace noesis {
namespace gym {
namespace filters {

/*!
 * @brief Abstract base class for signal filters.
 */
class Filter
{
public:
  
  /*
   * Instantiation
   */
  
  Filter() = default;
  
  virtual ~Filter() = default;
  
  /*
   * Properties
   */

  virtual const Eigen::VectorXd& input() const = 0;
  
  virtual const Eigen::VectorXd& state() const = 0;
  
  virtual const Eigen::VectorXd& output() const = 0;
  
  virtual Eigen::VectorXd& state() = 0;
  
  /*
   * Operations
   */
  
  virtual void initialize(const Eigen::VectorXd& x0) = 0;
  
  virtual Eigen::VectorXd advance(const Eigen::VectorXd& x) = 0;
  
  virtual void advance(const Eigen::VectorXd& x, Eigen::VectorXd& y) = 0;
};

/*!
 * @brief Implements an FIR filter realized using finite-difference approximations of a given signal.
 */
class FiniteDifferencesFilter final: public Filter
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  /*
   * Instantiation
   */
  
  FiniteDifferencesFilter() = default;
  
  explicit FiniteDifferencesFilter(size_t state_dim) { setDimensions(state_dim); }
  
  ~FiniteDifferencesFilter() final = default;
  
  /*
   * Configurations
   */
  
  void setDimensions(size_t state_dim) {
    x_.resize(state_dim);
    dx_.resize(state_dim);
    ddx_.resize(state_dim);
    history_.resize(3*state_dim);
  }
  
  /*
   * Properties
   */
  
  const Eigen::VectorXd& input() const override { return x_; }
  
  const Eigen::VectorXd& state() const override { return history_; }
  
  const Eigen::VectorXd& output() const override { return ddx_; }
  
  Eigen::VectorXd& state() override { return history_; }
  
  const Eigen::VectorXd& first_derivative() const { return dx_; }
  
  const Eigen::VectorXd& second_derivative() const { return ddx_; }
  
  /*
   * Operations
   */

  void initialize(const Eigen::VectorXd& x0) override {
    const auto Nx = x_.size();
    DNFATAL_IF(x0.size() != Nx, "[FiniteDifferencesFilter]: Initial state has invalid dimensions: Must be dim(x) = " << Nx);
    x_ = x0;
    dx_.setZero();
    ddx_.setZero();
    for (int k = 0; k < 3; ++k) { history_.segment(Nx*k, Nx) = x_; }
  }
  
  Eigen::VectorXd advance(const Eigen::VectorXd& x) override {
    const auto Nx = x_.size();
    DNFATAL_IF(x.size() != Nx, "[FiniteDifferencesFilter]: New state has invalid dimensions: Must be dim(x) = " << Nx);
    history_.head(2*Nx) = history_.tail(2*Nx);
    history_.tail(Nx) = x;
    x_ = x;
    dx_ = x_ - history_.segment(Nx, Nx);
    ddx_ = x_ - 2.0*history_.segment(Nx, Nx) + history_.segment(0, Nx);
    return ddx_;
  }
  
  void advance(const Eigen::VectorXd& x, Eigen::VectorXd& ddx) override {
    ddx = advance(x);
  }

private:
  Eigen::VectorXd x_{Eigen::VectorXd::Zero(1)};
  Eigen::VectorXd dx_{Eigen::VectorXd::Zero(1)};
  Eigen::VectorXd ddx_{Eigen::VectorXd::Zero(1)};
  Eigen::VectorXd history_{Eigen::VectorXd::Zero(3)};
};

/*!
 * @brief Implements linear filters using State-Space realizations of LTI systems.
 */
class StateSpaceFilter final: public Filter
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  /*
   * Instantiation
   */
  
  StateSpaceFilter() = default;
  
  explicit StateSpaceFilter(size_t input_dim, size_t state_dim, size_t output_dim) { setDimensions(input_dim, state_dim, output_dim); }
  
  ~StateSpaceFilter() final = default;
  
  /*
   * Configurations
   */
  
  // TODO: from YAML
  
  void setDimensions(size_t input_dim, size_t state_dim, size_t output_dim) {
    u_.resize(input_dim);
    x_.resize(state_dim);
    y_.resize(output_dim);
    A_.resize(state_dim, state_dim);
    B_ .resize(state_dim, input_dim);
    C_.resize(output_dim, state_dim);
    D_.resize(output_dim, input_dim);
  }
  
  void setMatrices(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXd& D
    ) {
    NFATAL_IF(!dimensions_match(A, A_), "[StateSpaceFilter]: Matrix A: Invalid dimensions: Must be " << A_.rows() << "x" << A_.cols());
    NFATAL_IF(!dimensions_match(B, B_), "[StateSpaceFilter]: Matrix B: Invalid dimensions: Must be " << B_.rows() << "x" << B_.cols());
    NFATAL_IF(!dimensions_match(C, C_), "[StateSpaceFilter]: Matrix C: Invalid dimensions: Must be " << C_.rows() << "x" << C_.cols());
    NFATAL_IF(!dimensions_match(D, D_), "[StateSpaceFilter]: Matrix D: Invalid dimensions: Must be " << D_.rows() << "x" << D_.cols());
    A_ = A;
    B_ = B;
    C_ = C;
    D_ = D;
    // Optionally produce verbose console output.
    NINFO_IF(verbose_, "[StateSpaceFilter]: Using new state-space model:"
      << "\nA:\n" << A_
      << "\nB:\n" << B_
      << "\nC:\n" << C_
      << "\nD:\n" << D_
    );
  }
  
  void setVerbose(bool verbose) {
    verbose_ = verbose;
  }
  
  /*
   * Properties
   */

  const Eigen::VectorXd& input() const override { return u_; }
  
  const Eigen::VectorXd& state() const override { return x_; }
  
  const Eigen::VectorXd& output() const override { return y_; }
  
  Eigen::VectorXd& state() override { return x_; }
  
  /*
   * Operations
   */

  void initialize(const Eigen::VectorXd& u0) override {
    // Retrieve system dimensions
    const auto Nu = u_.size();
    const auto Nx = x_.size();
    const auto Ny = y_.size();
    // Check
    // NOTE: This check is only used in DEBUG builds and is removed otherwise.
    DNFATAL_IF(u0.size() != Nu, "[StateSpaceFilter]: Input has invalid dimensions: Must be dim(u) = " << Nu);
    // Define lumped matrices for the linear system used to solve for initial states of the system.
    Eigen::MatrixXd Phi{Eigen::MatrixXd::Zero(Nx+Ny,Nx+Ny)};
    Eigen::VectorXd theta{Eigen::VectorXd::Zero(Nx+Ny)};
    Eigen::VectorXd psi{Eigen::VectorXd::Zero(Nx+Ny)};
    Phi.block(0,0,Nx,Nx) = A_ - Eigen::MatrixXd::Identity(Nx,Nx);
    Phi.block(Nx,0,Ny,Nx) = C_;
    Phi.block(Nx,Nx,Ny,Ny) = -Eigen::MatrixXd::Identity(Ny,Ny);
    theta.segment(0,Nx) = B_*u0;
    theta.segment(Nx,Ny) = D_*u0;
    // Solve linear problem to compute initial system states: psi := [x0;  y0]
    psi = math::matrix_pseudo_inverse(Phi)*theta;
    // Extract initial system state.
    u_ = u0;
    x_ = psi.segment(0,Nx);
    y_ = psi.segment(Nx,Ny);
    // Optionally produce verbose console output.
    NINFO_IF(verbose_, "[StateSpaceFilter]: Initializing filter state using specified initial input:"
      << "\nPhi:\n" << Phi
      << "\nTheta:\n" << theta
      << "\npsi: " << psi.transpose()
      << "\nx0: " << x_.transpose()
      << "\ny0: " << y_.transpose()
    );
  }
  
  Eigen::VectorXd advance(const Eigen::VectorXd& u) override {
    DNFATAL_IF(u.size() != u_.size(), "[StateSpaceFilter]: Input has invalid dimensions: Must be dim(u) = " << u_.size());
    u_ = u;
    x_ = A_ * x_ + B_ * u_;
    y_ = C_ * x_ + D_ * u_;
    return y_;
  }
  
  void advance(const Eigen::VectorXd& u, Eigen::VectorXd& y) override {
    y = advance(u);
  }

private:
  
  static inline bool dimensions_match(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    return (A.rows() == B.rows() && A.cols() == B.cols());
  }
  
private:
  Eigen::MatrixXd A_{Eigen::MatrixXd::Zero(1,1)};
  Eigen::MatrixXd B_{Eigen::MatrixXd::Zero(1,1)};
  Eigen::MatrixXd C_{Eigen::MatrixXd::Zero(1,1)};
  Eigen::MatrixXd D_{Eigen::MatrixXd::Zero(1,1)};
  Eigen::VectorXd u_{Eigen::VectorXd::Zero(1)};
  Eigen::VectorXd x_{Eigen::VectorXd::Zero(1)};
  Eigen::VectorXd y_{Eigen::VectorXd::Zero(1)};
  bool verbose_{false};
};

} // namespace filters
} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_COMMON_FILTERS_HPP_

/* EOF */
