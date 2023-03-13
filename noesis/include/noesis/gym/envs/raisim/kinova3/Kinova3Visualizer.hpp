/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_KINOVA3_VISUALIZER_HPP_
#define NOESIS_GYM_ENVS_RAISIM_KINOVA3_VISUALIZER_HPP_

// C/C++
#include <unistd.h>
#include <cmath>
#include <iostream>

// Noesis
#include "noesis/framework/utils/macros.hpp"
#include "noesis/gym/envs/raisim/common/visualizer.hpp"
#include "noesis/gym/envs/raisim/kinova3/Kinova3Simulation.hpp"

namespace noesis {
namespace gym {

struct Kinova3VisConfig: noesis::gym::RaiSimVisualizerConfig {
  bool show_goal_pose{false};
  bool show_goal_force{false};
};

class Kinova3Visualizer final: public noesis::gym::RaiSimVisualizer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  /*
   * Instantiation
   */

  explicit Kinova3Visualizer(const Kinova3VisConfig& config, Kinova3Simulation& simulation):
    noesis::gym::RaiSimVisualizer(config),
    config_(config),
    simulation_(&simulation)
  {
    this->setWorld(&simulation.world());
    this->setWindowTitle("Kinova3");
  }
  
  explicit Kinova3Visualizer(const noesis::gym::RaiSimVisualizerConfig& config, Kinova3Simulation& simulation):
    noesis::gym::RaiSimVisualizer(config),
    simulation_(&simulation)
  {
    this->setWorld(&simulation.world());
    this->setWindowTitle("Kinova3");
  }
  
  explicit Kinova3Visualizer(Kinova3Simulation& simulation):
    Kinova3Visualizer(noesis::gym::RaiSimVisualizerConfig(), simulation)
  {
  }
  
  ~Kinova3Visualizer() final = default;
  
  /*
   * Operations
   */
  
  void update(const RotationMatrix& R_G, const Position& r_G) {
    std::lock_guard<std::mutex> lock(mutex_);
    R_G_ = R_G;
    r_G_ = r_G;
  }
  
protected:
  
  /*
   * Implementations
   */
  
  void setup() override {
    // Get local handle to visualizer singleton
    auto* vis = ::raisim::OgreVis::get();
    vis->setAmbientLight(Ogre::ColourValue(0.1, 0.1, 0.1));
    // Set sky
    vis->addResourceDirectory(noesis::rootpath() + "/noesis/resources/materials/grid");
    vis->loadMaterialFile("grid.material");
    // Set default contact visualization configurations
    vis->setContactVisObjectSize(0.01, 0.1);
    // Create the robot visual
    kinova3_ = vis->createGraphicalObject(simulation_->kinova3(), "kinova3");
    vis->select((*kinova3_)[0], false);
    vis->getCameraMan()->setStyle(::raisim::CameraStyle::CS_ORBIT);
    vis->getCameraMan()->setYawPitchDist(Ogre::Radian(0.0), Ogre::Radian(-1.6f), Ogre::Real(3.0f), false);
    // Create the terrain visual
    auto terrain = simulation_->getWorldType();
    if (terrain == Kinova3Simulation::World::Grid) {
      world_ = vis->createGraphicalObject(simulation_->floor(), 100.0, "terrain", "terrain/grid");
    }
    // Create visualization elements
    eePose_.create("end_effector_pose", vis);
    if (config_.show_goal_pose) { goalPose_.create("goal_pose", vis); }
    if (config_.show_goal_force) {
      goalForce_ = vis->addVisualObject(
        "goal_force", "arrowMesh", "cyan",
        {0.1, 0.1, 0.2}, false,
        ::raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
  }
  
  void update() override {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& r_ee = simulation_->getPositionWorldToEndEffectorInWorldFrame();
    const auto& R_ee = simulation_->getOrientationWorldToEndEffector();
    eePose_.update(r_ee, R_ee);
    if (config_.show_goal_pose) { goalPose_.update(r_G_, R_G_); }
    if (config_.show_goal_force) {
      const auto& r_ee = simulation_->getPositionWorldToEndEffectorInWorldFrame();
      const auto R_f = math::vector_to_rotation_matrix(f_G_.normalized() - r_ee.normalized());
      goalForce_->setPosition(r_ee);
      goalForce_->setOrientation(R_f);
      goalForce_->setScale(0.03, 0.03, f_G_.norm());
    }
  }
  
private:
  // Configuration
  Kinova3VisConfig config_;
  // Buffers
  RotationMatrix R_G_{RotationMatrix::Identity()};
  Position r_G_{Position::Zero()};
  Vector3 f_G_{Vector3::Zero()};
  std::mutex mutex_;
  // Simulated elements
  Kinova3Simulation* simulation_{nullptr};
  std::vector<::raisim::GraphicObject>* kinova3_{nullptr};
  std::vector<::raisim::GraphicObject>* world_{nullptr};
  // Visualization-only elements
  raisim::CoordinateFrame eePose_;
  raisim::CoordinateFrame goalPose_;
  ::raisim::VisualObject* goalForce_{nullptr};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_KINOVA3_VISUALIZER_HPP_

/* EOF */
