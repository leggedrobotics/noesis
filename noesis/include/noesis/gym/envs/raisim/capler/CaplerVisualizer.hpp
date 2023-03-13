/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_RAISIM_CAPLER_VISUALIZER_HPP_
#define NOESIS_GYM_ENVS_RAISIM_CAPLER_VISUALIZER_HPP_

// C/C++
#include <unistd.h>
#include <cmath>
#include <iostream>

// Noesis
#include <noesis/framework/utils/macros.hpp>
#include <noesis/gym/envs/raisim/common/visualizer.hpp>
#include <noesis/gym/envs/raisim/capler/CaplerSimulation.hpp>

namespace noesis {
namespace gym {

struct CaplerVisConfig: noesis::gym::RaiSimVisualizerConfig {
  bool show_goal{true};
};

class CaplerVisualizer final: public noesis::gym::RaiSimVisualizer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  /*
   * Instantiation
   */

  explicit CaplerVisualizer(const CaplerVisConfig& config, CaplerSimulation& simulation):
    noesis::gym::RaiSimVisualizer(config),
    config_(config),
    simulation_(&simulation)
  {
    this->setWorld(&simulation.world());
    this->setWindowTitle("Capler");
  }
  
  explicit CaplerVisualizer(const noesis::gym::RaiSimVisualizerConfig& config, CaplerSimulation& simulation):
    noesis::gym::RaiSimVisualizer(config),
    simulation_(&simulation)
  {
    this->setWorld(&simulation.world());
    this->setWindowTitle("Capler");
  }
  
  explicit CaplerVisualizer(CaplerSimulation& simulation):
    CaplerVisualizer(noesis::gym::RaiSimVisualizerConfig(), simulation)
  {
  }
  
  ~CaplerVisualizer() final = default;
  
  /*
   * Operations
   */
  
  void update(const Position& W_r_WG) {
    std::lock_guard<std::mutex> lock(mutex_);
    W_r_WG_ = W_r_WG;
  }
  
protected:
  
  /*
   * Implementations
   */
  
  void setup() override {
    // Get local handle to visualizer singleton
    auto* vis = ::raisim::OgreVis::get();
    auto* sm = vis->getSceneManager();
    vis->setAmbientLight(Ogre::ColourValue(0.1, 0.1, 0.1));
    vis->getLight("default")->setVisible(false);
    // Set sky
    vis->addResourceDirectory(noesis::rootpath() + "/noesis/resources/materials/grid");
    vis->loadMaterialFile("grid.material");
    // Set default light configurations
    auto front = vis->addLight("front");
    front.first->setType(Ogre::Light::LT_POINT);
    front.first->setCastShadows(true);
    front.first->setPosition(Ogre::Vector3(100, 0, 75));
    front.first->setDiffuseColour(Ogre::ColourValue(0.5, 0.5, 0.5));
    front.first->setSpecularColour(Ogre::ColourValue(0.5, 0.5, 0.5));
    auto back = vis->addLight("back");
    back.first->setType(Ogre::Light::LT_POINT);
    back.first->setCastShadows(true);
    back.first->setPosition(Ogre::Vector3(-100, 0, 75));
    back.first->setDiffuseColour(Ogre::ColourValue(0.5, 0.5, 0.5));
    back.first->setSpecularColour(Ogre::ColourValue(0.5, 0.5, 0.5));
    auto right = vis->addLight("left");
    right.first->setType(Ogre::Light::LT_POINT);
    right.first->setCastShadows(true);
    right.first->setPosition(Ogre::Vector3(0, 100, 75));
    right.first->setDiffuseColour(Ogre::ColourValue(0.5, 0.5, 0.5));
    right.first->setSpecularColour(Ogre::ColourValue(0.5, 0.5, 0.5));
    auto left = vis->addLight("right");
    left.first->setType(Ogre::Light::LT_POINT);
    left.first->setCastShadows(true);
    left.first->setPosition(Ogre::Vector3(0, -100, 75));
    left.first->setDiffuseColour(Ogre::ColourValue(0.5, 0.5, 0.5));
    left.first->setSpecularColour(Ogre::ColourValue(0.5, 0.5, 0.5));
    // Set default shadow effects
    vis->getSceneManager()->setShadowFarDistance(10);
    vis->getSceneManager()->setShadowTextureSettings(4096, 3);
    vis->getSceneManager()->setShadowTechnique(Ogre::SHADOWTYPE_TEXTURE_ADDITIVE);
    // Set default contact visualization configurations
    vis->setContactVisObjectSize(0.01, 0.1);
    // Create the robot visual
    capler_ = vis->createGraphicalObject(simulation_->capler(), "capler");
    vis->select((*capler_)[0], false);
    vis->getCameraMan()->setStyle(::raisim::CameraStyle::CS_ORBIT);
    vis->getCameraMan()->setYawPitchDist(Ogre::Radian(0.0), Ogre::Radian(-1.6f), Ogre::Real(3.0f), false);
    // Create the terrain visual
    auto world = simulation_->getWorldType();
    if (world == CaplerSimulation::World::Grid) {
      terrain_ = vis->createGraphicalObject(simulation_->floor(), 100.0, "terrain", "terrain/grid");
    }
    // Create visualization elements
    std::vector<std::string> colors;
    colors = {"red", "green", "blue", "yellow", "magenta"};
    if (config_.show_goal) {
      goalPosition_ = vis->addVisualObject("goal_position", "sphereMesh", colors[1], {0.05, 0.05, 0.05}, false, ::raisim::OgreVis::RAISIM_OBJECT_GROUP);
    }
  }
  
  void update() override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_.show_goal) { updateGoal(W_r_WG_); }
  }

private:
  
  void updateGoal(const Position& W_r_WG) {
    // Goal markers
    goalPosition_->setPosition(W_r_WG_.x(), W_r_WG_.y(), W_r_WG_.z());
    goalPosition_->setOrientation(math::rotation_y(M_PI));
  }
  
private:
  // Configuration
  CaplerVisConfig config_;
  // Buffers
  Position W_r_WG_;
  std::mutex mutex_;
  // Simulated elements
  CaplerSimulation* simulation_{nullptr};
  std::vector<::raisim::GraphicObject>* capler_{nullptr};
  std::vector<::raisim::GraphicObject>* terrain_{nullptr};
  // Visualization-only elements
  ::raisim::VisualObject* goalPosition_{nullptr};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_RAISIM_CAPLER_VISUALIZER_HPP_

/* EOF */
