/*!
 * @author    JoonHo Lee
 * @email     junja94@gmail.com
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_CLASSIC_PENDULUM_PENDULUM_VISUALIZER_HPP_
#define NOESIS_GYM_ENVS_CLASSIC_PENDULUM_PENDULUM_VISUALIZER_HPP_

// Environments
#include "noesis/framework/system/process.hpp"
#include "noesis/gym/envs/classic/common/visualizer.hpp"
#include "noesis/gym/envs/classic/pendulum/PendulumEnvironment.hpp"

namespace noesis {
namespace gym {

class PendulumVisualizer final : public SfmlVisualizer<float>
{
public:
  
  // Alias
  using Base = SfmlVisualizer<float>;
  using Environment = typename Base::Environment;
  
  /*
   * Instantiation
   */
  
  explicit PendulumVisualizer(Environment* environment, const SfmlVisualizerConfig& config=SfmlVisualizerConfig()):
    Base(environment,
      std::bind(&PendulumVisualizer::setup, this),
      std::bind(&PendulumVisualizer::update, this),
      config),
    pendulum_(sf::Vector2f(20.0f, 150.0f)),
    center_(2.0f)
  {
    auto* env = dynamic_cast<PendulumEnvironment*>(this->env());
    NFATAL_IF(!env, "[" << namescope() << "]: This environment does not point to a PendulumEnvironment!");
    setFramesPerSecond(30.0f);
  }
  
  ~PendulumVisualizer() final = default;

protected:
  
  void setup() {
    // define shape and color for arrow (torque)
    arrow_.loadFromFile(noesis::rootpath() + "/noesis/resources/images/rotary_arrow.png");
    sprite_.setOrigin(arrow_.getSize().x / 2.0f, arrow_.getSize().y / 2.0f);
    sprite_.setPosition(getWindowWidth() / 2.0f, getWindowHeight() / 2.0f);
    sprite_.setTexture(arrow_);
    sprite_.setColor(sf::Color(200, 75, 66));
    // define shape and color of the pendulum pole
    pendulum_.setOrigin((float) 10.0f, (float) 8.0f);
    pendulum_.setPosition((float) getWindowWidth() / 2.0f, (float) getWindowHeight() / 2.0f);
    pendulum_.setFillColor(sf::Color(180, 180, 67));
    // define shape and color of the joint
    center_.setOrigin(2.0f, 2.0f);
    center_.setPosition((float) getWindowWidth() / 2.0f, (float) getWindowHeight() / 2.0f);
    center_.setFillColor(sf::Color::Black);
  }
  
  void update() {
    auto* env = dynamic_cast<PendulumEnvironment*>(this->env());
    NFATAL_IF(!env, "[" << namescope() << "]: This environment does not point to a PendulumEnvironment!");
    auto theta = env->getPosition();
    auto torque = env->getTorque();
    auto max_torque = env->getMaxTorque();
    // pendulum
    pendulum_.setRotation((float) ((theta + M_PI) * 180.0 / M_PI));
    // scale objects
    pendulum_.setScale(getWindowScale(), getWindowScale());
    center_.setScale(getWindowScale(), getWindowScale());
    // set arrow direction
    auto scale = static_cast<float>(0.5 * torque/max_torque);
    if (torque > 0) {
      sprite_.setScale(scale, -scale);
    } else {
      sprite_.setScale(scale, scale);
    }
    // update texture
    getTexture().clear(sf::Color::White);
    getTexture().draw(sprite_);
    getTexture().draw(pendulum_);
    getTexture().draw(center_);
    getTexture().display();
    // update window
    getWindow().clear(sf::Color::White);
    getWindow().draw(sf::Sprite(getTexture().getTexture()));
  }
  
private:
  sf::RectangleShape pendulum_;
  sf::CircleShape center_;
  sf::Sprite sprite_;
  sf::Texture arrow_;
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_CLASSIC_PENDULUM_PENDULUM_VISUALIZER_HPP_

/* EOF */
