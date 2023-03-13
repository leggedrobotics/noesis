/*!
 * @author    HaoChih Lin
 * @email     hlin@ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_CLASSIC_ACROBOT_ACROBOT_VISUALIZER_HPP_
#define NOESIS_GYM_ENVS_CLASSIC_ACROBOT_ACROBOT_VISUALIZER_HPP_

// Environments
#include "noesis/framework/system/process.hpp"
#include "noesis/gym/envs/classic/common/visualizer.hpp"
#include "noesis/gym/envs/classic/acrobot/AcrobotEnvironment.hpp"

namespace noesis {
namespace gym {

class AcrobotVisualizer final : public SfmlVisualizer<float>
{
public:
  
  // Alias
  using Base = SfmlVisualizer<float>;
  using Environment = typename Base::Environment;
  
  /*
   * Instantiation
   */
  
  explicit AcrobotVisualizer(Environment* environment, const SfmlVisualizerConfig& config=SfmlVisualizerConfig()):
    Base(environment,
      std::bind(&AcrobotVisualizer::setup, this),
      std::bind(&AcrobotVisualizer::update, this),
      config)
  {
    auto* env = dynamic_cast<AcrobotEnvironment*>(this->env());
    NFATAL_IF(!env, "[" << namescope() << "]: This environment does not point to a AcrobotEnvironment!");
    setFramesPerSecond(15.0f);
  }
  
  ~AcrobotVisualizer() final = default;

protected:
  
  void setup() {
    // define shape and color for arrow (torque)
    arrowTexture_.loadFromFile(noesis::rootpath() + "/noesis/resources/images/rotary_arrow.png");
    sprite_.setOrigin(arrowTexture_.getSize().x / 2.0f, arrowTexture_.getSize().y / 2.0f);
    sprite_.setPosition(joint2_.getPosition());
    sprite_.setTexture(arrowTexture_);
    sprite_.setColor(sf::Color(200, 75, 66));
    // define shape and color for link 1
    link1_.setOrigin(offset_, offset_);
    link1_.setPosition((float) getWindowWidth() / 2.0f, (float) getWindowHeight() / 2.0f);
    link1_.setFillColor(sf::Color(180, 180, 67));
    // define shape and color for joint 1
    joint1_.setOrigin(offset_, offset_);
    joint1_.setPosition(link1_.getPosition());
    joint1_.setFillColor(sf::Color::Black);
    // define shape and color for link 2
    link2_.setOrigin(offset_, offset_);
    link2_.setPosition(link1_.getPosition().x, link1_.getPosition().y + link1_.getSize().y - 2.0f * offset_);
    link2_.setFillColor(sf::Color(180, 180, 67));
    // define shape and color for joint 2
    joint2_.setOrigin(offset_, offset_);
    joint2_.setPosition(link2_.getPosition());
    joint2_.setFillColor(sf::Color::Black);
  }
  
  void update() {
    auto* env = dynamic_cast<AcrobotEnvironment*>(this->env());
    NFATAL_IF(!env, "[" << namescope() << "]: This environment does not point to a AcrobotEnvironment!");
    auto q = env->getPositions();
    auto torque = env->getTorque();
    auto max_torque = env->getMaxTorque();
    // set orientation
    link1_.setRotation((float) (q(0) * 180.0 / M_PI));
    link2_.setPosition(link1_.getPosition().x - (link1_.getSize().y - 2.0f * offset_) * (float) sin(q(0)),
                       link1_.getPosition().y + (link1_.getSize().y - 2.0f * offset_) * (float) cos(q(0)));
    link2_.setRotation((float) ((q(0) + q(1)) * 180.0 / M_PI));
    joint2_.setPosition(link2_.getPosition());
    sprite_.setPosition(joint2_.getPosition());
    // set arrow direction
    float scale = torque/5.0f;
    if (torque > 0) {
      sprite_.setScale(scale, -scale);
    } else {
      sprite_.setScale(scale, scale);
    }
    // update texture
    getTexture().clear(sf::Color::White);
    getTexture().draw(sprite_);
    getTexture().draw(link1_);
    getTexture().draw(joint1_);
    getTexture().draw(link2_);
    getTexture().draw(joint2_);
    getTexture().display();
    // update window
    getWindow().clear(sf::Color::White);
    getWindow().draw(sf::Sprite(getTexture().getTexture()));
  }
  
private:
  sf::RectangleShape link1_{sf::Vector2f(20.0f, 100.0f)};
  sf::RectangleShape link2_{sf::Vector2f(20.0f, 100.0f)};
  sf::CircleShape joint1_{10.0};
  sf::CircleShape joint2_{10.0};
  sf::Sprite sprite_;
  sf::Texture arrowTexture_;
  const float offset_{10.0};
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_CLASSIC_ACROBOT_ACROBOT_VISUALIZER_HPP_

/* EOF */
