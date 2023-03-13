/*!
 * @author    JoonHo Lee
 * @email     junja94@gmail.com
 * @author    Mayank Mittal
 * @email     mittalma@ethz.ch
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#ifndef NOESIS_GYM_ENVS_CLASSIC_CARTPOLE_CARTPOLE_VISUALIZER_HPP_
#define NOESIS_GYM_ENVS_CLASSIC_CARTPOLE_CARTPOLE_VISUALIZER_HPP_

// Environments
#include "noesis/framework/system/process.hpp"
#include "noesis/gym/envs/classic/common/visualizer.hpp"
#include "noesis/gym/envs/classic/cartpole/CartpoleEnvironment.hpp"

namespace noesis {
namespace gym {

class CartpoleVisualizer final : public SfmlVisualizer<float>
{
public:
  
  // Alias
  using Base = SfmlVisualizer<float>;
  using Environment = typename Base::Environment;
  
  /*
   * Instantiation
   */
  
  explicit CartpoleVisualizer(Environment* environment, const SfmlVisualizerConfig& config=SfmlVisualizerConfig()):
    Base(environment,
      std::bind(&CartpoleVisualizer::setup, this),
      std::bind(&CartpoleVisualizer::update, this),
      config)
  {
    auto* env = dynamic_cast<CartpoleEnvironment*>(this->env());
    NFATAL_IF(!env, "[" << namescope() << "]: This environment does not point to a CartpoleEnvironment!");
    setFramesPerSecond(50.0f);
  }
  
  ~CartpoleVisualizer() final = default;

protected:
  
  void setup() {
    // define shape and color for arrow (torque)
    arrowTexture_.loadFromFile(noesis::rootpath() + "/noesis/resources/images/linear_arrow.png");
    sprite_.setTexture(arrowTexture_);
    sprite_.setColor(sf::Color(200, 75, 66));
    sprite_.setOrigin(0.0f, arrowTexture_.getSize().y / 2.0f);
    // define shape and color for the track
    track_.setPosition(-5.0f * getWindowWidth(), 3.0f * getWindowHeight() / 5.0f);
    track_.setFillColor(sf::Color(200, 200, 200, 255));
    // define shape and color for the cart
    cart_.setOrigin(25.0f, 15.0f);
    cart_.setFillColor(sf::Color::Black);
    // define shape and color for the pole
    pole_.setOrigin((float) 5.0f, 150.0f / 150.0f * 145.0f);
    pole_.setFillColor(sf::Color(180, 180, 67, 255));
    // define shape and color for the joint
    center_.setOrigin(2.0f, 2.0f);
    center_.setFillColor(sf::Color::Black);
  }
  
  void update() {
    auto* env = dynamic_cast<CartpoleEnvironment*>(this->env());
    NFATAL_IF(!env, "[" << namescope() << "]: This environment does not point to a CartpoleEnvironment!");
    auto q = env->getPositions();
    auto force = env->getForce();
    // scaling factors for background
    float scaledX = (float) q(0) * getWindowWidth() / 5.0f + getWindowWidth() / 2.0f;
    float trackY = 3.0f * getWindowHeight() / 5.0f;
    // cart
    cart_.setRotation(0.0f);
    cart_.setPosition(scaledX, trackY);
    // pole
    pole_.setPosition(scaledX, trackY);
    pole_.setRotation((float) (q(1) * 180.0 / M_PI));
    // center
    center_.setPosition(scaledX, trackY);
    float scale = (float) force / 2.0f;
    if (force > 0) {
      sprite_.setPosition(scaledX + 15.0f, trackY);
      sprite_.setScale(scale, 1);
    } else {
      sprite_.setPosition(scaledX - 15.0f, trackY);
      sprite_.setScale(scale, 1);
    }
    // update texture
    getTexture().clear(sf::Color::White);
    getTexture().draw(track_);
    getTexture().draw(sprite_);
    getTexture().draw(cart_);
    getTexture().draw(pole_);
    getTexture().draw(center_);
    getTexture().display();
    // update window
    getWindow().clear(sf::Color::White);
    getWindow().draw(sf::Sprite(getTexture().getTexture()));
  }
  
private:
  sf::RectangleShape cart_{sf::Vector2f(50.0f, 30.0f)};
  sf::RectangleShape pole_{sf::Vector2f(10.0f, 150.0f)};
  sf::RectangleShape track_{sf::Vector2f(6000.0f, 10.0f)};
  sf::CircleShape center_{2.0f};
  sf::Sprite sprite_;
  sf::Texture arrowTexture_;
};

} // namespace gym
} // namespace noesis

#endif // NOESIS_GYM_ENVS_CLASSIC_CARTPOLE_CARTPOLE_VISUALIZER_HPP_

/* EOF */
