/*!
 * @author    Vassilios Tsounis
 * @email     tsounisv@ethz.ch
 *
 * Copyright (C) 2023 Robotic Systems Lab, ETH Zurich.
 * All rights reserved.
 * http://www.rsl.ethz.ch/
 */
#pragma once

// Noesis
#include <noesis/gym/core/Environment.hpp>

namespace example {

class Environment final: public noesis::gym::Environment<double>
{
public:
  
  using Base = noesis::gym::Environment<double>;
  
  explicit Environment(std::string str="foo", double dbl=3.14159, int i=42, bool b=false):
    Base(
      {{"obs", {3,2}}}, // Define observations
      {{"act", {3,2}}}, // Defines actions
      {"rews"}, // The reward/task vector
      {"b*i"}, // Example metric
      0.997, // Discount-factor (gamma)
      static_cast<size_t>(100u), // max episode steps
      1u, // History size
      1u, // History size
      "Environment", "/Example", true), // name, scope, verbose
    dbl_(dbl),
    i_(i),
    b_(b)
  {
    str_ = std::make_unique<std::string>(str);
    NNOTIFY_IF(isVerbose(), "[Example]: New instance at: " << std::hex << this);
    NNOTIFY_IF(isVerbose(), "[Example]: Data:"
      << "\nstr: " << *str_
      << "\ndbl: " << dbl_
      << "\ni: " << i_
      << "\nb: " << std::boolalpha << b_
    );
  }
  
  ~Environment() final = default;
  
private:
  
  void setup() override { /* NOOP: Add what would configured depending on hyper-parameters. */ }
  
  void randomize(int seed) override { /* NOOP: Add domain randomization here. */ }
  
  bool initialize(Observations& observations, Terminations& terminations) override {
    NWARNING("Initialization!");
    observations[0].setZero();
    terminations.back().type = Termination::Type::Unterminated;
    return true;
  }
  
  bool transition(
      const Actions& actions,
      Observations& observations,
      Rewards& rewards,
      Terminations& terminations,
      Metrics& metrics) override {
    NWARNING("Transition!");
    observations[0] += actions[0];
    rewards.setConstant(actions[0][0]);
    if (steps() > 5) {
      NWARNING("Termination!");
      terminations.back().type = Termination::Type::TerminalState;
      terminations.back().value = -1;
      NWARNING("[Example]: Data:"
        << "\nstr: " << *str_
        << "\ndbl: " << dbl_
        << "\ni: " << i_
        << "\nb: " << std::boolalpha << b_
      );
    }
    return true;
  }
  
  std::string info() const override { return "[example::Environment]:" + observations()[0].info(); }

private:
  // NOTE: This member is a unique_ptr in order to test what the system does w/ non-copyable types
  std::unique_ptr<std::string> str_;
  double dbl_;
  int i_;
  bool b_;
};

} // namespace example

/* EOF */
