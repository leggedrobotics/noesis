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

// C/C++
#include <iostream>
#include <thread>

// Noesis
#include <noesis/noesis.hpp>

// function for multi-thread testing (thread 1)
void info_thread() {
  for(int i=0; i<=200; i++) {
    NINFO("Extra thread 1, no. " + std::to_string(i));
  }
}

// function for multi-thread testing (thread 2)
void notify_thread() {
  for(int i=0; i<=200; i++) {
    NNOTIFY("Extra thread 2, no. " + std::to_string(i));
  }
}


int main(int argc, char **argv)
{
  noesis::init("noesis_message_example");

  // General Testing
  NINFO("This is a info message!");
  NNOTIFY("This is a notification message!");
  NWARNING("This is a warning message!");
  NERROR("This is a error message!");

  // Manually call write() func
  noesis::log::message::logger->write();

  // Test autosave (buffer size: 20 lines)
  for(int i=0; i<30; i++) {
    NERROR("Testing: " + std::to_string(i));
  }

  // Multi-thread testing
  std::thread thread1(info_thread);
  std::thread thread2(notify_thread);
  for(int i=0; i<200; i++) {
    NERROR("Main thread, no. " + std::to_string(i));
  }

  // Cleanup worker threads
  thread1.join();
  thread2.join();

  // Termination from fatal error
  sleep(5);
  NFATAL("This is a fatal error message!");
  return 0;
}

/* EOF */
