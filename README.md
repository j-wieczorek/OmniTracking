# OmniTracking
Kernel Particle Filter implementation for omnidirectional camera based on 'Tracking unknown moving targets on omnidirectional vision' by Y. Shu-Ying, G. WeiMin and Z. Cheng

Required libraries:
	- OpenCV 3.0
	- Eigen 3.6

This project implements one of the object-tracking methods. Optical flow is used in order to detect object for tracking and then it's position is being continously updated by particle filter.

Software is divided into two classe - OpticalFlow for optical flow computation and KPF for position esitimation. 
Project is still under development, so at the moment it may not work as it should be expected.