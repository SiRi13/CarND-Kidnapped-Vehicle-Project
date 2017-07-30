/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles. Initialize all particles to first
  //	position (based on estimates of x, y, theta and their uncertainties from
  //	GPS) and all weights to 1. Add random Gaussian noise to each
  //	particle.
  // NOTE: Consult particle_filter.h for more information about this
  //	method (and others in this file).
  num_particles = 100;
  // random engine
  default_random_engine gen;
  // Create normal distributions for x, y and psi.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle p = {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0};
    particles.push_back(p);
  }
  weights.resize(num_particles);
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // random engine
  default_random_engine gen;
  // Create normal distributions for x, y and psi.
  normal_distribution<double> dist_v(velocity, std_pos[0]);
  normal_distribution<double> dist_yaw(yaw_rate, std_pos[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle p = particles[i];
    double v = dist_v(gen);
    double yaw = dist_yaw(gen);

    if (fabs(0.0001) > yaw_rate) {
      p.x += v * delta_t * cos(p.theta);
      p.y += v * delta_t * sin(p.theta);
    } else {
      p.x += (v / yaw) * (sin(p.theta + yaw * delta_t) - sin(p.theta));
      p.y += (v / yaw) * (cos(p.theta) - cos(p.theta + yaw * delta_t));
      p.theta += yaw * delta_t;
    }

    particles[i] = p;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // Find the predicted measurement that is closest to each observed
  //	measurement and assign the observed measurement to this particular
  //	landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  //	probably find it useful to implement this method and use it as a helper
  //	during the updateWeights phase.

  for (unsigned j = 0; j < observations.size(); ++j) {
    LandmarkObs o = observations[j];
    double minDistance = std::numeric_limits<double>::max();

    for (unsigned i = 0; i < predicted.size(); ++i) {
      LandmarkObs p = predicted[i];
      double distance = dist(o.x, o.y, p.x, p.y);

      if (minDistance > distance) {
        minDistance = distance;
        o.id = p.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  //	distribution. You can read more about this distribution here:
  //		https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  //	particles are located according to the MAP'S coordinate system. You will
  //	need to transform between the two systems. Keep in mind that this
  //	transformation requires both rotation AND translation (but no scaling).
  //	The	following is a good resource for the theory:
  // https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //	and the following is a good resource for the actual equation to
  //	implement look at equation 3.33 http://planning.cs.uiuc.edu/node99.html

  unsigned m = particles.size();
  for (unsigned i = 0; i < m; ++i) {
    Particle p = particles[i];

    // get landmarks within range
    std::vector<LandmarkObs> landmarks;
    for (unsigned j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      Map::single_landmark_s l = map_landmarks.landmark_list[j];
      double l_dist = dist(p.x, p.y, l.x_f, l.y_f);
      if (l_dist <= sensor_range) {
        landmarks.push_back(LandmarkObs{l.id_i, l.x_f, l.y_f});
      }
    }

    // transform observations to map space
    std::vector<LandmarkObs> tObs = transformToMapSpace(p, observations);
    // find nearest neighbor
    dataAssociation(landmarks, tObs);
    // TODO: calculate weight
  }
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional
  //	to their weight.
  // NOTE: You may find std::discrete_distribution helpful here:
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<Particle> p;

  default_random_engine gen;
  discrete_distribution<unsigned> dist_index(0, num_particles);

  unsigned index = dist_index(gen);
  double beta = 0.0;
  double mw = *max_element(weights.begin(), weights.end());
  uniform_real_distribution<double> dist_beta(0.0, 2.0 * mw);

  for (int i = 0; i < num_particles; ++i) {
    beta += dist_beta(gen);
    while (beta > weights[index]) {
      beta -= weights[index];
      index = ++index % num_particles;
    }
    p.push_back(particles[index]);
  }
  particles = p;
}

std::vector<LandmarkObs> ParticleFilter::transformToMapSpace(
    Particle particle, std::vector<LandmarkObs> observations) {
  std::vector<LandmarkObs> transformedObservations;
  for (unsigned i = 0; i < observations.size(); ++i) {
    double o_x = observations[i].x, o_y = observations[i].y;
    double p_theta = particle.theta, p_x = particle.x, p_y = particle.y;
    double x = (o_x * cos(p_theta)) - (o_y * sin(p_theta)) + p_x;
    double y = (o_x * sin(p_theta)) + (o_y * cos(p_theta)) + p_y;
    LandmarkObs t_obs = {observations[i].id, x, y};
    transformedObservations.push_back(t_obs);
  }
  return transformedObservations;
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and
  // association's (x,y) world coordinates mapping to associations: The
  // landmark id that goes along with each listed association sense_x: the
  // associations x mapping already converted to world coordinates sense_y:
  // the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
