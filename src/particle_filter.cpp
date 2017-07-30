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

  // number of particles
  num_particles = 100;

  // random engine
  default_random_engine gen;

  // Create normal distributions for x, y and psi.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // initialize random particles
  for (int i = 0; i < num_particles; ++i) {
    Particle particle = {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0};
    particles.push_back(particle);
  }

  // set initialized
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
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; ++i) {
    Particle particle = particles[i];
    double cos_theta = cos(particle.theta), sin_theta = sin(particle.theta);

    if (0.0001 > fabs(yaw_rate)) {
      particle.x += velocity * delta_t * cos_theta;
      particle.y += velocity * delta_t * sin_theta;
    } else {
      particle.x += (velocity / yaw_rate) *
                    (sin(particle.theta + yaw_rate * delta_t) - sin_theta);
      particle.y += (velocity / yaw_rate) *
                    (cos_theta - cos(particle.theta + yaw_rate * delta_t));
      particle.theta += yaw_rate * delta_t;
    }

    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_theta(gen);

    particles[i] = particle;
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
    LandmarkObs observation = observations[j];
    double minDistance = std::numeric_limits<double>::max();

    for (unsigned i = 0; i < predicted.size(); ++i) {
      LandmarkObs prediction = predicted[i];
      double distance =
          dist(observation.x, observation.y, prediction.x, prediction.y);

      if (minDistance > distance) {
        minDistance = distance;
        observation.id = prediction.id;
      }
    }
    observations[j] = observation;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian
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

  for (unsigned i = 0; i < particles.size(); ++i) {
    Particle particle = particles[i];
    particle.weight = 1.0;

    // get landmarks within range
    std::vector<LandmarkObs> landmarks;
    for (unsigned j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      Map::single_landmark_s lm = map_landmarks.landmark_list[j];
      double l_dist = dist(particle.x, particle.y, lm.x_f, lm.y_f);
      // if landmark in sensor_range, add as LandmarkObs to landmark
      if (l_dist <= sensor_range) {
        landmarks.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }
    }

    // transform observations to map space
    std::vector<LandmarkObs> transformedObservations;
    double p_theta = particle.theta, p_x = particle.x, p_y = particle.y;
    double cos_theta = cos(p_theta);
    double sin_theta = sin(p_theta);

    for (unsigned i = 0; i < observations.size(); ++i) {
      LandmarkObs observationOnMap;
      double o_x = observations[i].x, o_y = observations[i].y;
      observationOnMap.x = (o_x * cos_theta) - (o_y * sin_theta) + p_x;
      observationOnMap.y = (o_x * sin_theta) + (o_y * cos_theta) + p_y;

      transformedObservations.push_back(observationOnMap);
    }

    // find nearest neighbor
    dataAssociation(landmarks, transformedObservations);

    // calculate weight
    for (unsigned j = 0; j < transformedObservations.size(); ++j) {
      LandmarkObs o = transformedObservations[j];
      Map::single_landmark_s lm = map_landmarks.landmark_list.at(o.id - 1);

      double x = pow((o.x - lm.x_f), 2) / (2 * pow(std_landmark[0], 2));
      double y = pow((o.y - lm.y_f), 2) / (2 * pow(std_landmark[1], 2));

      particle.weight *=
          exp(-(x + y)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    }

    weights.push_back(particle.weight);
    particles[i] = particle;
  }
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional
  //	to their weight.
  // NOTE: You may find std::discrete_distribution helpful here:
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  random_device rd;
  mt19937 gen(rd());
  discrete_distribution<> dist_index(weights.begin(), weights.end());

  std::vector<Particle> resampled;
  resampled.resize(num_particles);

  for (unsigned i = 0; i < particles.size(); ++i) {
    int index = dist_index(gen);
    resampled[i] = particles[index];
  }

  particles = resampled;
  weights.clear();
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
