/***************************************************************
Copyright (c) 2023 Zichong Chen. All Rights Reserved.

FileName	: method_naive_sph.cpp
Author		: Zichong Chen, 
Version		: 0.1
***************************************************************/

#include "fluid_system.h"



void ParticleSystem::NaiveSPH23() {
	const float mass = param_[PMASS];
	const float sim_scale = param_[PSIMSCALE];
	const float sr = param_[PSMOOTHRADIUS];
	const float own_kernel = param_[PKERNELSELF];
	const float own_density = param_[PKERNELSELF] * mass;
	const float density0 = param_[PRESTDENSITY];
	const float viscosity = param_[PVISC];
	Vector3DF gravity = vec_[PPLANE_GRAV_DIR];
	const float stiffness = param_[PGASCONSTANT];
	//const float stiffness = 10;

	start.SetSystemTime(ACC_NSEC);
	InsertParticlesCPU(num_points());
	Record(PTIME_INSERT, "", start);

	//compute density and pressure 
	start.SetSystemTime(ACC_NSEC);

	for (int i = 0; i < num_points_; i++) {
		if (ptype_[i] != RIGID)
			continue;
		delta_[i] = own_kernel;
		Vector3DF ipos = pos_[i];
		const uint i_cell_index = particle_grid_cell_index_[i];
		if (i_cell_index != GRID_UNDEF) {
			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
				if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1) {
					continue;
				}
				int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
				while (j != GRID_UNDEF) {
					if (i == j || ptype_[j] != RIGID) {
						j = next_particle_index_in_the_same_cell_[j];
						continue;
					}
					Vector3DF pos_i_minus_j = ipos - pos_[j] * sim_scale;
					float dist = pos_i_minus_j.Length();
					if (0 <= dist && dist <= sr) {
						delta_[i] += kernelM4(dist, sr);
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
	}

	for (int i = 0; i < num_points_; i++) {
		if (ptype_[i] != SPH)
			continue;
		density_[i] = own_density;
		float rho = 0.0;
		Vector3DF ipos = pos_[i];
		const uint i_cell_index = particle_grid_cell_index_[i];
		if (i_cell_index != GRID_UNDEF) {
			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
				if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
					continue;
				int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
				while (j != GRID_UNDEF) {
					if (i == j || ptype_[j] == FLIP) {
						j = next_particle_index_in_the_same_cell_[j];
						continue;
					}
					Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
					float dist = pos_i_minus_j.Length();
					if (0 <= dist && dist <= sr) {
						//rho += kernelM4Lut(dist, sr);			// lookup table, slower
						//rho += kernelM4(dist, sr);
						if (ptype_[j] == SPH) {
							rho += kernelM4(dist, sr) * mass;
						}
						else if (ptype_[j] == RIGID) {
							float Psi = density0 / delta_[j];
							rho += kernelM4(dist, sr) * Psi;
						}
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		density_[i] += rho;
		//std::cout << density_[i] << " density(" << i << ")\n";
		pressure_[i] = max(0.0, stiffness * (density_[i] - density0));
		//pressure_[i] = stiffness * (density_[i] - density0);
		//cout << pressure_[i] << endl;
	}
	Record(PTIME_PRESS, "", start);

	// compute force
	start.SetSystemTime(ACC_NSEC);
	for (int i = 0; i < num_points_; i++) {
		if (ptype_[i] != SPH)
			continue;
		Vector3DF force(0, 0, 0);
		Vector3DF Fv(0, 0, 0);
		Vector3DF Fp(0, 0, 0);
		Vector3DF ipos = pos_[i];
		Vector3DF ivel = vel_eval_[i];
		float idensity = density_[i];
		float ipressure = pressure_[i];
		int cnt = 0;
		const uint i_cell_index = particle_grid_cell_index_[i];
		if (i_cell_index != GRID_UNDEF) {
			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
				if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1))
					continue;
				int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
				while (j != GRID_UNDEF) {
					if (i == j || ptype_[j] == FLIP) {
						j = next_particle_index_in_the_same_cell_[j];
						continue;
					}
					Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
					Vector3DF vel_j_minus_i = vel_eval_[j] - ivel;
					float dist = pos_i_minus_j.Length();
					if (0 <= dist && dist <= sr) {
						Vector3DF NablaW = pos_i_minus_j * kernelPressureGradLut(dist, sr);
						cnt += 1;
						float laplaceW = lap_kern_ * (sr - dist);

						if (ptype_[j] == SPH) {
							float jdensity = density_[j];
							Fv += vel_j_minus_i * (laplaceW / jdensity / idensity * mass);
							Fp += NablaW * ((pressure_[j] + ipressure) / jdensity / idensity / 2 * mass);
						}
						else if (ptype_[j] == RIGID) {
							float Psi = density0 / delta_[j];
							Fp += NablaW * (ipressure / idensity / idensity * Psi);
						}


					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		Fv *= viscosity * mass;
		Fp *= -(mass);
		force += Fv;
		force += Fp;
		force += gravity * mass;
		force_[i] = force;
		//cout << Fv.x << " " << Fv.y << " " << Fv.z << " Fv" << endl;
		//cout << Fp.x << " " << Fp.y << " " << Fp.z << " Fp" << endl;
		//cout << force_[i].x << " " << force_[i].y << " " << force_[i].z << " force" << endl;
		//cout << "cnt " << cnt << endl;
	}
	Record(PTIME_FORCE, "", start);

	// compute v and x
	start.SetSystemTime(ACC_NSEC);
	for (int i = 0; i < num_points_; i++) {
		if (particle_grid_cell_index_[i] == GRID_UNDEF || ptype_[i] != SPH)
			continue;
		vel_eval_[i] += force_[i] * (time_step_ / mass);
		//cout << vel_eval_[i].x << " " << vel_eval_[i].y << " " << vel_eval_[i].z << " (vel)   ";
		//cout << pos_[i].x << " " << pos_[i].y << " " << pos_[i].z << " (pos)  ";
		pos_[i] += vel_eval_[i] * (time_step_ / sim_scale);
		//cout << pos_[i].x << " " << pos_[i].y << " " << pos_[i].z << " (pos)" << endl;
	}

}

