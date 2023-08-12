/***************************************************************
Copyright (c) 2023 Zichong Chen. All Rights Reserved.

FileName	: methohd_cisph.cpp
Author		: Zichong Chen, 
Version		: 0.1
***************************************************************/

#include "fluid_system.h"


/**************************************************************************
fluid-rigid particles coupling interaction using CISPH
**************************************************************************/
void ParticleSystem::CISPH23(float sr) {
	const float mass = param_[PMASS];
	const float density0 = param_[PRESTDENSITY];
	const float sim_scale = param_[PSIMSCALE];
	const float viscosity = param_[PVISC];
	const float own_kernel_contribution = param_[PKERNELSELF];
	const float own_density_contribution = param_[PKERNELSELF] * mass;
	const float stiffness = param_[PGASCONSTANT];
	const float LaplaceW = lap_kern_;
	Vector3DF gravity = vec_[PPLANE_GRAV_DIR];
	const float eps = 1.0e-16;
	float C = 0.;
	float avg_density_divergence_err = 0;
	int iterations = 0;

	start.SetSystemTime(ACC_NSEC);
	// find neighbor
	InsertParticlesCPU(num_points());
	Record(PTIME_INSERT, "", start);

	start.SetSystemTime(ACC_NSEC);
	// density and pressure
	for (int i = 0; i < num_points_; i++) {	// rigid variable (delta_)
		if (ptype_[i] != RIGID)
			continue;
		Vector3DF ipos = pos_[i];
		float sumWik = own_kernel_contribution;
		const uint i_cell_index = particle_grid_cell_index_[i];
		if (i_cell_index != GRID_UNDEF) {
			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
				if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1) {
					continue;
				}
				int k = grid_head_cell_particle_index_array_[neighbor_cell_index];
				while (k != GRID_UNDEF) {
					if (i == k || ptype_[k] != RIGID) {
						k = next_particle_index_in_the_same_cell_[k];
						continue;
					}
					Vector3DF kpos = pos_[k];
					Vector3DF pos_i_minus_k = (ipos - kpos) * sim_scale;
					float dist = pos_i_minus_k.Length();
					if (0 <= dist && dist <= sr) {
						float W = kernelM4(dist, sr);
						sumWik += W;
					}
					k = next_particle_index_in_the_same_cell_[k];
				}
			}
		}
		delta_[i] = sumWik;
	}

	for (int i = 0; i < num_points_; i++) {
		if (ptype_[i] != SPH)
			continue;
		neighbor_has_unwant_[i] = false;
		density_[i] = own_density_contribution;
		Vector3DF ipos = pos_[i];
		const uint i_cell_index = particle_grid_cell_index_[i];
		if (i_cell_index != GRID_UNDEF) {
			int cnt = 0;
			for(int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
				if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
					continue;
				int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
				while (j != GRID_UNDEF) {
					if (i == j || ptype_[j] == FLIP) {
						j = next_particle_index_in_the_same_cell_[j];
						continue;
					}
					Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;			// sim scale
					float dist = pos_i_minus_j.Length();
					if (0 <= dist && dist <= sr) {
						float W = kernelM4(dist, sr);
						//std::cout << "density W:" << W << std::endl;
						if (ptype_[j] == SPH) {
							Vector3DF NablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
							Vector3DF mnw = NablaW * mass;
							density_[i] += mass * W;
						}
						else if (ptype_[j] == RIGID) {
							neighbor_has_unwant_[i] = true;
							float Psi = density0 / delta_[j];
							density_[i] += Psi * W;
						}
						cnt++;
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		pressure_[i] = max(0.0f, stiffness * (density_[i] - density0));
	}

	// force and velocity
	for (int i = 0; i < num_points_; i++) {
		if (ptype_[i] != SPH)
			continue;
		Vector3DF force(0, 0, 0);
		Vector3DF ipos = pos_[i];
		Vector3DF iveleval = vel_eval_[i];
		float idensity = density_[i];
		float ipressure = pressure_[i];
		Vector3DF pterm(0, 0, 0);

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
					Vector3DF pos_i_minus_j = (ipos - pos_[j])*sim_scale;
					float dist = pos_i_minus_j.Length();
					float jpressure = pressure_[j];
					float jdensity = density_[j];
					if (dist <= sr && dist >= 0) {
						Vector3DF vel_j_minus_i = (vel_eval_[j] - iveleval);
						float h_minus_r = sr - dist;
						if (ptype_[j] == SPH) {
							force += vel_j_minus_i * (h_minus_r / jdensity / idensity) * mass * viscosity;
							pterm += pos_i_minus_j * ((ipressure + jpressure) / (2 * idensity*jdensity) * kernelPressureGrad(dist, sr)) * mass;
							//pterm += pos_i_minus_j *(jpressure / idensity / jdensity * kernelPressureGrad(dist, sr)*mass);
						}
						else if (ptype_[j] == RIGID) {
							float Psi = density0 / delta_[j];
							pterm += pos_i_minus_j * (ipressure / idensity / idensity * kernelPressureGrad(dist, sr)) * Psi;
							// viscosity from rigid
							force += (Vector3DF(0,0,0)-iveleval) * (h_minus_r / idensity / delta_[j] * (viscosity*(float)0.4));	// mu = mu0
						}
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		force *= mass * LaplaceW;		// viscosity force
		force -= pterm * (mass);				// pressure force
		//if (toggle_[PUSELOADEDSCENE] == true && (pos_[i].x > 60. || pos_[i].z < -60.)) {
		//	// no gravity
		//}
		
		if (toggle_[PUSELOADEDSCENE] == true && ADDITIONSCENE == 2 ) { //&& pos_[i].x > 60
			//if (pos_[i].x >= 100)
			//	gravity.Set(-8, 0, 0);
			//else if (pos_[i].x < 100) {
			//	//single
			//	gravity.Set(-3, -2, 0);
			//	//full
			//	//gravity.Set(-3, 0, 0);
			//}

			// ablation
			gravity.Set(0, -9.8, 0);


			force += gravity * mass;
		}
		else if (toggle_[PUSELOADEDSCENE] == true && ADDITIONSCENE == 1 && pos_[i].x > 85. || pos_[i].z < -85.){}
		else if (!(toggle_[PUSELOADEDSCENE] == true && ADDITIONSCENE == 2)){
			force += gravity * (mass);					// gravity
		}
		force_[i] = force;
	}

	for (int i = 0; i < num_points_; i++) {
		if (particle_grid_cell_index_[i] == GRID_UNDEF)
			continue;
		if (ptype_[i] != SPH)
			continue;
		vel_eval_[i] += force_[i] * (time_step_ / mass);
	}
	Record(PTIME_FORCE, "", start);

	start.SetSystemTime(ACC_NSEC);
	// cisph correction
	while (((avg_density_divergence_err > 50) || (iterations < 1)) && (iterations < 50)) {
		int num_sph = 0;
		// density
		for (int i = 0; i < num_points_; i++) {
			if (ptype_[i] != SPH)
				continue;
			num_sph++;
			neighbor_has_unwant_[i] = false;
			density_[i] = own_density_contribution;
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
							float W = kernelM4(dist, sr);
							if (ptype_[j] == SPH) {
								Vector3DF NablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
								Vector3DF mnw = NablaW * mass;
								density_[i] += mass * W;
							}
							else if (ptype_[j] == RIGID) {
								neighbor_has_unwant_[i] = true;
								float Psi = density0 / delta_[j];
								density_[i] += Psi * W;
							}
						}
						j = next_particle_index_in_the_same_cell_[j];
					}
				}
			}
		}
		avg_density_divergence_err = 0.;
		for (int i = 0; i < num_points_; i++) {
			if (ptype_[i] != SPH)
				continue;
			Vector3DF ipos = pos_[i];
			Vector3DF ivel = vel_eval_[i];
			float ipressure = pressure_[i];
			float idensity = density_[i];
			float sumNablajC2 = 0.0;
			float sumDivergencei = 0.0;
			Vector3DF nablaCi(0, 0, 0);

			const uint i_cell_index = particle_grid_cell_index_[i];
			if (i_cell_index != GRID_UNDEF)	{
				for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
					const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
					if (neighbor_cell_index == GRID_UNDEF || (neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1))	{
						continue;
					}
					int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
					while (j != GRID_UNDEF) {
						if (i == j || ptype_[j] == FLIP) {		
							j = next_particle_index_in_the_same_cell_[j];
							continue;
						}
						Vector3DF pos_i_minus_j = (ipos - pos_[j])*sim_scale;
						float dist = pos_i_minus_j.Length();

						if (dist <= sr && dist >= 0) {
							const float h_minus_r = sr-dist;
							const float h_minus_r2 = h_minus_r*h_minus_r;
							Vector3DF pi_pjnorm = pos_i_minus_j* (1.0f / dist);
							float m = (ptype_[j]==SPH)?mass:density0/delta_[j]; //mass or Psi

							if (idensity >= density0) {
								Vector3DF nabla_jC = (pi_pjnorm * h_minus_r2 - pos_i_minus_j * (h_minus_r2 / dist - 2 * h_minus_r))*lap_kern_*m;
								const float nabla_jC2 = nabla_jC.Dot(nabla_jC);
								sumNablajC2 += nabla_jC2;
								nablaCi += nabla_jC*(-1);

								const float divergence = h_minus_r2 * dist * lap_kern_ * m;
								sumDivergencei += divergence;
							}
							else {
								Vector3DF nabla_jC = (pi_pjnorm * h_minus_r2*(-1) - pos_i_minus_j * (h_minus_r2 / dist - 2 * h_minus_r))*lap_kern_*m;
								const float nabla_jC2 = nabla_jC.Dot(nabla_jC);
								sumNablajC2 += nabla_jC2;
								nablaCi += nabla_jC*(-1);

								const float divergence = h_minus_r2 * dist * lap_kern_ * m;
								sumDivergencei += divergence;
							}

						}
						j = next_particle_index_in_the_same_cell_[j];
					}
				}
			}
			C = abs(idensity - density0)* 0.01 + sumDivergencei;
			avg_density_divergence_err += C*0.01;
			float nablaCi2 = nablaCi.Dot(nablaCi);
			Vector3DF deltaXi = nablaCi * ((-1)*C / (nablaCi2 + sumNablajC2 + eps));
			pos_[i] += deltaXi * 0.0001f;
		}
		avg_density_divergence_err /= num_sph;
		iterations++;
	}	// while
	//std::cout << iterations << " " << avg_density_divergence_err << std::endl;
	Record(PTIME_PRESS, "", start);

}