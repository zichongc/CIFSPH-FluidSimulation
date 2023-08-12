/***************************************************************
Copyright (c) 2023 Zichong Chen. All Rights Reserved.

FileName	: method_dfsph.cpp
Author		: Zichong Chen, 
Version		: 0.1
***************************************************************/

#include "fluid_system.h"


void ParticleSystem::DFSPH23(float sr) {
	const float mass = param_[PMASS];
	const float sim_scale = param_[PSIMSCALE];

	start.SetSystemTime(ACC_NSEC);
	// find neighbor
	InsertParticlesCPU(num_points());
	Record(PTIME_INSERT, "", start);

	// compute Fadv
	start.SetSystemTime(ACC_NSEC);
	// compute density and factor alpha
	//ComputeDensityAlpha23(sr);
	ComputeDensityDelta23(sr);
	//ComputeForceAdv23(sr);
	TwoWayCoupleForce23(sr);
	Record(PTIME_FORCE, "", start);

	start.SetSystemTime(ACC_NSEC);
	// compute v*
	for (int i = 0; i < num_points_; i++) {
		if (particle_grid_cell_index_[i] == GRID_UNDEF)
			continue;
		if (ptype_[i] != SPH)
			continue;
		vel_eval_[i] += force_[i] * (time_step_ / mass);
	}

	// correctDivergenceError solver
	//DivergenceFreeSolver23(sr);
	CorrectDivergenceErrorFR23(sr);
	// correctDensityError solver
	//ConstantDensitySolver23(sr);
	CorrectDensityErrorFR23(sr);
	Record(PTIME_PRESS, "", start);
}


/**************************************************************************
DFSPH for fluid particles only
**************************************************************************/
void ParticleSystem::ComputeDensityAlpha23(float sr) {
	//const float sr = param_[PSMOOTHRADIUS];
	const float mass = param_[PMASS];
	const float sim_scale = param_[PSIMSCALE];
	const float own_density_contribution = param_[PKERNELSELF] * mass;
	const float stiffness = param_[PGASCONSTANT];
	const float density0 = param_[PRESTDENSITY];

	for (int i = 0; i < num_points(); i++) {
		if (ptype_[i] == 1)
			continue;
		density_[i] = own_density_contribution;
		Vector3DF abssum2(0, 0, 0);
		float sumabs2 = .0;	// for factorAlpha_i
		Vector3DF ipos = pos_[i];
		const uint i_cell_index = particle_grid_cell_index_[i];
		if (i_cell_index != GRID_UNDEF) {
			int cnt = 0;
			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
				if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
					continue;
				int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
				while (j != GRID_UNDEF) {
					if (i == j || ptype_[j] == 1) {				
						j = next_particle_index_in_the_same_cell_[j];
						continue;
					}
					Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;			
					float dist = pos_i_minus_j.Length();
					if (0 <= dist && dist <= sr) {
						float W = kernelM4(dist, sr);
						Vector3DF NablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
						Vector3DF mnw = NablaW * mass;
						density_[i] += mass * W;
						abssum2 += mnw;
						sumabs2 += mnw.Dot(mnw);
						cnt++;
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		pressure_[i] = max(0.0f, stiffness * (density_[i] - density0));
		factor_alpha_[i] = density_[i] / (abssum2.Dot(abssum2) + sumabs2);
	}
}

void ParticleSystem::ComputeForceAdv23(float sr) {
	const float mass = param_[PMASS];
	const float sim_scale = param_[PSIMSCALE];
	const float visc = param_[PVISC];
	const float LaplaceW = lap_kern_;
	Vector3DF gravity = vec_[PPLANE_GRAV_DIR];

	for (int i = 0; i < num_points(); i++) {
		if (ptype_[i] == 1)
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
					if (i == j || ptype_[j] == 1) {					
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
						force += vel_j_minus_i * (h_minus_r / density_[j] / idensity);
						pterm += pos_i_minus_j * ((ipressure + jpressure) / (2 * idensity*jdensity) * kernelPressureGrad(dist, sr));
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		force *= mass * mass * visc * LaplaceW;		// viscosity force
		force -= pterm * (mass*mass);				// pressure force
		force += gravity * mass;					// gravity
		force_[i] = force;
	}
}

void ParticleSystem::ConstantDensitySolver23(float sr) {
	const float density0 = param_[PRESTDENSITY];
	const float mass = param_[PMASS];
	const float sim_scale = param_[PSIMSCALE];
	const float threshold = 500;
	const float timestep = time_step_;
	float density_error_avg = 0.0;
	int iter = 0;

	while ((density_error_avg > threshold || iter < 1) && iter < 10) {
		density_error_avg = 0.0;
		for (int i = fstart; i < fend; i++) {
			if (ptype_[i] == 1)
				continue;
			float deltat_Drho_Dt = 0.0;
			Vector3DF ivel = vel_eval_[i];
			Vector3DF ipos = pos_[i];
			const uint i_cell_index = particle_grid_cell_index_[i];
			if (i_cell_index != GRID_UNDEF) {
				int cnt = 0;
				for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
					const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
					if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
						continue;
					int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
					while (j != GRID_UNDEF) {
						if (i == j || ptype_[j] == 1 || j < fstart || j >= fend) {			
							j = next_particle_index_in_the_same_cell_[j];
							continue;
						}
						Vector3DF vel_i_minus_j = ivel - vel_eval_[j];
						Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
						float dist = pos_i_minus_j.Length();
						if (dist >= 0 && dist <= sr) {
							Vector3DF NablaW = pos_i_minus_j * kernelPressureGradLut(dist, sr);
							deltat_Drho_Dt += vel_i_minus_j.Dot(NablaW);
							cnt++;
						}
						j = next_particle_index_in_the_same_cell_[j];
					}
				}
			}
			deltat_Drho_Dt *= timestep * mass;
			float irho = density_[i] + deltat_Drho_Dt;
			kappa_[i] = (irho - density0) * factor_alpha_[i] / (timestep*timestep);
			density_error_avg += abs(irho - density0) / (fend - fstart);
		}

		for (int i = fstart; i < fend; i++) {
			if (ptype_[i] == 1)
				continue;
			float kappai_div_rhoi = kappa_[i] / density_[i];
			float idensity = density_[i];
			Vector3DF ipos = pos_[i];
			Vector3DF vterm(0, 0, 0);
			const uint i_cell_index = particle_grid_cell_index_[i];
			if (i_cell_index != GRID_UNDEF) {
				for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
					const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
					if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
						continue;

					int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
					while (j != GRID_UNDEF) {
						if (i == j || ptype_[j] == 1 || j < fstart || j >= fend) {					
							j = next_particle_index_in_the_same_cell_[j];
							continue;
						}
						Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
						float dist = pos_i_minus_j.Length();
						if (dist >= 0 && dist <= sr) {
							Vector3DF NablaW = pos_i_minus_j * kernelPressureGradLut(dist, sr);
							float kappaj_div_rhoj = kappa_[j] / density_[j];
							vterm += NablaW * (kappai_div_rhoi + kappaj_div_rhoj);
						}
						j = next_particle_index_in_the_same_cell_[j];
					}
				}
			}
			vterm *= mass*timestep *mass / idensity;
			vel_eval_[i] -= vterm;
		}
		iter++;
	}
}

void ParticleSystem::DivergenceFreeSolver23(float sr) {
	const float mass = param_[PMASS];
	const float sim_scale = param_[PSIMSCALE];
	const float timestep = time_step_;
	const float threshold = 10000;
	float Drho_div_Dt_avg = 0.0;
	int iter = 0;
	while (iter < 10 && (iter < 1 || Drho_div_Dt_avg > threshold)) {
		Drho_div_Dt_avg = 0.0;
		for (int i = fstart; i < fend; i++) {
			if (ptype_[i] == 1)
				continue;
			Vector3DF ivel = vel_eval_[i];
			Vector3DF ipos = pos_[i];
			float term = 0.0;
			const uint i_cell_index = particle_grid_cell_index_[i];
			if (i_cell_index != GRID_UNDEF) {
				for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
					const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
					if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
						continue;

					int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
					while (j != GRID_UNDEF) {
						if (i == j || ptype_[j] == 1 || j < fstart || j >= fend) {				
							j = next_particle_index_in_the_same_cell_[j];
							continue;
						}

						Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
						float dist = pos_i_minus_j.Length();
						if (dist >= 0 && dist <= sr) {
							Vector3DF vel_i_minus_j = ivel - vel_eval_[j];
							Vector3DF NablaW = pos_i_minus_j * kernelPressureGradLut(dist, sr);
							term += vel_i_minus_j.Dot(NablaW);
						}
						j = next_particle_index_in_the_same_cell_[j];
					}
				}
			}
			term = term * mass;
			Drho_div_Dt_[i] = term;
			kappa_[i] = term * factor_alpha_[i] / timestep;
			Drho_div_Dt_avg += abs(term) / (fend - fstart);
		}

		for (int i = fstart; i < fend; i++) {
			if (ptype_[i] == 1)
				continue;
			Vector3DF ipos = pos_[i];
			float kappai_div_rhoi = kappa_[i] / density_[i];
			float idensity = density_[i];
			Vector3DF term(0, 0, 0);
			const uint i_cell_index = particle_grid_cell_index_[i];
			if (i_cell_index != GRID_UNDEF) {
				for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
					const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
					if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
						continue;

					int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
					while (j != GRID_UNDEF) {
						if (i == j || ptype_[j] == 1 || j < fstart || j >= fend) {					
							j = next_particle_index_in_the_same_cell_[j];
							continue;
						}
						Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
						float dist = pos_i_minus_j.Length();
						if (dist >= 0 && dist <= sr) {
							Vector3DF NablaW = pos_i_minus_j * kernelPressureGradLut(dist, sr);
							float kappaj_div_rhoj = kappa_[j] / density_[j];
							term += NablaW * (kappai_div_rhoi + kappaj_div_rhoj);
						}
						j = next_particle_index_in_the_same_cell_[j];
					}
				}
			}
			term *= mass * timestep *mass / idensity;
			vel_eval_[i] -= term;
		}
		iter++;
	}
}



/**************************************************************************
fluid-rigid particles coupling interaction using DFSPH
**************************************************************************/

void ParticleSystem::ComputeDensityDelta23(float sr) {
	const float mass = param_[PMASS];
	const float sim_scale = param_[PSIMSCALE];
	const float own_kernel_contribution = param_[PKERNELSELF];
	const float own_density_contribution = param_[PKERNELSELF] * mass;
	const float stiffness = param_[PGASCONSTANT];
	const float density0 = param_[PRESTDENSITY];

	// compute delta of rigid particles		
	for (int i = 0; i < num_points_; i++) {
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
	// compute density of fluid particles
	for (int i = 0; i < num_points_; i++) {
		if (ptype_[i] != SPH)
			continue;
		neighbor_has_unwant_[i] = false;
		density_[i] = own_density_contribution;
		Vector3DF fluid_abssum2(0, 0, 0);
		Vector3DF rigid_abssum2(0, 0, 0);
		float fluid_sumabs2 = .0;	// for factorAlpha_i
		float rigid_sumabs2 = .0;
		Vector3DF ipos = pos_[i];
		const uint i_cell_index = particle_grid_cell_index_[i];
		if (i_cell_index != GRID_UNDEF) {
			int cnt = 0;
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
							fluid_abssum2 += mnw;
							fluid_sumabs2 += mnw.Dot(mnw);
						}
						else if (ptype_[j] == RIGID) {
							neighbor_has_unwant_[i] = true;
							float Psi = density0 / delta_[j];
							density_[i] += Psi * W;
							Vector3DF NablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
							Vector3DF mnw = NablaW * Psi;
							rigid_abssum2 += mnw;
							rigid_sumabs2 += mnw.Dot(mnw);
						}
						cnt++;
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		pressure_[i] = max(0.0f, stiffness * (density_[i] - density0));
		factor_alpha_[i] = density_[i] / (fluid_abssum2.Dot(fluid_abssum2) + fluid_sumabs2 + rigid_abssum2.Dot(rigid_abssum2) + rigid_sumabs2);
	}
}


void ParticleSystem::TwoWayCoupleForce23(float sr) {
	const float mass = param_[PMASS];
	const float sim_scale = param_[PSIMSCALE];
	const float visc = param_[PVISC];
	const float LaplaceW = lap_kern_;
	Vector3DF gravity = vec_[PPLANE_GRAV_DIR];
	const float density0 = param_[PRESTDENSITY];

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
							force += vel_j_minus_i * (h_minus_r / jdensity / idensity) * mass * visc;
							pterm += pos_i_minus_j * ((ipressure + jpressure) / (2 * idensity*jdensity) * kernelPressureGrad(dist, sr)) * mass;
						}
						else if (ptype_[j] == RIGID) {
							float Psi = density0 / delta_[j];
							pterm += pos_i_minus_j * (ipressure / idensity / idensity * kernelPressureGrad(dist, sr)) * Psi;
							force += vel_j_minus_i * (h_minus_r / idensity / delta_[j] * (visc));	// mu = mu0
						}
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		force *= mass * LaplaceW;		// viscosity force
		force -= pterm * (mass);				// pressure force
		if (toggle_[PUSELOADEDSCENE] == true && (pos_[i].x > 60. || pos_[i].z < -60.)) {
			// no gravity
		}
		else {
			force += gravity * (mass);					// gravity
		}
		force_[i] = force;
	}
}

void ParticleSystem::CorrectDivergenceErrorFR23(float sr) {
	const float mass = param_[PMASS];
	const float sim_scale = param_[PSIMSCALE];
	const float density0 = param_[PRESTDENSITY];
	const float timestep = time_step_;
	const float threshold = 10000;
	float Drho_div_Dt_avg = 0.0;
	int iter = 0;
	while (iter < 10 && (iter < 1 || Drho_div_Dt_avg > threshold)) {
		Drho_div_Dt_avg = 0.0;
		// compute Drho/Dt, kappa
		for (int i = 0; i < num_points_; i++) {
			if (ptype_[i] != SPH)
				continue;
			Vector3DF ivel = vel_eval_[i];
			Vector3DF ipos = pos_[i];
			float term = 0.0;
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
						if (dist >= 0 && dist <= sr) {
							Vector3DF vel_i_minus_j = ivel - vel_eval_[j];
							Vector3DF NablaW = pos_i_minus_j * kernelPressureGradLut(dist, sr);
							if (ptype_[j] == SPH) {
								term += vel_i_minus_j.Dot(NablaW) * mass;
							}
							else if (ptype_[j] = RIGID) {
								float Psi = density0 / delta_[j];
								vel_i_minus_j = ivel;
								term += vel_i_minus_j.Dot(NablaW) * Psi;
							}
						}
						j = next_particle_index_in_the_same_cell_[j];
					}
				}
			}

			Drho_div_Dt_[i] = term;
			kappa_[i] = term * factor_alpha_[i] / timestep;
			Drho_div_Dt_avg += abs(term) / (fend - fstart);
		}

		for (int i = 0; i < num_points_; i++) {
			if (ptype_[i] != SPH)
				continue;
			Vector3DF ipos = pos_[i];
			float kappai_div_rhoi = kappa_[i] / density_[i];
			float idensity = density_[i];
			Vector3DF term(0, 0, 0);
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
						if (dist >= 0 && dist <= sr) {
							Vector3DF NablaW = pos_i_minus_j * kernelPressureGradLut(dist, sr);
							if (ptype_[j] == SPH) {
								float kappaj_div_rhoj = kappa_[j] / density_[j];
								term += NablaW * (kappai_div_rhoi + kappaj_div_rhoj) * mass;
							}
							else if (ptype_[j] == RIGID) {
								float Psi = density0 / delta_[j];
								term += NablaW * (kappai_div_rhoi*Psi);
							}
						}
						j = next_particle_index_in_the_same_cell_[j];
					}
				}
			}
			term *= mass / idensity * timestep;
			vel_eval_[i] -= term;
		}
		iter++;
	}
}

void ParticleSystem::CorrectDensityErrorFR23(float sr) {
	const float mass = param_[PMASS];
	const float sim_scale = param_[PSIMSCALE];
	const float density0 = param_[PRESTDENSITY];
	const float threshold = 500;
	const float timestep = time_step_;
	float density_error_avg = 0.0;
	int iter = 0;

	while ((density_error_avg > threshold || iter < 1) && iter < 10) {
		density_error_avg = 0.0;
		for (int i = 0; i < num_points_; i++) {
			if (ptype_[i] != SPH)
				continue;
			float deltat_Drho_Dt = 0.0;
			Vector3DF ivel = vel_eval_[i];
			Vector3DF ipos = pos_[i];
			const uint i_cell_index = particle_grid_cell_index_[i];
			if (i_cell_index != GRID_UNDEF) {
				int cnt = 0;
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
						Vector3DF vel_i_minus_j = ivel - vel_eval_[j];
						Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;// sim scale
						float dist = pos_i_minus_j.Length();
						if (dist >= 0 && dist <= sr) {
							Vector3DF NablaW = pos_i_minus_j * kernelPressureGradLut(dist, sr);
							if (ptype_[j] == SPH) {
								deltat_Drho_Dt += vel_i_minus_j.Dot(NablaW) * mass;
							}
							else if (ptype_[j] == RIGID) {
								float Psi = density0 / delta_[j];
								vel_i_minus_j = ivel;
								deltat_Drho_Dt += vel_i_minus_j.Dot(NablaW) * Psi;
							}
							cnt++;
						}
						j = next_particle_index_in_the_same_cell_[j];
					}
				}
			}
			deltat_Drho_Dt *= timestep;
			float irho = density_[i] + deltat_Drho_Dt;
			kappa_[i] = (irho - density0) * factor_alpha_[i] / (timestep*timestep);
			density_error_avg += abs(irho - density0) / (fend - fstart);
		}

		for (int i = 0; i < num_points_; i++) {
			if (ptype_[i] != SPH)
				continue;
			float kappai_div_rhoi = kappa_[i] / density_[i];
			float idensity = density_[i];
			Vector3DF ipos = pos_[i];
			Vector3DF vterm(0, 0, 0);
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
						if (dist >= 0 && dist <= sr) {
							Vector3DF NablaW = pos_i_minus_j * kernelPressureGradLut(dist, sr);

							if (ptype_[j] == SPH) {
								float kappaj_div_rhoj = kappa_[j] / density_[j];
								vterm += NablaW * (kappai_div_rhoi + kappaj_div_rhoj) * mass;
							}
							else if (ptype_[j] == RIGID) {
								float Psi = density0 / delta_[j];
								vterm += NablaW*(kappai_div_rhoi * Psi);
							}
						}
						j = next_particle_index_in_the_same_cell_[j];
					}
				}
			}
			vterm *= mass * timestep / idensity;
			vel_eval_[i] -= vterm;
		}
		iter++;
	}
}
