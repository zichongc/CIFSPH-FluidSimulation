/***************************************************************
Copyright (c) 2023 Zichong Chen. All Rights Reserved.

FileName	: method_iisph.cpp
Author		: Zichong Chen, 
Version		: 0.1
***************************************************************/

#include "fluid_system.h"

/****************************************************
IISPH: without coupling boundary particles
****************************************************/
void ParticleSystem::IISPH23(float sr) {
	start.SetSystemTime(ACC_NSEC);
	InsertParticlesCPU(num_points());
	Record(PTIME_INSERT, "", start);

	start.SetSystemTime(ACC_NSEC);
	PredictAdvection23(sr);
	Record(PTIME_PRESS, "", start);

	start.SetSystemTime(ACC_NSEC);
	PressureSlover23(sr);
	
	Integration23(sr);
	Record(PTIME_FORCE, "", start);
}

void ParticleSystem::PredictAdvection23(float sr) {
	const float mass = param_[PMASS];
	const float sim_scale = param_[PSIMSCALE];
	const float viscosity = param_[PVISC];
	const float own_density = param_[PKERNELSELF] * mass;
	const float own_kernel_contribution = param_[PKERNELSELF];
	const float kappa = param_[PGASCONSTANT];
	const float density0 = param_[PRESTDENSITY];
	const float timestep = time_step_;
	const float laplaceW = lap_kern_;
	Vector3DF gravity = vec_[PPLANE_GRAV_DIR];
	Vector3DF* dii = labetalv;	// rename, reuse
	float* aii = beiyong1;		// rename, reuse
	float* rho_adv = beiyong2;	// rename, reuse
	float* pressure_loop = predicted_density_;	// rename, reuse

	// compute delta of rigid particle
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

	float avg_density = 0.;
	int num = 0;
	// compute density of sph particles
	for (int i = 0; i < num_points_; i++) {
		if (ptype_[i] != SPH)
			continue;	// sph particle
		Vector3DF ipos = pos_[i];
		float idensity = own_density;
		const uint i_cell_index = particle_grid_cell_index_[i];
		int cnt = 0;
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
							idensity += mass * W;
						}
						else if (ptype_[j] == RIGID) {
							float Psi = density0 / delta_[j];
							idensity += Psi * W;
						}
					}
					cnt++;
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		density_[i] = idensity;
		avg_density += idensity;
		num++;
		//std::cout << idensity << " rho\n";
		if (frame_ == 0) {
			//pressure_[i] = max(0.0, kappa * (idensity - density0));
			pressure_[i] = 0.0;
			pressure_loop[i] = 0.5*pressure_[i];
		}
		//cout << "neighbor" << cnt << " " << ptype_[i] << " " << density_[i] << "  press " << pressure_[i] << "  pl " << pressure_loop[i] << endl;
	}
	std::cout << avg_density / (float)num << " avg_density" << std::endl;

	// compute force^adv, v^adv and dii
	for (int i = 0; i < num_points_; i++) {
		if (ptype_[i] != SPH)
			continue;	// sph particle
		Vector3DF ipos = pos_[i];
		Vector3DF ivel = vel_eval_[i];
		Vector3DF iforce(0, 0, 0);
		Vector3DF diii(0, 0, 0);
		float idensity = density_[i];

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
					Vector3DF vel_j_minus_i = vel_eval_[j] - ivel;
					float dist = pos_i_minus_j.Length();
					if (0 <= dist && dist <= sr) {
						float h_minus_r = sr - dist;
						Vector3DF nablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
						if (ptype_[j] == SPH) {
							iforce += vel_j_minus_i * (h_minus_r / idensity / density_[j] * mass * viscosity);
							diii -= nablaW*mass;
						}
						else if (ptype_[j] == RIGID) {
							float Psi = density0 / delta_[j];
							diii -= nablaW*Psi;
							iforce += vel_j_minus_i * (h_minus_r / idensity / delta_[j] * viscosity);
						}
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		iforce *= mass*laplaceW;
		if (toggle_[PUSELOADEDSCENE] == true && (pos_[i].x > 60. || pos_[i].z < -60.)) {
			// no gravity
		}
		else {
			iforce += gravity * (mass);					// gravity
		}
		dii[i] = diii * (timestep*timestep / idensity / idensity);
		force_[i] = iforce;
		vel_eval_[i] += iforce * (timestep / mass);
	}

	// compute predict density and aii
	for (int i = 0; i < num_points_; i++) {
		if (ptype_[i] != SPH)
			continue;	// sph particle
		Vector3DF ipos = pos_[i];
		Vector3DF ivel = vel_eval_[i];
		Vector3DF diii = dii[i];
		float idensity = density_[i];
		float rho_predict = 0.0;
		float aiii = 0.0;
		rho_adv[i] = density_[i];

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
						float jdensity = density_[j];
						Vector3DF vel_i_minus_j = ivel - vel_eval_[j];
						Vector3DF nablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
						Vector3DF dji = nablaW * (timestep*timestep*mass / idensity / idensity);
						if (ptype_[j] == SPH) {
							rho_predict += vel_i_minus_j.Dot(nablaW) * mass;
							aiii += (diii - dji).Dot(nablaW) * mass;
						}
						else if (ptype_[j] == RIGID) {
							float Psi = density0 / delta_[j];
							aiii += (diii - dji).Dot(nablaW) * Psi;
							//rho_predict += (ivel - vel_eval_[j]).Dot(nablaW) * Psi;
							rho_predict += (ivel).Dot(nablaW)*Psi;
						}
						//Vector3DF dji = nablaW * (-timestep*timestep*mass / idensity / idensity);		
						
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		rho_adv[i] += rho_predict * timestep;
		aii[i] = (aiii == 0) ? 1 : aiii;
		if (frame_ > 0)
			pressure_loop[i] = 0.5*pressure_[i];
	}
}

void ParticleSystem::PressureSlover23(float sr) {
	const float mass = param_[PMASS];
	const float density0 = param_[PRESTDENSITY];
	const float sim_scale = param_[PSIMSCALE];
	const float timestep = time_step_;
	const float eta = 1. * density0;
	const float omega = 0.5;
	Vector3DF* sumdijpj = detapos_;				// rename, reuse
	Vector3DF* dii = labetalv;					// rename, reuse
	float* aii = beiyong1;						// rename, reuse
	float* rho_adv = beiyong2;					// rename, reuse
	float* pressure_loop = predicted_density_;	// rename, reuse
	int iteration = 0;
	float density_error = 0.0;
	float last_error = density_error + 1.0;

	//while((iteration < 1 || density_error > eta) && (iteration < 5)) {
	while ((iteration < 1 || density_error > eta) && iteration < 5 && density_error != last_error) {
		last_error = density_error;
		density_error = 0.0;
		for (int i = 0; i < num_points_; i++) {
			if (ptype_[i] != SPH)
				continue;
			Vector3DF ipos = pos_[i];
			Vector3DF temp(0, 0, 0);

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
							Vector3DF nablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
							if (ptype_[j] == SPH) {
								float jdensity = density_[j];
								temp -= nablaW * (pressure_loop[j] / jdensity / jdensity);
							}
							else if (ptype_[j] == RIGID) {
								//float Psi = density0 / delta_[j];
								//temp -=
							}
						}
						j = next_particle_index_in_the_same_cell_[j];
					}
				}
			}
			sumdijpj[i] = temp * (mass*timestep*timestep);
		}

		for (int i = 0; i < num_points_; i++) {
			if (ptype_[i] != SPH)
				continue;
			Vector3DF ipos = pos_[i];
			Vector3DF term_sumdijpj = sumdijpj[i];
			const float bi = density0 - rho_adv[i];
			float idensity = density_[i];
			float ipressure = pressure_loop[i];
			float aijpj = 0.0;
			float aikpk = 0.0;

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
							Vector3DF nablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);

							if (ptype_[j] == SPH) {
								Vector3DF djjpj = dii[j] * pressure_loop[j];
								Vector3DF dji = nablaW * (timestep*timestep*mass / idensity / idensity);
								Vector3DF temp_sumdjkpk = sumdijpj[j] - dji * ipressure;
								aijpj += (term_sumdijpj - djjpj - temp_sumdjkpk).Dot(nablaW);
							}
							else if (ptype_[j] == RIGID) {
								aikpk += term_sumdijpj.Dot(nablaW) * density0 / delta_[j];
							}
						}
						j = next_particle_index_in_the_same_cell_[j];
					}

				}
			}
			aijpj *= mass;
			//cout << rho_adv[i] << endl;
			density_error += abs((rho_adv[i] + pressure_[i] * aii[i] + aijpj) - density0) / num_points();
			pressure_[i] = max(0.0, (1 - omega) * pressure_loop[i] + omega / aii[i] * (bi - aijpj - aikpk));
			//pressure_[i] = max(0.0, (1 - omega) * pressure_loop[i] + omega / aii[i] * (bi - aijpj));
		}
		for (int i = 0; i < num_points(); i++) {
			if (particle_grid_cell_index_[i] == GRID_UNDEF || ptype_[i] != SPH)
				continue;
			pressure_loop[i] = pressure_[i];
		}
		iteration++;

		cout << iteration << " " << density_error << endl;
	}
}

void ParticleSystem::Integration23(float sr) {
	const float sim_scale = param_[PSIMSCALE];
	const float mass = param_[PMASS];
	const float timestep = time_step_;
	const float density0 = param_[PRESTDENSITY];

	for (int i = 0; i < num_points_; i++) {
		if (ptype_[i] != SPH) continue;
		Vector3DF ipos = pos_[i];
		Vector3DF pforce(0, 0, 0);
		float ipressure = pressure_[i];
		float idensity = density_[i];
		float iterm = ipressure / idensity / idensity;
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
						Vector3DF nablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
						float jdensity = density_[j];
						if (ptype_[j] == SPH) {
							//pforce -= nablaW * (iterm + pressure_[j] / jdensity / jdensity * mass);
							pforce -= nablaW * ((iterm + pressure_[j]/ jdensity/jdensity)*mass);
						}
						else if (ptype_[j] == RIGID) {
							float Psi = density0 / delta_[j];
							pforce -= nablaW * (iterm* Psi);
						}
					}
					j = next_particle_index_in_the_same_cell_[j];
				}

			}
		}
		pforce *= mass;
		vel_eval_[i] += pforce * (timestep / mass);
		//pos_[i] += vel_eval_[i] * (timestep / sim_scale);
	}

}


//
//void ParticleSystem::PredictAdvection23(float sr) {
//	const float mass = param_[PMASS];
//	const float sim_scale = param_[PSIMSCALE];
//	const float viscosity = param_[PVISC];
//	const float own_density = param_[PKERNELSELF] * mass;
//	const float kappa = param_[PGASCONSTANT];
//	const float density0 = param_[PRESTDENSITY];
//	const float timestep = time_step_;
//	const float laplaceW = lap_kern_;
//	Vector3DF gravity = vec_[PPLANE_GRAV_DIR];
//	Vector3DF* dii = labetalv;	// rename, reuse
//	float* aii = beiyong1;		// rename, reuse
//	float* rho_adv = beiyong2;	// rename, reuse
//	float* pressure_loop = predicted_density_;	// rename, reuse
//
//	for (int i = 0; i < num_points(); i++) {
//		if (ptype_[i]) continue;	// sph particle
//		Vector3DF ipos = pos_[i];
//		float idensity = own_density;
//		const uint i_cell_index = particle_grid_cell_index_[i];
//		int cnt = 0;
//		if (i_cell_index != GRID_UNDEF) {
//			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
//				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
//				if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
//					continue;
//				int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
//
//				while (j != GRID_UNDEF) {
//					if (i == j || ptype_[j]) {
//						j = next_particle_index_in_the_same_cell_[j];
//						continue;
//					}
//					Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
//					float dist = pos_i_minus_j.Length();
//					if (0 <= dist && dist <= sr) {
//						float W = kernelM4(dist, sr);
//						idensity += mass * W;
//					}
//					cnt++;
//					j = next_particle_index_in_the_same_cell_[j];
//				}
//			}
//		}
//		density_[i] = idensity;
//
//		if (frame_ == 0) {
//			//pressure_[i] = max(0.0, kappa * (idensity - density0));
//			pressure_[i] = 0.0;
//			pressure_loop[i] = 0.5*pressure_[i];
//		}
//		//cout << "neighbor" << cnt << " " << ptype_[i] << " " << density_[i] << "  press " << pressure_[i] << "  pl " << pressure_loop[i] << endl;
//	}
//
//	for (int i = 0; i < num_points(); i++) {
//		if (ptype_[i]) continue;	// sph particle
//		Vector3DF ipos = pos_[i];
//		Vector3DF ivel = vel_eval_[i];
//		Vector3DF iforce(0, 0, 0);
//		Vector3DF diii(0, 0, 0);
//		float idensity = density_[i];
//
//		const uint i_cell_index = particle_grid_cell_index_[i];
//		if (i_cell_index != GRID_UNDEF) {
//			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
//				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
//				if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
//					continue;
//				int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
//				while (j != GRID_UNDEF) {
//					if (i == j || ptype_[j]) {
//						j = next_particle_index_in_the_same_cell_[j];
//						continue;
//					}
//					Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
//					Vector3DF vel_j_minus_i = vel_eval_[j] - ivel;
//					float dist = pos_i_minus_j.Length();
//					if (0 <= dist && dist <= sr) {
//						float h_minus_r = sr - dist;
//						Vector3DF nablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
//						iforce += vel_j_minus_i * (h_minus_r / idensity / density_[j]);
//						diii -= nablaW;
//					}
//					j = next_particle_index_in_the_same_cell_[j];
//				}
//			}
//		}
//		iforce *= mass*mass*viscosity*laplaceW;
//		iforce += gravity * mass;				// Fadv
//		dii[i] = diii * (mass*timestep*timestep / idensity / idensity);
//		force_[i] = iforce;
//		vel_eval_[i] += iforce * (timestep / mass);
//	}
//
//	for (int i = 0; i < num_points(); i++) {
//		if (ptype_[i]) continue;	// sph particle
//		Vector3DF ipos = pos_[i];
//		Vector3DF ivel = vel_eval_[i];
//		Vector3DF diii = dii[i];
//		float idensity = density_[i];
//		float rho_predict = 0.0;
//		float aiii = 0.0;
//		rho_adv[i] = density_[i];
//
//		const uint i_cell_index = particle_grid_cell_index_[i];
//		if (i_cell_index != GRID_UNDEF) {
//			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
//				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
//				if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
//					continue;
//				int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
//				while (j != GRID_UNDEF) {
//					if (i == j || ptype_[j]) {
//						j = next_particle_index_in_the_same_cell_[j];
//						continue;
//					}
//					Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
//					float dist = pos_i_minus_j.Length();
//					if (0 <= dist && dist <= sr) {
//						float jdensity = density_[j];
//						Vector3DF vel_i_minus_j = ivel - vel_eval_[j];
//						Vector3DF nablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
//
//						//Vector3DF dji = nablaW * (-timestep*timestep*mass / idensity / idensity);		
//						Vector3DF dji = nablaW * (timestep*timestep*mass / idensity / idensity);
//
//						rho_predict += vel_i_minus_j.Dot(nablaW);
//						aiii += (diii - dji).Dot(nablaW);
//					}
//					j = next_particle_index_in_the_same_cell_[j];
//				}
//			}
//		}
//		rho_adv[i] += rho_predict * mass * timestep;
//		aii[i] = (aiii == 0) ? 1 : aiii * mass;
//		if (frame_ > 0)
//			pressure_loop[i] = 0.5*pressure_[i];
//	}
//}
//
//void ParticleSystem::PressureSlover23(float sr) {
//	const float mass = param_[PMASS];
//	const float density0 = param_[PRESTDENSITY];
//	const float sim_scale = param_[PSIMSCALE];
//	const float timestep = time_step_;
//	const float eta = 0.01 * density0;
//	const float omega = 0.5;
//	Vector3DF* sumdijpj = detapos_;				// rename, reuse
//	Vector3DF* dii = labetalv;					// rename, reuse
//	float* aii = beiyong1;						// rename, reuse
//	float* rho_adv = beiyong2;					// rename, reuse
//	float* pressure_loop = predicted_density_;	// rename, reuse
//	int iteration = 0;
//	float density_error = 0.0;
//	float last_error = density_error + 1.0;
//
//	while ((iteration < 2 || density_error > eta) && iteration < 5 && density_error != last_error) {
//		last_error = density_error;
//		density_error = 0.0;
//		for (int i = 0; i < num_points(); i++) {
//			if (ptype_[i] == 1) continue;
//			Vector3DF ipos = pos_[i];
//			Vector3DF temp(0, 0, 0);
//
//			const uint i_cell_index = particle_grid_cell_index_[i];
//			if (i_cell_index != GRID_UNDEF) {
//				for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
//					const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
//					if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
//						continue;
//					int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
//					while (j != GRID_UNDEF) {
//						if (i == j || ptype_[j] == 1) {
//							j = next_particle_index_in_the_same_cell_[j];
//							continue;
//						}
//						Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
//						float dist = pos_i_minus_j.Length();
//						if (0 <= dist && dist <= sr) {
//							Vector3DF nablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
//							float jdensity = density_[j];
//							temp -= nablaW * (pressure_loop[j] / jdensity / jdensity);
//						}
//						j = next_particle_index_in_the_same_cell_[j];
//					}
//				}
//			}
//			sumdijpj[i] = temp * (mass*timestep*timestep);
//		}
//
//		for (int i = 0; i < num_points(); i++) {
//			if (ptype_[i] == 1) continue;
//			Vector3DF ipos = pos_[i];
//			Vector3DF term_sumdijpj = sumdijpj[i];
//			const float bi = density0 - rho_adv[i];
//			float idensity = density_[i];
//			float ipressure = pressure_loop[i];
//			float aijpj = 0.0;
//
//			const uint i_cell_index = particle_grid_cell_index_[i];
//			if (i_cell_index != GRID_UNDEF) {
//				for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
//					const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
//					if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
//						continue;
//					int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
//					while (j != GRID_UNDEF) {
//						if (i == j || ptype_[j] == 1) {
//							j = next_particle_index_in_the_same_cell_[j];
//							continue;
//						}
//						Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
//						float dist = pos_i_minus_j.Length();
//						if (0 <= dist && dist <= sr) {
//							Vector3DF nablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
//							Vector3DF djjpj = dii[j] * pressure_loop[j];
//
//							//Vector3DF dji = nablaW * (-timestep*timestep*mass / idensity / idensity);
//							Vector3DF dji = nablaW * (timestep*timestep*mass / idensity / idensity);
//
//							Vector3DF temp_sumdjkpk = sumdijpj[j] - dji * ipressure;
//							aijpj += (term_sumdijpj - djjpj - temp_sumdjkpk).Dot(nablaW);
//						}
//						j = next_particle_index_in_the_same_cell_[j];
//					}
//
//				}
//			}
//			aijpj *= mass;
//			//cout << rho_adv[i] << endl;
//			density_error += abs((rho_adv[i] + pressure_[i] * aii[i] + aijpj) - density0) / num_points();
//			pressure_[i] = max(0.0, (1 - omega) * pressure_loop[i] + omega / aii[i] * (bi - aijpj));
//
//		}
//		for (int i = 0; i < num_points(); i++) {
//			if (particle_grid_cell_index_[i] == GRID_UNDEF || ptype_[i] == 1)
//				continue;
//			pressure_loop[i] = pressure_[i];
//		}
//		iteration++;
//
//		//cout << iteration << " " << density_error << endl;
//	}
//}
//
//void ParticleSystem::Integration23(float sr) {
//	const float sim_scale = param_[PSIMSCALE];
//	const float mass = param_[PMASS];
//	const float timestep = time_step_;
//
//	for (int i = 0; i < num_points(); i++) {
//		if (ptype_[i] == 1) continue;
//		Vector3DF ipos = pos_[i];
//		Vector3DF pforce(0, 0, 0);
//		float ipressure = pressure_[i];
//		float idensity = density_[i];
//		float iterm = ipressure / idensity / idensity;
//		const uint i_cell_index = particle_grid_cell_index_[i];
//		if (i_cell_index != GRID_UNDEF) {
//			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
//				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
//				if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
//					continue;
//				int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
//				while (j != GRID_UNDEF) {
//					if (i == j || ptype_[j] == 1) {
//						j = next_particle_index_in_the_same_cell_[j];
//						continue;
//					}
//					Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
//					float dist = pos_i_minus_j.Length();
//					if (0 <= dist && dist <= sr) {
//						Vector3DF nablaW = pos_i_minus_j * kernelPressureGrad(dist, sr);
//						float jdensity = density_[j];
//						pforce -= nablaW * (iterm + pressure_[j] / jdensity / jdensity);
//					}
//					j = next_particle_index_in_the_same_cell_[j];
//				}
//
//			}
//		}
//		pforce *= (mass*mass);
//		vel_eval_[i] += pforce * (timestep / mass);
//		pos_[i] += vel_eval_[i] * (timestep / sim_scale);
//	}
//
//}
//
//void ParticleSystem::IISPH23(float sr) {
//	//const float sr = param_[PSMOOTHRADIUS];
//
//	start.SetSystemTime(ACC_NSEC);
//	InsertParticlesCPU(num_points());
//	Record(PTIME_INSERT, "", start);
//
//	start.SetSystemTime(ACC_NSEC);
//	PredictAdvection23(sr);
//	Record(PTIME_PRESS, "", start);
//
//	start.SetSystemTime(ACC_NSEC);
//	PressureSlover23(sr);
//	Record(PTIME_FORCE, "", start);
//
//	start.SetSystemTime(ACC_NSEC);
//	Integration23(sr);
//	Record(PTIME_PCI_STEP, "", start);
//}