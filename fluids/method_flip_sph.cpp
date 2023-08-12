/***************************************************************
Copyright (c) 2023 Zichong Chen. All Rights Reserved.

FileName	: method_flip_sph.cpp
Author		: Zichong Chen, 
Version		: 0.1
***************************************************************/

#include "fluid_system.h"

/************************************************************
FLIP-SPH: coupling FLIP and SPH together,
with the boundary particles setting
************************************************************/

void ParticleSystem::FLIP_SPH23(int mode) {
	const float sim_scale = param_[PSIMSCALE];
	const float sr = param_[PSMOOTHRADIUS];
	// compute vis*
	start.SetSystemTime(ACC_NSEC);
	InsertParticlesCPU(num_points());
	ComputeVelStarSPH23(sr);
	Record(PTIME_OTHER_FORCE, "", start);
	// compute vis(t+t')
	switch (mode)
	{
	case METHOD_IISPH:
		IISPH23(sr);
		break;
	case METHOD_DFSPH:
		DFSPH23(sr);
		break;
	case METHOD_CISPH:
		CISPH23(sr);
		break;
	default:
		break;
	}	// Vs(t+t') in vel_eval_	

	// compute vif
	start.SetSystemTime(ACC_NSEC);
	ComputeVelFLIP23(sr);

	Record(PTIME_PCI_STEP, "", start);

}

void ParticleSystem::ComputeVelStarSPH23(float sr) {
	const float sim_scale = param_[PSIMSCALE];
	//const float sr = param_[PSMOOTHRADIUS];	

	for (int i = 0; i < num_points_; i++) {
		if (ptype_[i] != SPH)
			continue;
		Vector3DF ipos = pos_[i];
		Vector3DF sumVel(0, 0, 0);
		float sumW = 0.0;
		int neighbor = 0;

		const uint i_cell_index = particle_grid_cell_index_[i];
		if (i_cell_index != GRID_UNDEF) {
			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
				if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
					continue;
				int j = grid_head_cell_particle_index_array_[cell];
				while (j != GRID_UNDEF) {
					if (i == j || ptype_[j] != FLIP) {
						j = next_particle_index_in_the_same_cell_[j];
						continue;
					}
					Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
					float dist = pos_i_minus_j.Length();
					if (dist >= 0 && dist <= sr) {
						float W = kernelM4(dist, sr);
						sumW += W;
						sumVel += vel_eval_[j] * W;
						neighbor++;
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		if (neighbor == 0)
			vel_[i] = vel_eval_[i];
		else
			vel_[i] = sumVel * (1 / sumW);
		//cout << "vel " << vel_[i].x << " " << vel_[i].y << " " << vel_[i].z << endl;
	}
}

void ParticleSystem::ComputeVelFLIP23(float sr) {
	const float sim_scale = param_[PSIMSCALE];
	//const float viscosity = param_[PVISC];
	const float viscosity = 0.01;
	const float mass = param_[PMASS];
	//const float h = param_[PSMOOTHRADIUS];
	const float h = param_[PSPACINGREALWORLD];
	float alpha = min(1, 6 * time_step_*viscosity / h / h);
	//cout << "alpha " << alpha << " " << 6 * time_step_*viscosity / h / h << endl;

	for (int i = 0; i < num_points_; i++) {
		if (ptype_[i] != FLIP)
			continue;
		neighbor_has_unwant_[i] = false;
		float sumW_ifjs = 0.0;
		Vector3DF sumVel_if_SPH(0, 0, 0);
		Vector3DF sumVel_if_FLIP(0, 0, 0);
		Vector3DF ivel = vel_eval_[i];
		Vector3DF ipos = pos_[i];
		int neighbor = 0;

		const uint i_cell_index = particle_grid_cell_index_[i];
		if (i_cell_index != GRID_UNDEF) {
			for (int cell = 0; cell < max_num_adj_grid_cells_cpu; cell++) {
				const int neighbor_cell_index = i_cell_index + grid_neighbor_cell_index_offset_[cell];
				if (neighbor_cell_index == GRID_UNDEF || neighbor_cell_index < 0 || neighbor_cell_index > grid_total_ - 1)
					continue;

				int j = grid_head_cell_particle_index_array_[neighbor_cell_index];
				while (j != GRID_UNDEF) {
					if (i == j || ptype_[j] != SPH) {
						j = next_particle_index_in_the_same_cell_[j];
						continue;
					}

					Vector3DF pos_i_minus_j = (ipos - pos_[j]) * sim_scale;
					float dist = pos_i_minus_j.Length();
					if (0 <= dist && dist <= sr) {
						if (neighbor_has_unwant_[j])
							neighbor_has_unwant_[i] = true;

						float W = kernelM4(dist, sr);
						Vector3DF jvel = vel_eval_[j];
						sumW_ifjs += W;
						sumVel_if_SPH += jvel * W;
						sumVel_if_FLIP += (jvel - vel_[j]) * W;
						neighbor++;
					}
					j = next_particle_index_in_the_same_cell_[j];
				}
			}
		}
		if (neighbor != 0) {
			vel_[i] = sumVel_if_SPH * (1 / sumW_ifjs);
			vel_eval_[i] += sumVel_if_FLIP * (1 / sumW_ifjs);
		}
		else {
			Vector3DF gravity = vec_[PPLANE_GRAV_DIR];		// global force
			if (toggle_[PUSELOADEDSCENE] == true && ADDITIONSCENE == 2) {
				gravity.Set(-8, -2, 0);
			}
			vel_[i] = gravity * time_step_;
			vel_eval_[i] = gravity * time_step_;
		}
		//cout << "vel " << vel_eval_[i].x << " " << vel_eval_[i].y << " " << vel_eval_[i].z << endl;
		//cout << "vel " << sumVelisF.x << " " << sumVelisF.y << " " << sumVelisF.z << endl;
	}

	for (int i = 0; i < num_points_; i++) {
		if (particle_grid_cell_index_[i] == GRID_UNDEF)
			continue;
		if (ptype_[i] != FLIP)
			continue;
		if (neighbor_has_unwant_[i]) {
			vel_eval_[i] = vel_[i];
		}
		else {
			vel_eval_[i] = vel_eval_[i] * (1 - alpha);
			vel_eval_[i] += vel_[i] * alpha;
		}
		//cout << "vel " << vel_eval_[i].x << " " << vel_eval_[i].y << " " << vel_eval_[i].z << endl;
		//cout << "vel " << vel_[i].x << " " << vel_[i].y << " " << vel_[i].z << endl;
	}

}

/*****************************************************************

pos_ ---- graphics_scale
---- * sim_scale ->  simulation scale

param_[PSMOOTHRADIUS/...] ---- simulation scale

vel_eval_ ---- simulation scale


FLIP particle:
vel_ ---- V_iF^SPH
vel_eval_ ---- V_iF^FLIP
SPH particle:
vel_ ---- V_iS^*
vel_eval_ ---- V_iS

ptype_:
0 ---- SPH particle
1 ---- FLIP particle
2 ---- RIGID particle

********************************************************************/