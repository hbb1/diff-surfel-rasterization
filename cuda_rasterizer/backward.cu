/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ cov3Ds,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dcov3D,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x + 0.5, (float)pix.y + 0.5};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];

#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	// float dL_dcos_norm;
	// float dL_dcos_depth;
	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];
		// dL_dcos_depth = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		// dL_dcos_norm = dL_depths[DISTNORM_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// float accum_cos_depth_rec = 0;
	// float accum_cos_norm_rec = 0;
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {cov3Ds[9 * coll_id+0], cov3Ds[9 * coll_id+1], cov3Ds[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {cov3Ds[9 * coll_id+3], cov3Ds[9 * coll_id+4], cov3Ds[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {cov3Ds[9 * coll_id+6], cov3Ds[9 * coll_id+7], cov3Ds[9 * coll_id+8]};
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
				// collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// compute ray-splat intersection as before
			// float2 xy = collected_xy[j];
			float3 Tu = collected_Tu[j];
			float3 Tv = collected_Tv[j];
			float3 Tw = collected_Tw[j];
			// compute two planes intersection as the ray intersection
			float3 k = {-Tu.x + pixf.x * Tw.x, -Tu.y + pixf.x * Tw.y, -Tu.z + pixf.x * Tw.z};
			float3 l = {-Tv.x + pixf.y * Tw.x, -Tv.y + pixf.y * Tw.y, -Tv.z + pixf.y * Tw.z};

			if ((k.x * l.y - k.y * l.x) == 0.0f) continue;

			float inv_norm = 1.0f / (k.x * l.y - k.y * l.x);
			float2 s = {(l.z * k.y - k.z * l.y) * inv_norm, -(l.z * k.x - k.z * l.x) * inv_norm};
			float rho3d = (s.x * s.x + s.y * s.y); // splat distance
			
			// add low pass filter according to Botsch et al. [2005].
			// float2 xy = {Tu.z / Tw.z, Tv.z / Tw.z};
			float2 xy = collected_xy[j];
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); // screen distance
			float rho = min(rho3d, rho2d);
			
			// compute accurate depth when necessary
#if INTERSECT_DEPTH
			// float c_d = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
			float c_d = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
#else
			float c_d = Tw.z;
#endif
			float4 con_o = collected_conic_opacity[j];
			float normal[3] = {con_o.x, con_o.y, con_o.z};

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			float dL_dz = 0.0f;
#if RENDER_AXUTILITY

			// D = D - T * alpha * c_d;
			// D2 = D2 - T * alpha * c_d * c_d;
			float dL_dweight = (final_D2 + c_d * c_d * final_A - 2 * c_d * final_D) * dL_dreg;
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			dL_dz += 2.0f * (T * alpha) * (c_d * final_A - final_D) * dL_dreg;

			// float w = T * alpha;
			// float A = (1-T);
			// float A_last = (1-T_last);
			// float A_final = (1-T_final);

			// float A_error = (A + A_last - A_final) * dL_dreg;
			// float D_error = (D + D_last - D_final) * dL_dreg;
			// float dL_dweight = A_error * c_d - D_error;
			// dL_dalpha += dL_dweight - last_dL_dT;
			// last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;

			// compute ray-splat cosines
			// glm::vec3 dir = glm::vec3((pixf.x - 0.5*float(W)) / focal_x, (pixf.y-0.5*float(H)) / focal_y, 1);
			// dir = dir / glm::length(dir);
			// float cos = glm::dot(glm::vec3(normal[0], normal[1], normal[2]), -dir);
			// float clamp = 1.0f;

			// // Propagate gradients w.r.t ray-splat distortions
			// accum_cos_depth_rec = last_alpha * last_cos * last_depth + (1.f - last_alpha) * accum_cos_depth_rec;
			// accum_cos_norm_rec = last_alpha * last_cos + (1.f - last_alpha) * accum_cos_norm_rec;
			// last_cos = cos;
			// dL_dalpha += (cos * c_d - accum_cos_depth_rec) * dL_dcos_depth;
			// dL_dalpha += (cos - accum_cos_norm_rec) * dL_dcos_norm;

			// float dL_dcos = c_d * alpha * T * dL_dcos_depth;
			// dL_dcos += alpha * T * dL_dcos_norm;
			// dL_dcos *= clamp;

			// dL_dz += cos * alpha * T * dL_dcos_depth;
			// atomicAdd((&dL_dnormal3D[global_id * 3 + 0]), -dir.x * dL_dcos);
			// atomicAdd((&dL_dnormal3D[global_id * 3 + 1]), -dir.y * dL_dcos);
			// atomicAdd((&dL_dnormal3D[global_id * 3 + 2]), -dir.z * dL_dcos);

			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			// Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal[ch];
				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			}
#endif

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			dL_dz += alpha * T * dL_ddepth; 

#if DEBUG
			if (collected_id[j] > 0 && pix.x == W / 4 && pix.y == H / 2) {
				printf("%d backward %d depth %.8f\n", contributor, collected_id[j], c_d);
				// printf("%d backward %d D %.8f\n", contributor, collected_id[j], D);
				printf("%d backward %d T %.8f\n", contributor, collected_id[j], T);
				printf("%d backward %d dL_dalpha %.8f\n", contributor, collected_id[j], dL_dalpha);
				printf("%d backward %d dL_dz %.8f\n", contributor, collected_id[j], dL_dz);
				printf("%d backward %d max_contrib %d\n", contributor, collected_id[j], max_contributor);
				printf("-----------\n");
			}
#endif

			if (rho3d <= rho2d) {
				// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
#if INTERSECT_DEPTH
				float2 dL_ds = {
					dL_dG * -G * s.x + dL_dz * Tw.x,
					dL_dG * -G * s.y + dL_dz * Tw.y
				};
				float3 dz_dTw = {s.x, s.y, 1.0};
#else
				float2 dL_ds = {
					dL_dG * -G * s.x,
					dL_dG * -G * s.y
				};
				float3 dz_dTw = {0.0, 0.0, 1.0};
#endif
				float dsx_dnorm = s.x * (-inv_norm);
				float dsy_dnorm = s.y * (-inv_norm);
				float3 dnorm_dk = {l.y, -l.x, 0.0};
				float3 dnorm_dl = {-k.y, k.x, 0.0};

				// can be optimized but factor out inv_norm
				float3 dsx_dk = {0.0 * inv_norm + dsx_dnorm * dnorm_dk.x, l.z * inv_norm + dsx_dnorm * dnorm_dk.y, -l.y * inv_norm + dsx_dnorm * dnorm_dk.z};
				float3 dsy_dk = {-l.z * inv_norm + dsy_dnorm * dnorm_dk.x, 0.0 * inv_norm + dsy_dnorm * dnorm_dk.y, l.x * inv_norm + dsy_dnorm * dnorm_dk.z};
				float3 dsx_dl = {0.0 * inv_norm + dsx_dnorm * dnorm_dl.x, -k.z * inv_norm + dsx_dnorm * dnorm_dl.y, k.y * inv_norm + dsx_dnorm * dnorm_dl.z};
				float3 dsy_dl = {k.z * inv_norm + dsy_dnorm * dnorm_dl.x, 0.0 * inv_norm + dsy_dnorm * dnorm_dl.y, -k.x * inv_norm + dsy_dnorm * dnorm_dl.z};

				float3 dL_dk = {
					dL_ds.x * dsx_dk.x + dL_ds.y * dsy_dk.x, 
					dL_ds.x * dsx_dk.y + dL_ds.y * dsy_dk.y, 
					dL_ds.x * dsx_dk.z + dL_ds.y * dsy_dk.z, 
				};

				float3 dL_dl = {
					dL_ds.x * dsx_dl.x + dL_ds.y * dsy_dl.x, 
					dL_ds.x * dsx_dl.y + dL_ds.y * dsy_dl.y, 
					dL_ds.x * dsx_dl.z + dL_ds.y * dsy_dl.z, 
				};

				float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
				float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
				float3 dL_dTw = {
					pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
					pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
					pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};


				// Update gradients w.r.t. 3D covariance (3x3 matrix)
				atomicAdd(&dL_dcov3D[global_id * 9 + 0],  dL_dTu.x);
				atomicAdd(&dL_dcov3D[global_id * 9 + 1],  dL_dTu.y);
				atomicAdd(&dL_dcov3D[global_id * 9 + 2],  dL_dTu.z);
				atomicAdd(&dL_dcov3D[global_id * 9 + 3],  dL_dTv.x);
				atomicAdd(&dL_dcov3D[global_id * 9 + 4],  dL_dTv.y);
				atomicAdd(&dL_dcov3D[global_id * 9 + 5],  dL_dTv.z);
				atomicAdd(&dL_dcov3D[global_id * 9 + 6],  dL_dTw.x);
				atomicAdd(&dL_dcov3D[global_id * 9 + 7],  dL_dTw.y);
				atomicAdd(&dL_dcov3D[global_id * 9 + 8],  dL_dTw.z);

				// __syncthreads();
				// for testing the correctness of gradients
				// if (global_id == 0 && pix.x == 0 && pix.y == H / 2) {
				// 	printf("%d pix (%.4f %.4f)\n", global_id, pixf.x, pixf.y);
				// 	printf("%d G %.4f\n", global_id, G);
				// 	printf("%d depth %.8f\n", global_id, c_d);
				// 	// printf("%d rho3d %.8f\n", global_id, rho3d);
				// 	printf("%d dL_dG %.8f\n", global_id, dL_dG);
				// 	printf("%d dL_dddepth %.8f\n", global_id, dL_ddepth);
				// 	printf("%d Tu %.8f %.8f %.8f\n", global_id, Tu.x, Tu.y, Tu.z);
				// 	printf("%d Tv %.8f %.8f %.8f\n", global_id, Tv.x, Tv.y, Tv.z);
				// 	printf("%d Tw %.8f %.8f %.8f\n", global_id, Tw.x, Tw.y, Tw.z);
				// 	printf("%d dL_dTu %.8f %.8f %.8f\n", global_id, dL_dTu.x, dL_dTu.y, dL_dTu.z);
				// 	printf("%d dL_dTv %.8f %.8f %.8f\n", global_id, dL_dTv.x, dL_dTv.y, dL_dTv.z);
				// 	printf("%d dL_dTw %.8f %.8f %.8f\n", global_id, dL_dTw.x, dL_dTw.y, dL_dTw.z);
				// }

			} else {
				// // Update gradients w.r.t. center of Gaussian 2D mean position
				float dG_ddelx = -G * FilterInvSquare * d.x;
				float dG_ddely = -G * FilterInvSquare * d.y;
				// atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled
				// atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled
				atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled
				atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled
				atomicAdd(&dL_dcov3D[global_id * 9 + 8],  dL_dz); // propagate depth loss
				// float inv_Twz = 1 / Tw.z;
				// float dL_dTuz = dL_dG * dG_ddelx * inv_Twz;
				// float dL_dTvz = dL_dG * dG_ddely * inv_Twz;
				// float dL_dTwz = dL_dG * dG_ddelx * (-Tu.z * inv_Twz * inv_Twz) + dL_dG * dG_ddely * (-Tv.z * inv_Twz * inv_Twz) + dL_ddepth; // add depth loss here
				// atomicAdd(&dL_dcov3D[global_id * 9 + 2],  dL_dTuz); // propagate loss
				// atomicAdd(&dL_dcov3D[global_id * 9 + 5],  dL_dTvz); // propagate loss
				// atomicAdd(&dL_dcov3D[global_id * 9 + 8],  dL_dTwz); // propagate loss
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

inline __device__ void computeCov3D(
	const glm::vec3 & p_world,
	const glm::vec4 & quat,
	const glm::vec3 & scale,
	const float* viewmat,
	const float4 & intrins,
	float tan_fovx, 
	float tan_fovy,
	const float* cov3D,
	const float* dL_dcov3D,
	const float* dL_dnormal3D,
	glm::vec3 & dL_dmean3D,
	glm::vec3 & dL_dscale,
	glm::vec4 & dL_drot
) {
	// camera information 
	const glm::mat3 W = glm::mat3(
		viewmat[0],viewmat[1],viewmat[2],
		viewmat[4],viewmat[5],viewmat[6],
		viewmat[8],viewmat[9],viewmat[10]
	); // viewmat 

	const glm::vec3 cam_pos = glm::vec3(viewmat[12], viewmat[13], viewmat[14]); // camera center
	const glm::mat4 P = glm::mat4(
		intrins.x, 0.0, 0.0, 0.0,
		0.0, intrins.y, 0.0, 0.0,
		intrins.z, intrins.w, 1.0, 1.0,
		0.0, 0.0, 0.0, 0.0
	);

	glm::mat3 S = scale_to_mat({scale.x, scale.y, 1.0f}, 1.0f);
	glm::mat3 R = quat_to_rotmat(quat);
	glm::mat3 RS = R * S;
	glm::vec3 p_view = W * p_world + cam_pos;
#if CLIP
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float pxpz = p_view.x / p_view.z;
	const float pypz = p_view.y / p_view.z;
	p_view.x = min(limx, max(-limx, pxpz)) * p_view.z;
	p_view.y = min(limy, max(-limy, pypz)) * p_view.z;
#endif

	glm::mat3 M = glm::mat3(W * RS[0], W * RS[1], p_view);


	glm::mat4x3 dL_dT = glm::mat4x3(
		dL_dcov3D[0], dL_dcov3D[1], dL_dcov3D[2],
		dL_dcov3D[3], dL_dcov3D[4], dL_dcov3D[5],
		dL_dcov3D[6], dL_dcov3D[7], dL_dcov3D[8],
		0.0, 0.0, 0.0
	);

	glm::mat3x4 dL_dM_aug = glm::transpose(P) * glm::transpose(dL_dT);
	glm::mat3 dL_dM = glm::mat3(
		glm::vec3(dL_dM_aug[0]),
		glm::vec3(dL_dM_aug[1]),
		glm::vec3(dL_dM_aug[2])
	);

	glm::vec3 dL_dRS0 = glm::transpose(W) * dL_dM[0];
	glm::vec3 dL_dRS1 = glm::transpose(W) * dL_dM[1];
	glm::vec3 dL_dpw = glm::transpose(W) * dL_dM[2];
	glm::vec3 dL_dtn = glm::transpose(W) * glm::vec3(dL_dnormal3D[0], dL_dnormal3D[1], dL_dnormal3D[2]);

#if DUAL_VISIABLE
	glm::vec3 tn = W*R[2];
	float cos = glm::dot(-tn, M[2]);
	float multiplier = cos > 0 ? 1 : -1;
	dL_dtn *= multiplier;
#endif

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS0 * glm::vec3(scale.x),
		dL_dRS1 * glm::vec3(scale.y),
		dL_dtn
	);

	dL_drot = quat_to_rotmat_vjp(quat, dL_dR);
	dL_dscale = glm::vec3(
		(float)glm::dot(dL_dRS0, R[0]),
		(float)glm::dot(dL_dRS1, R[1]),
		0.0f
	);

#if CLIP
	const float x_grad_mul = pxpz < -limx || pxpz > limx ? 0 : 1;
	const float y_grad_mul = pypz < -limy || pypz > limy ? 0 : 1;
	dL_dpw.x *= x_grad_mul;
	dL_dpw.y *= y_grad_mul;
#endif
	dL_dmean3D = dL_dpw;
	// unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
	// if (idx == 0) {
	// 		glm::mat4x3 T = glm::transpose(P * glm::mat3x4(
	// 		glm::vec4(M[0], 0.0),
	// 		glm::vec4(M[1], 0.0),
	// 		glm::vec4(M[2], 1.0)
	// 	));
	// 	printf("T %d [%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]\n", idx, T[0].x, T[0].y, T[0].z, T[1].x, T[1].y, T[1].z, T[2].x, T[2].y, T[2].z, T[3].x, T[3].y, T[3].z);
	//     printf("dL_dT %d [%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]\n", idx, dL_dT[0].x, dL_dT[0].y, dL_dT[0].z, dL_dT[1].x, dL_dT[1].y, dL_dT[1].z, dL_dT[2].x, dL_dT[2].y, dL_dT[2].z, dL_dT[3].x, dL_dT[3].y, dL_dT[3].z);
	//     printf("dL_dM %d [%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]\n", idx, dL_dM[0].x, dL_dM[0].y, dL_dM[0].z, dL_dM[1].x, dL_dM[1].y, dL_dM[1].z, dL_dM[2].x, dL_dM[2].y, dL_dM[2].z);
	// 	printf("dL_dM_aug %d [%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]\n", idx, dL_dM_aug[0].x, dL_dM_aug[0].y, dL_dM_aug[0].z, dL_dM_aug[0].w, dL_dM_aug[1].x, dL_dM_aug[1].y, dL_dM_aug[1].z, dL_dM_aug[1].w, dL_dM_aug[2].x, dL_dM_aug[2].y, dL_dM_aug[2].z, dL_dM_aug[2].w);
	//     printf("dL_dR %d [%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]\n", idx, dL_dR[0].x, dL_dR[0].y, dL_dR[0].z, dL_dR[1].x, dL_dR[1].y, dL_dR[1].z, dL_dR[2].x, dL_dR[2].y, dL_dR[2].z);
	// 	printf("dL_dscale %d [%.8f, %.8f, %.8f]\n", idx, dL_dscale.x, dL_dscale.y, dL_dscale.z);
	// 	printf("dL_drot %d [%.8f, %.8f, %.8f, %.8f]\n", idx, dL_drot.x, dL_drot.y, dL_drot.z, dL_drot.w);
	// 	printf("dL_dmean3d %d [%.8f, %.8f, %.8f]\n", idx, dL_dmean3D.x, dL_dmean3D.y, dL_dmean3D.z);
	// }
}



template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* cov3Ds,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 
	// grad input
	const float* dL_dcov3Ds,
	const float* dL_dnormal3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	// grad output
	glm::vec3* dL_dmean3Ds,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const float* cov3D = &(cov3Ds[9 * idx]);
	const float* dL_dcov3D = &(dL_dcov3Ds[9 * idx]);
	const float* dL_dnormal3D = &(dL_dnormal3Ds[3 * idx]);

	glm::vec3 p_world = glm::vec3(means3D[idx].x, means3D[idx].y, means3D[idx].z);
	float4 intrins = {focal_x, focal_y, focal_x * tan_fovx, focal_y * tan_fovy};

	glm::vec3 dL_dmean3D;
	glm::vec3 dL_dscale;
	glm::vec4 dL_drot;
	computeCov3D(
		p_world,
		rotations[idx],
		scales[idx],
		viewmatrix,
		intrins,
		tan_fovx,
		tan_fovy,
		cov3D, 
		dL_dcov3D,
		dL_dnormal3D,
		dL_dmean3D, 
		dL_dscale,
		dL_drot
	);
	// update 
	dL_dmean3Ds[idx] = dL_dmean3D;
	dL_dscales[idx] = dL_dscale;
	dL_drots[idx] = dL_drot;

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dshs);
}

__global__ void computeCenter(int P, 
	const int * radii,
	const float * cov3Ds,
	const float3 * dL_dmean2Ds,
	float *dL_dcov3Ds) {
	
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;
	
	const float* cov3D = cov3Ds + 9 * idx;

	const float3 dL_dmean2D = dL_dmean2Ds[idx];
	glm::mat4x3 T = glm::mat4x3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[3], cov3D[4], cov3D[5],
		cov3D[6], cov3D[7], cov3D[8],
		cov3D[6], cov3D[7], cov3D[8]
	);

	float d = glm::dot(glm::vec3(1.0, 1.0, -1.0), T[3] * T[3]);
	glm::vec3 f = glm::vec3(1.0, 1.0, -1.0) * (1.0f / d);

	glm::vec3 p = glm::vec3(
		glm::dot(f, T[0] * T[3]),
		glm::dot(f, T[1] * T[3]), 
		glm::dot(f, T[2] * T[3]));

	glm::vec3 dL_dT0 = dL_dmean2D.x * f * T[3];
	glm::vec3 dL_dT1 = dL_dmean2D.y * f * T[3];
	glm::vec3 dL_dT3 = dL_dmean2D.x * f * T[0] + dL_dmean2D.y * f * T[1];
	glm::vec3 dL_df = (dL_dmean2D.x * T[0] * T[3]) + (dL_dmean2D.y * T[1] * T[3]);
	float dL_dd = glm::dot(dL_df, f) * (-1.0 / d);
	glm::vec3 dd_dT3 = glm::vec3(1.0, 1.0, -1.0) * T[3] * 2.0f;
	dL_dT3 += dL_dd * dd_dT3;
	dL_dcov3Ds[9 * idx + 0] += dL_dT0.x;
	dL_dcov3Ds[9 * idx + 1] += dL_dT0.y;
	dL_dcov3Ds[9 * idx + 2] += dL_dT0.z;
	dL_dcov3Ds[9 * idx + 3] += dL_dT1.x;
	dL_dcov3Ds[9 * idx + 4] += dL_dT1.y;
	dL_dcov3Ds[9 * idx + 5] += dL_dT1.z;
	dL_dcov3Ds[9 * idx + 6] += dL_dT3.x;
	dL_dcov3Ds[9 * idx + 7] += dL_dT3.y;
	dL_dcov3Ds[9 * idx + 8] += dL_dT3.z;

	// if (idx == 0) {
	// 	printf("dL_dmean2d %d [%.8f, %.8f]\n", idx, dL_dmean2D.x, dL_dmean2D.y);
	// 	printf("T %d [%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]\n", idx, T[0].x, T[0].y, T[0].z, T[1].x, T[1].y, T[1].z, T[2].x, T[2].y, T[2].z, T[3].x, T[3].y, T[3].z);
	//     printf("dL_dT %d [%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]\n", idx, dL_dT0.x, dL_dT0.y, dL_dT0.z, dL_dT1.x, dL_dT1.y, dL_dT1.z, dL_dT3.x, dL_dT3.y, dL_dT3.z);
	//     // printf("dL_dM %d [%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]\n", idx, dL_dM[0].x, dL_dM[0].y, dL_dM[0].z, dL_dM[1].x, dL_dM[1].y, dL_dM[1].z, dL_dM[2].x, dL_dM[2].y, dL_dM[2].z);
	// 	// printf("dL_dM_aug %d [%.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f]\n", idx, dL_dM_aug[0].x, dL_dM_aug[0].y, dL_dM_aug[0].z, dL_dM_aug[0].w, dL_dM_aug[1].x, dL_dM_aug[1].y, dL_dM_aug[1].z, dL_dM_aug[1].w, dL_dM_aug[2].x, dL_dM_aug[2].y, dL_dM_aug[2].z, dL_dM_aug[2].w);
	// }
}


void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	const glm::vec3* campos, 
	const float3* dL_dmean2Ds,
	const float* dL_dnormal3Ds,
	float* dL_dcov3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	// propagate gradients to cov3D

	// we do not use the center actually
	computeCenter << <(P + 255) / 256, 256 >> >(
		P, 
		radii,
		cov3Ds,
		dL_dmean2Ds,
		dL_dcov3Ds);
	
	// propagate gradients from cov3d to mean3d, scale, rot, sh, color
	preprocessCUDA<NUM_CHANNELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		cov3Ds,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		focal_x, 
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,	
		dL_dcov3Ds,
		dL_dnormal3Ds,
		dL_dcolors,
		dL_dshs,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots
	);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* cov3Ds,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_depths,
	float * dL_dcov3D,
	float3* dL_dmean2D,
	float* dL_dnormal3D,
	float* dL_dopacity,
	float* dL_dcolors)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		bg_color,
		means2D,
		conic_opacity,
		cov3Ds,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_depths,
		dL_dcov3D,
		dL_dmean2D,
		dL_dnormal3D,
		dL_dopacity,
		dL_dcolors
		);
}
