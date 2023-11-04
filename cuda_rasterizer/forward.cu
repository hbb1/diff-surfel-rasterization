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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
#define COND_THRES 10000

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

__device__ bool computeConic3D(const glm::vec3 &p_world, const glm::vec4 &quat, const glm::vec3 &scale, const float *viewmat, const float4 &intrins, float *cur_cov3d, glm::vec3 & normal) {
    // camera information 
    const glm::mat3 W = glm::mat3(
        viewmat[0],viewmat[1],viewmat[2],
        viewmat[4],viewmat[5],viewmat[6],
		viewmat[8],viewmat[9],viewmat[10]
    ); // viewmat 

    const glm::vec3 px = glm::vec3(p_world.x, p_world.y, p_world.z);            // center
    const glm::mat3 T = glm::mat3(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(viewmat[12], viewmat[13], viewmat[14]));
    const glm::mat3 P = glm::mat3(
        intrins.x, 0.0, 0.0, 
        0.0, intrins.y, 0.0,
        intrins.z, intrins.w, 1.0
    );

    glm::mat3 R = quat_to_rotmat(glm::vec4(quat.x, quat.y, quat.z, quat.w));
    glm::mat3 S = scale_to_mat({scale.x, scale.y, 1.0f}, 1.0f);                 // scale
    glm::mat3 M = glm::mat3(R[0], R[1], px);                                    // object local coordinate
    glm::mat3 M_view = P * (W * M + T) * S;                                     // view space
    glm::mat3 M_inv = glm::inverse(M_view);

    // conditiM on number of M_view
    glm::mat3 M_un = (W * M + T);
    glm::mat3 M_un_inv = S * M_inv * P;
    float norm = glm::dot(M_un[0], M_un[0]) + glm::dot(M_un[1], M_un[1]) + glm::dot(M_un[2], M_un[2]);
    float norm_inv = glm::dot(M_un_inv[0], M_un_inv[0]) + glm::dot(M_un_inv[1], M_un_inv[1]) + glm::dot(M_un_inv[2], M_un_inv[2]);
    float cond_num = norm * norm_inv;
    if (cond_num > COND_THRES) return false;

    glm::mat3 Qu = glm::mat3(
        1.0,0.0,0.0,
        0.0,1.0,0.0,
        0.0,0.0,-1.0
    );

    glm::mat3 Qx = glm::transpose(M_inv) * Qu * M_inv;
    cur_cov3d[0] = Qx[0][0]; // A
    cur_cov3d[1] = Qx[0][1]; // B
    cur_cov3d[2] = Qx[1][1]; // C
    cur_cov3d[3] = Qx[0][2]; // D
    cur_cov3d[4] = Qx[1][2]; // E
    cur_cov3d[5] = Qx[2][2]; // F

    normal = {R[2].x, R[2].y, R[2].z};

	// unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
	// if (idx % 32 == 0) {
    //     printf("%d quat %.4f %.4f %.4f %.4f\n", idx, quat.x, quat.y, quat.z,quat.w);
    //     printf("%d scale %.4f %.4f %.4f\n", idx, scale.x, scale.y, scale.z);
	// 	// printf("%d camera center %.4f %.4f %.4f\n", idx, viewmat[12], viewmat[13], viewmat[14]);
    //     printf("%d R[0] %.4f %.4f %.4f\n", idx, R[0].x, R[0].y, R[0].z);
    //     printf("%d R[1] %.4f %.4f %.4f\n", idx, R[1].x, R[1].y, R[1].z);
    //     printf("%d R[2] %.4f %.4f %.4f\n", idx, R[2].x, R[2].y, R[2].z);
	// }
    return true;
}

__device__ bool computeConic2D(const float *cur_cov3d, float3 &conic, float2 &center, float2 &aabb) {
    float A = cur_cov3d[0]; // A
    float B = cur_cov3d[1]; // B
    float C = cur_cov3d[2]; // C
    float D = cur_cov3d[3]; // D
    float E = cur_cov3d[4]; // E
    float F = -cur_cov3d[5]; // F

    const float det = A * C - B * B;
    if (det == 0.0f) 
        return false;
    
    float inv_det = 1.f / det;

    const float cx = (B * E - C * D) * inv_det;
    const float cy = (B * D - A * E) * inv_det;
    
    float isoval = (F - D * cx - E * cy);
    // handle zero isovalue
    if (isoval <= 0.0f) return false;

    const float dx = sqrtf(C * isoval * inv_det); // bounding dx
    const float dy = sqrtf(A * isoval * inv_det); // bounding dy

	float inv_iso = 1.0f / isoval;
	A = A * inv_iso;
	B = B * inv_iso;
	C = C * inv_iso;

    aabb = {dx, dy};
    center = {cx, cy};
    conic = {A, B, C};
    return true;
}

__device__ bool
compute_conic_bounds(const float3& conic, float3 & cov2d, float&radius) {
    float inv_det = conic.x * conic.z - conic.y * conic.y;
    if (inv_det == 0.f)
        return false;
    float det = 1.f / inv_det;
    cov2d.x = conic.z * det;
    cov2d.y = -conic.y * det;
    cov2d.z = conic.x * det;
    float b = 0.5f * (cov2d.x + cov2d.z);
    float v1 = b + sqrt(max(0.1f, b * b - det));
    float v2 = b - sqrt(max(0.1f, b * b - det));
    // take 3 sigma of covariance
    radius = ceil(3.f * sqrt(max(v1, v2)));
    return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	// float* isovals,
	// float3* normals,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	
	float4 intrins = {focal_x, focal_y, float(W)/2.0, float(H)/2.0};
	glm::vec3 p_world = glm::vec3(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
	glm::vec3 scale = scales[idx];
	glm::vec4 quat = rotations[idx];
	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// compute conics 
    float *cur_cov3d = &(cov3Ds[6 * idx]);
	
	bool ok;
    glm::vec3 normal;
	ok = computeConic3D(p_world, quat, scale, viewmatrix, intrins, cur_cov3d, normal);
	if (!ok) return;

	float3 conic;
    float2 center;
    float2 aabb;
    ok = computeConic2D(cur_cov3d, conic, center, aabb);
    if (!ok) return;

	float3 cov2d;
    float radius;
    ok = compute_conic_bounds(conic, cov2d, radius);
	if (!ok) return;

	uint2 rect_min, rect_max;
	getRect(center, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// compute colors 
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// assign values
	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = center;
	conic_opacity[idx] = {conic.x, conic.y, conic.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

	// if (idx % 32 == 0) {
    //     printf("%d center %.4f %.4f depth %.4f\n", idx, center.x, center.y, p_view.z);
    //     printf("%d normal %.4f %.4f %.4f\n", idx, normal.x, normal.y, normal.z);
	// 	printf("%d conic %.4f %.4f %.4f\n", idx, conic.x, conic.y, conic.z);
	// }
        // printf("%d W[0] %.4f %.4f %.4f\n", idx, W[0].x, W[0].y, W[0].z);
        // printf("%d W[1] %.4f %.4f %.4f\n", idx, W[1].x, W[1].y, W[1].z);
        // printf("%d W[2] %.4f %.4f %.4f\n", idx, W[2].x, W[2].y, W[2].z);
		// printf("%d conic %.4f %.4f %.4f\n", idx, conic.x, conic.y, conic.z);
		// printf("%d isoval %.4f \n", idx, isoval);
        // printf("%d centerx centery %.4f %.4f\n", idx, center.x, center.y);
        // printf("%d p_world %.4f %.4f %.4f\n", idx, p_world.x, p_world.y,p_world.z);
        // printf("%d p_view %.4f %.4f %.4f\n", idx, p_view.x, p_view.y, p_view.z);
        // printf("%d scale %.4f %.4f %.4f\n", idx, scale.x, scale.y, scale.z);
        // printf("%d quat %.4f %.4f %.4f\n", idx, quat.x, quat.y, quat.z, quat.w);
        // printf("%d R[0] %.4f %.4f %.4f\n", idx, R[0].x, R[0].y, R[0].z);
        // printf("%d R[1] %.4f %.4f %.4f\n", idx, R[1].x, R[1].y, R[1].z);
        // printf("%d M_inv[0] %.4f %.4f %.4f\n", idx, M_inv[0].x, M_inv[0].y, M_inv[0].z);
        // printf("%d M_inv[1] %.4f %.4f %.4f\n", idx, M_inv[1].x, M_inv[1].y, M_inv[1].z);    
        // printf("%d M_inv[2] %.4f %.4f %.4f\n", idx, M_inv[2].x, M_inv[2].y, M_inv[2].z);
        // printf("%d conic %.4f %.4f %.4f\n", idx, conic.x, conic.y, conic.z);
        // printf("%d Q[0] %.4f %.4f %.4f\n", idx, Qx[0].x, Qx[0].y, Qx[0].z);
        // printf("%d Q[1] %.4f %.4f %.4f\n", idx, Qx[1].x, Qx[1].y, Qx[1].z);
        // printf("%d Q[2] %.4f %.4f %.4f\n", idx, Qx[2].x, Qx[2].y, Qx[2].z);
    // }
	// isovals[idx] = isoval;
	// normals[idx] = normal;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_depth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = { 0 };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			D += depths[collected_id[j]] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = D;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		depths,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	// float* isovals,
	// float3* normals,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		// isovals,
		// normals,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
