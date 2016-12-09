/* Copyright 2016 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "snap.h"
#include "sweep.h"

#include <stdlib.h>
#include <x86intrin.h>

extern LegionRuntime::Logger::Category log_snap;

using namespace LegionRuntime::Accessor;

//------------------------------------------------------------------------------
MiniKBATask::MiniKBATask(const Snap &snap, const Predicate &pred, bool even,
                         const SnapArray &flux, const SnapArray &fluxm,
                         const SnapArray &qtot, const SnapArray &vdelt, 
                         const SnapArray &dinv, const SnapArray &t_xs,
                         const SnapArray &time_flux_in, 
                         const SnapArray &time_flux_out,
                         const SnapArray &qim,
                         int group, int corner, const int ghost_offsets[3])
  : SnapTask<MiniKBATask, Snap::MINI_KBA_TASK_ID>(
      snap, Rect<3>(Point<3>::ZEROES(), Point<3>::ZEROES()), pred), 
    mini_kba_args(MiniKBAArgs(corner, group))
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&mini_kba_args, sizeof(mini_kba_args));
  Snap::SnapFieldID group_field = SNAP_ENERGY_GROUP_FIELD(group);
  // If you add projection requirements here, remember to update
  // the value of NON_GHOST_REQUIREMENTS in sweep.h
  qtot.add_projection_requirement(READ_ONLY, *this, 
                                  group_field, Snap::SWEEP_PROJECTION);
  // We need reduction privileges on the flux field since all sweeps
  // will be contributing to it
  flux.add_projection_requirement(Snap::SUM_REDUCTION_ID, *this, 
                                  group_field, Snap::SWEEP_PROJECTION);
  fluxm.add_projection_requirement(Snap::QUAD_REDUCTION_ID, *this,
                                   group_field, Snap::SWEEP_PROJECTION);
  // Add the dinv array for this field
  dinv.add_projection_requirement(READ_ONLY, *this,
                                  group_field, Snap::SWEEP_PROJECTION);
  time_flux_in.add_projection_requirement(READ_ONLY, *this,
                                          group_field, Snap::SWEEP_PROJECTION);
  time_flux_out.add_projection_requirement(WRITE_DISCARD, *this,
                                           group_field, Snap::SWEEP_PROJECTION);
  t_xs.add_projection_requirement(READ_ONLY, *this,
                                  group_field, Snap::SWEEP_PROJECTION);
  // Then add our writing ghost regions
  for (int i = 0; i < Snap::num_dims; i++)
  {
    Snap::SnapFieldID ghost_write = even ? 
      SNAP_GHOST_FLUX_FIELD_EVEN(group, corner, i) :
      SNAP_GHOST_FLUX_FIELD_ODD(group, corner, i);
    flux.add_projection_requirement(WRITE_DISCARD, *this, 
                                    ghost_write, Snap::SWEEP_PROJECTION);
  }
  qim.add_projection_requirement(
      (Snap::source_layout == Snap::MMS_SOURCE) ? READ_ONLY : NO_ACCESS, *this,
                                  group_field, Snap::SWEEP_PROJECTION);
  assert(region_requirements.size() == MINI_KBA_NON_GHOST_REQUIREMENTS);
  // Add our reading ghost regions
  for (int i = 0; i < Snap::num_dims; i++)
  {
    // Reverse polarity for these ghost fields
    Snap::SnapFieldID ghost_read = even ?
      SNAP_GHOST_FLUX_FIELD_ODD(group, corner, i) :
      SNAP_GHOST_FLUX_FIELD_EVEN(group, corner, i);
    // We know our projection ID now
    Snap::SnapProjectionID proj_id = SNAP_GHOST_PROJECTION(i, ghost_offsets[i]);
    flux.add_projection_requirement(READ_ONLY, *this, ghost_read, proj_id);
  }
  // This one last since it's not a projection requirement
  vdelt.add_region_requirement(READ_ONLY, *this, group_field);
}

//------------------------------------------------------------------------------
void MiniKBATask::dispatch_wavefront(int wavefront, const Domain &launch_d,
                                     Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  // Save our wavefront
  this->mini_kba_args.wavefront = wavefront;
  // Set our launch domain
  this->launch_domain = launch_d;
  // Then call the normal dispatch routine
  dispatch(ctx, runtime);
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  register_cpu_variant<cpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::preregister_gpu_variants(void)
//------------------------------------------------------------------------------
{
  register_gpu_variant<gpu_implementation>(true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.print("Running Mini-KBA Sweep");

  assert(task->arglen == sizeof(MiniKBAArgs));
  const MiniKBAArgs *args = reinterpret_cast<const MiniKBAArgs*>(task->args);
    
  // This implementation of the sweep assumes three dimensions
  assert(Snap::num_dims == 3);

  RegionAccessor<AccessorType::Generic,MomentQuad> fa_qtot = 
    regions[0].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<MomentQuad>();
  RegionAccessor<AccessorType::Generic,double> fa_flux = 
    regions[1].get_accessor().typeify<double>();
  RegionAccessor<AccessorType::Generic,MomentQuad> fa_fluxm = 
    regions[2].get_accessor().typeify<MomentQuad>();

  // Reduction accessors don't support structured points yet
  ByteOffset flux_offsets[3], fluxm_offsets[3];
  double *const flux = fa_flux.raw_rect_ptr<3>(flux_offsets);
  MomentQuad *const fluxm = fa_fluxm.raw_rect_ptr<3>(fluxm_offsets);

  // No types here since the size of these fields are dependent
  // on the number of angles
  const double vdelt = regions[regions.size()-1].get_field_accessor(
      SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<double>().read(
      DomainPoint::from_point<1>(Point<1>::ZEROES()));
  RegionAccessor<AccessorType::Generic> fa_dinv = 
    regions[3].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic> fa_time_flux_in = 
    regions[4].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic> fa_time_flux_out = 
    regions[5].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic,double> fa_t_xs = 
    regions[6].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<double>();

  // Output ghost regions
  RegionAccessor<AccessorType::Generic> fa_ghostx_out = 
    regions[7].get_field_accessor(
        *(task->regions[7].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_ghosty_out = 
    regions[8].get_field_accessor(
        *(task->regions[8].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_ghostz_out = 
    regions[9].get_field_accessor(
        *(task->regions[9].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_qim = 
    regions[10].get_field_accessor(
        *(task->regions[10].privilege_fields.begin()));
  // Input ghost regions
  RegionAccessor<AccessorType::Generic> fa_ghostx_in = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS].get_field_accessor(
        *(task->regions[MINI_KBA_NON_GHOST_REQUIREMENTS].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_ghosty_in = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS+1].get_field_accessor(
        *(task->regions[MINI_KBA_NON_GHOST_REQUIREMENTS+1].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_ghostz_in = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS+2].get_field_accessor(
        *(task->regions[MINI_KBA_NON_GHOST_REQUIREMENTS+2].privilege_fields.begin()));

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  // Figure out the origin point based on which corner we are
  const bool stride_x_positive = ((args->corner & 0x1) != 0);
  const bool stride_y_positive = ((args->corner & 0x2) != 0);
  const bool stride_z_positive = ((args->corner & 0x4) != 0);
  const coord_t origin_ints[3] = { 
    (stride_x_positive ? subgrid_bounds.lo[0] : subgrid_bounds.hi[0]),
    (stride_y_positive ? subgrid_bounds.lo[1] : subgrid_bounds.hi[1]),
    (stride_z_positive ? subgrid_bounds.lo[2] : subgrid_bounds.hi[2]) };
  const Point<3> origin(origin_ints);

  // Local arrays
  const size_t angle_buffer_size = Snap::num_angles * sizeof(double);
  double *psi = (double*)malloc(angle_buffer_size);
  double *pc = (double*)malloc(angle_buffer_size);
  double *psii = (double*)malloc(angle_buffer_size);
  double *psij = (double*)malloc(angle_buffer_size);
  double *psik = (double*)malloc(angle_buffer_size);
  double *time_flux_in = (double*)malloc(angle_buffer_size);
  double *time_flux_out = (double*)malloc(angle_buffer_size);
  double *temp_array = (double*)malloc(angle_buffer_size);
  double *hv_x = (double*)malloc(angle_buffer_size);
  double *hv_y = (double*)malloc(angle_buffer_size);
  double *hv_z = (double*)malloc(angle_buffer_size);
  double *hv_t = (double*)malloc(angle_buffer_size);
  double *fx_hv_x = (double*)malloc(angle_buffer_size);
  double *fx_hv_y = (double*)malloc(angle_buffer_size);
  double *fx_hv_z = (double*)malloc(angle_buffer_size);
  double *fx_hv_t = (double*)malloc(angle_buffer_size);

  const double tolr = 1.0e-12;

  // There's no parallelism here, better to walk pencils in one direction
  // and try and maintain some locality. Hopefully the 2-D slices will 
  // remain in the last level cache
  // Assume y_range 16 and z_range 16
  // 2K angles * 8 bytes * 16 * 16 = 4MB
  // This is maybe not as good from a blocking standpoint (maybe better
  // to block for L2, something like 2x2x2 blocks), but it will result 
  // in linear strides through memory which will be better for the 
  // prefetchers and likely result in overall better performance
  // because the very small 2x2x2 size will be too small to warm up
  // the prefetchers and they will be confused by the access pattern
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  double *yflux_pencil = (double*)malloc(x_range * angle_buffer_size);
  double *zflux_plane  = (double*)malloc(y_range * x_range * angle_buffer_size);

  for (int z = 0; z < z_range; z++) {
    for (int y = 0; y < y_range; y++) {
      for (int x = 0; x < x_range; x++) {
        // Figure out the local point that we are working on    
        Point<3> local_point = origin;
        if (stride_x_positive)
          local_point.x[0] += x;
        else
          local_point.x[0] -= x;
        if (stride_y_positive)
          local_point.x[1] += y;
        else
          local_point.x[1] -= y;
        if (stride_z_positive)
          local_point.x[2] += z;
        else
          local_point.x[2] -= z;

        // Compute the angular source
        const DomainPoint dp = DomainPoint::from_point<3>(local_point);
        const MomentQuad quad = fa_qtot.read(dp);
        for (int ang = 0; ang < Snap::num_angles; ang++)
          psi[ang] = quad[0];
        if (Snap::num_moments > 1) {
          const int corner_offset = 
            args->corner * Snap::num_angles * Snap::num_moments;
          for (unsigned l = 1; 1 < Snap::num_moments; l++) {
            const int moment_offset = corner_offset + l * Snap::num_angles;
            for (int ang = 0; ang < Snap::num_angles; ang++) {
              psi[ang] += Snap::ec[moment_offset+ang] * quad[l];
            }
          }
        }

        // If we're doing MMS, there is an additional term
        if (Snap::source_layout == Snap::MMS_SOURCE)
        {
          fa_qim.read_untyped(DomainPoint::from_point<3>(local_point),
                              temp_array, angle_buffer_size);
          for (int ang = 0; ang < Snap::num_angles; ang++)
            psi[ang] += temp_array[ang];
        }

        // Compute the initial solution
        for (int ang = 0; ang < Snap::num_angles; ang++)
          pc[ang] = psi[ang];
        // X ghost cells
        if (stride_x_positive) {
          // reading from x-1 
          Point<3> ghost_point = local_point;
          ghost_point.x[0] -= 1;
          if (x == 0) {
            // Ghost cell array
            fa_ghostx_in.read_untyped(DomainPoint::from_point<3>(ghost_point),   
                                      psii, angle_buffer_size); 
          } 
          // Else nothing: psii already contains next flux
        } else {
          // reading from x+1
          Point<3> ghost_point = local_point;
          ghost_point.x[0] += 1;
          // Local coordinates here
          if (x == 0) {
            // Ghost cell array
            fa_ghostx_in.read_untyped(DomainPoint::from_point<3>(ghost_point), 
                                      psii, angle_buffer_size);
          }
          // Else nothing: psii already contains next flux
        }
        for (int ang = 0; ang < Snap::num_angles; ang++)
          pc[ang] += psii[ang] * Snap::mu[ang] * Snap::hi;
        // Y ghost cells
        if (stride_y_positive) {
          // reading from y-1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] -= 1;
          if (y == 0) {
            // Ghost cell array
            fa_ghosty_in.read_untyped(DomainPoint::from_point<3>(ghost_point), 
                                      psij, angle_buffer_size);
          } else {
            // Local array
            const int offset = x * Snap::num_angles;
            memcpy(psij, yflux_pencil+offset, angle_buffer_size);
          }
        } else {
          // reading from y+1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] += 1;
          // Local coordinates here
          if (y == 0) {
            // Ghost cell array
            fa_ghosty_in.read_untyped(DomainPoint::from_point<3>(ghost_point), 
                                      psij, angle_buffer_size);
          } else {
            // Local array
            const int offset = x * Snap::num_angles;
            memcpy(psij, yflux_pencil+offset, angle_buffer_size);
          }
        }
        for (int ang = 0; ang < Snap::num_angles; ang++)
          pc[ang] += psij[ang] * Snap::eta[ang] * Snap::hj;
        // Z ghost cells
        if (stride_z_positive) {
          // reading from z-1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] -= 1;
          if (z == 0) {
            // Ghost cell array
            fa_ghostz_in.read_untyped(DomainPoint::from_point<3>(ghost_point), 
                                      psik, angle_buffer_size);
          } else {
            // Local array
            const int offset = (y * x_range + x) * Snap::num_angles;
            memcpy(psik, zflux_plane+offset, angle_buffer_size);
          }
        } else {
          // reading from z+1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] += 1;
          // Local coordinates here
          if (z == 0) {
            // Ghost cell array
            fa_ghostz_in.read_untyped(DomainPoint::from_point<3>(ghost_point), 
                                      psik, angle_buffer_size);
          } else {
            // Local array
            const int offset = (y * x_range + x) * Snap::num_angles;
            memcpy(psik, zflux_plane+offset, angle_buffer_size);
          }
        }
        for (int ang = 0; ang < Snap::num_angles; ang++)
          pc[ang] += psik[ang] * Snap::xi[ang] * Snap::hk;

        // See if we're doing anything time dependent
        if (vdelt != 0.0) 
        {
          fa_time_flux_in.read_untyped(DomainPoint::from_point<3>(local_point),
                                       time_flux_in, angle_buffer_size);
          for (int ang = 0; ang < Snap::num_angles; ang++)
            pc[ang] += vdelt * time_flux_in[ang];
        }
        // Multiple by the precomputed denominator inverse
        fa_dinv.read_untyped(DomainPoint::from_point<3>(local_point),
                             temp_array, angle_buffer_size);
        for (int ang = 0; ang < Snap::num_angles; ang++)
          pc[ang] *= temp_array[ang];

        if (Snap::flux_fixup) {
          // DO THE FIXUP
          unsigned old_negative_fluxes = 0;
          for (int ang = 0; ang < Snap::num_angles; ang++)
            hv_x[ang] = 1.0;
          for (int ang = 0; ang < Snap::num_angles; ang++)
            hv_y[ang] = 1.0;
          for (int ang = 0; ang < Snap::num_angles; ang++)
            hv_z[ang] = 1.0;
          for (int ang = 0; ang < Snap::num_angles; ang++)
            hv_t[ang] = 1.0;
          const double t_xs = fa_t_xs.read(DomainPoint::from_point<3>(local_point));
          while (true) {
            unsigned negative_fluxes = 0;
            // Figure out how many negative fluxes we have
            for (int ang = 0; ang < Snap::num_angles; ang++) {
              fx_hv_x[ang] = 2.0 * pc[ang] - psii[ang];
              if (fx_hv_x[ang] < 0.0) {
                hv_x[ang] = 0.0;
                negative_fluxes++;
              }
            }
            for (int ang = 0; ang < Snap::num_angles; ang++) {
              fx_hv_y[ang] = 2.0 * pc[ang] - psij[ang];
              if (fx_hv_y[ang] < 0.0) {
                hv_y[ang] = 0.0;
                negative_fluxes++;
              }
            }
            for (int ang = 0; ang < Snap::num_angles; ang++) {
              fx_hv_z[ang] = 2.0 * pc[ang] - psik[ang];
              if (fx_hv_z[ang] < 0.0) {
                hv_z[ang] = 0.0;
                negative_fluxes++;
              }
            }
            if (vdelt != 0.0) {
              for (int ang = 0; ang < Snap::num_angles; ang++) {
                fx_hv_t[ang] = 2.0 * pc[ang] - time_flux_in[ang];
                if (fx_hv_t[ang] < 0.0) {
                  hv_t[ang] = 0.0;
                  negative_fluxes++;
                }
              }
            }
            if (negative_fluxes == old_negative_fluxes)
              break;
            old_negative_fluxes = negative_fluxes; 
            if (vdelt != 0.0) {
              for (int ang = 0; ang < Snap::num_angles; ang++) {
                pc[ang] = psi[ang] + 0.5 * (
                    psii[ang] * Snap::mu[ang] * Snap::hi * (1.0 + hv_x[ang]) + 
                    psij[ang] * Snap::eta[ang] * Snap::hj * (1.0 + hv_y[ang]) + 
                    psik[ang] * Snap::xi[ang] * Snap::hk * (1.0 + hv_z[ang]) + 
                    time_flux_in[ang] * vdelt * (1.0 + hv_t[ang]) );
                double den = (pc[ang] <= 0.0) ? 0.0 : (t_xs + 
                  Snap::mu[ang] * Snap::hi * hv_x[ang] + 
                  Snap::eta[ang] * Snap::hj * hv_y[ang] +
                  Snap::xi[ang] * Snap::hk * hv_z[ang] + vdelt * hv_t[ang]);
                if (den < tolr)
                  pc[ang] = 0.0;
                else
                  pc[ang] /= den;
              }
            } else {
              for (int ang = 0; ang < Snap::num_angles; ang++) {
                pc[ang] = psi[ang] + 0.5 * (
                    psii[ang] * Snap::mu[ang] * Snap::hi * (1.0 + hv_x[ang]) + 
                    psij[ang] * Snap::eta[ang] * Snap::hj * (1.0 + hv_y[ang]) +
                    psik[ang] * Snap::xi[ang] * Snap::hk * (1.0 + hv_z[ang]) );
                double den = (pc[ang] <= 0.0) ? 0.0 : (t_xs + 
                  Snap::mu[ang] * Snap::hi * hv_x[ang] + 
                  Snap::eta[ang] * Snap::hj * hv_y[ang] +
                  Snap::xi[ang] * Snap::hk * hv_z[ang]);
                if (den < tolr)
                  pc[ang] = 0.0;
                else
                  pc[ang] /= den;
              }
            }
          }
          // Fixup done so compute the updated values
          for (int ang = 0; ang < Snap::num_angles; ang++)
            psii[ang] = fx_hv_x[ang] * hv_x[ang];
          for (int ang = 0; ang < Snap::num_angles; ang++)
            psij[ang] = fx_hv_y[ang] * hv_y[ang];
          for (int ang = 0; ang < Snap::num_angles; ang++)
            psik[ang] = fx_hv_z[ang] * hv_z[ang];
          if (vdelt != 0.0)
          {
            for (int ang = 0; ang < Snap::num_angles; ang++)
              time_flux_out[ang] = fx_hv_t[ang] * hv_t[ang];
            fa_time_flux_out.write_untyped(DomainPoint::from_point<3>(local_point),
                                           time_flux_out, angle_buffer_size);
          }
        } else {
          // NO FIXUP
          for (int ang = 0; ang < Snap::num_angles; ang++)
            psii[ang] = 2.0 * pc[ang] - psii[ang]; 
          for (int ang = 0; ang < Snap::num_angles; ang++)
            psij[ang] = 2.0 * pc[ang] - psij[ang];
          for (int ang = 0; ang < Snap::num_angles; ang++)
            psik[ang] = 2.0 * pc[ang] - psik[ang];
          if (vdelt != 0.0) 
          {
            // Write out the outgoing temporal flux
            for (int ang = 0; ang < Snap::num_angles; ang++)
              time_flux_out[ang] = 2.0 * pc[ang] - time_flux_in[ang];
            fa_time_flux_out.write_untyped(DomainPoint::from_point<3>(local_point),
                                           time_flux_out, angle_buffer_size);
          }
        }

        // Write out the ghost regions 
        // X ghost
        if (stride_x_positive) {
          // Writing to x+1
          if (x == (Snap::nx_per_chunk-1)) {
            // We write out on our own region
            fa_ghostx_out.write_untyped(DomainPoint::from_point<3>(local_point),
                                        psii, angle_buffer_size);
          } 
          // Else nothing: psii just gets caried over to next iteration
        } else {
          // Writing to x-1
          // Local coordinates here
          if (x == (Snap::nx_per_chunk-1)) {
            // Write out on our own region
            fa_ghostx_out.write_untyped(DomainPoint::from_point<3>(local_point),
                                        psii, angle_buffer_size);
          } 
          // Else nothing: psii just gets carried over to next iteration 
        }
        // Y ghost
        if (stride_y_positive) {
          // Writing to y+1
          if (y == (Snap::ny_per_chunk-1)) {
            // Write out on our own region
            fa_ghosty_out.write_untyped(DomainPoint::from_point<3>(local_point),
                                        psij, angle_buffer_size);
          } else {
            // Write to the pencil 
            const int offset = x * Snap::num_angles;
            memcpy(yflux_pencil+offset, psij, angle_buffer_size);
          }
        } else {
          // Writing to y-1
          // Local coordinates here
          if (y == (Snap::ny_per_chunk-1)) {
            // Write out on our own region
            fa_ghosty_out.write_untyped(DomainPoint::from_point<3>(local_point),
                                        psij, angle_buffer_size);
          } else {
            // Write to the pencil
            const int offset = x * Snap::num_angles;
            memcpy(yflux_pencil+offset, psij, angle_buffer_size);
          }
        }
        // Z ghost
        if (stride_z_positive) {
          // Writing to z+1
          if (z == (Snap::nz_per_chunk-1)) {
            // Write out on our own region
            fa_ghostz_out.write_untyped(DomainPoint::from_point<3>(local_point),
                                        psik, angle_buffer_size);
          } else {
            // Write to the plane
            const int offset = (y * x_range + x) * Snap::num_angles;
            memcpy(zflux_plane+offset, psik, angle_buffer_size);
          }
        } else {
          // Writing to z-1
          // Local coordinates here
          if (z == (Snap::nz_per_chunk-1)) {
            // Write out on our own region
            fa_ghostz_out.write_untyped(DomainPoint::from_point<3>(local_point),
                                        psik, angle_buffer_size);
          } else {
            // Write to the plane
            const int offset = (y * x_range + x) * Snap::num_angles;
            memcpy(zflux_plane+offset, psik, angle_buffer_size);
          }
        }

        // Finally we apply reductions to the flux moments
        double total = 0.0;
        for (int ang = 0; ang < Snap::num_angles; ang++) {
          psi[ang] = Snap::w[ang] * pc[ang]; 
          total += psi[ang];
        }
        double *local_flux = flux;
        for (int i = 0; i < 3; i++)
          local_flux += local_point[i] * flux_offsets[i];
        SumReduction::fold<false>(*local_flux, total);
        if (Snap::num_moments > 1) {
          MomentQuad quad;
          for (int l = 1; l < Snap::num_moments; l++) {
            unsigned offset = l * Snap::num_angles + 
              args->corner * Snap::num_angles * Snap::num_moments;
            total = 0.0;
            for (int ang = 0; ang < Snap::num_angles; ang++) {
              total += Snap::ec[offset+ang] * psi[ang]; 
            }
            quad[l] = total;
          }
          MomentQuad *local_fluxm = fluxm;
          for (int i = 0; i < 3; i++)
            local_fluxm += local_point[i] * fluxm_offsets[i];
          QuadReduction::fold<false>(*local_fluxm, quad);
        }
      }
    }
  }

  free(psi);
  free(pc);
  free(psii);
  free(psij);
  free(psik);
  free(time_flux_in);
  free(time_flux_out);
  free(temp_array);
  free(hv_x);
  free(hv_y);
  free(hv_z);
  free(hv_t);
  free(fx_hv_x);
  free(fx_hv_y);
  free(fx_hv_z);
  free(fx_hv_t);
  free(yflux_pencil);
  free(zflux_plane);
#endif
}

inline ByteOffset operator*(const ByteOffset offsets[3], const Point<3> &point)
{
  return (offsets[0] * point.x[0] + offsets[1] * point.x[1] + offsets[2] * point.x[2]);
}

inline __m128d* get_sse_angle_ptr(void *ptr, const ByteOffset offsets[3],
                                  const Point<3> &point, size_t angle_buffer_size)
{
  return (__m128d*)(ptr + ((offsets * point) * angle_buffer_size));
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::sse_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.print("Running SSE Mini-KBA Sweep");

  assert(task->arglen == sizeof(MiniKBAArgs));
  const MiniKBAArgs *args = reinterpret_cast<const MiniKBAArgs*>(task->args);
    
  // This implementation of the sweep assumes three dimensions
  assert(Snap::num_dims == 3);

  RegionAccessor<AccessorType::Generic,MomentQuad> fa_qtot = 
    regions[0].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<MomentQuad>();
  RegionAccessor<AccessorType::Generic,double> fa_flux = 
    regions[1].get_accessor().typeify<double>();
  RegionAccessor<AccessorType::Generic,MomentQuad> fa_fluxm = 
    regions[2].get_accessor().typeify<MomentQuad>();
  ByteOffset qtot_offsets[3], flux_offsets[3], fluxm_offsets[3];
  const MomentQuad *const qtot_ptr = fa_qtot.raw_rect_ptr<3>(qtot_offsets);
  double *const flux = fa_flux.raw_rect_ptr<3>(flux_offsets);
  MomentQuad *const fluxm = fa_fluxm.raw_rect_ptr<3>(fluxm_offsets);

  // No types here since the size of these fields are dependent
  // on the number of angles
  const double vdelt = regions[regions.size()-1].get_field_accessor(
      SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<double>().read(
      DomainPoint::from_point<1>(Point<1>::ZEROES()));
  RegionAccessor<AccessorType::Generic> fa_dinv = 
    regions[3].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic> fa_time_flux_in = 
    regions[4].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic> fa_time_flux_out = 
    regions[5].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic,double> fa_t_xs = 
    regions[6].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<double>();
  ByteOffset dinv_offsets[3], time_flux_in_offsets[3], 
             time_flux_out_offsets[3], t_xs_offsets[3];
  void *const dinv_ptr = fa_dinv.raw_rect_ptr<3>(dinv_offsets);
  void *const time_flux_in_ptr = 
    fa_time_flux_in.raw_rect_ptr<3>(time_flux_in_offsets);
  void *const time_flux_out_ptr = 
    fa_time_flux_out.raw_rect_ptr<3>(time_flux_out_offsets);
  const double *const t_xs_ptr = fa_t_xs.raw_rect_ptr<3>(t_xs_offsets);

  // Output ghost regions
  RegionAccessor<AccessorType::Generic> fa_ghostx_out = 
    regions[7].get_field_accessor(
        *(task->regions[7].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_ghosty_out = 
    regions[8].get_field_accessor(
        *(task->regions[8].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_ghostz_out = 
    regions[9].get_field_accessor(
        *(task->regions[9].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_qim = 
    regions[10].get_field_accessor(
        *(task->regions[10].privilege_fields.begin()));
  ByteOffset ghostx_out_offsets[3], ghosty_out_offsets[3], 
             ghostz_out_offsets[3], qim_offsets[3];
  void *const ghostx_out_ptr = fa_ghostx_out.raw_rect_ptr<3>(ghostx_out_offsets);
  void *const ghosty_out_ptr = fa_ghosty_out.raw_rect_ptr<3>(ghosty_out_offsets);
  void *const ghostz_out_ptr = fa_ghostz_out.raw_rect_ptr<3>(ghostz_out_offsets);
  void *const qim_ptr = fa_qim.raw_rect_ptr<3>(qim_offsets);

  // Input ghost regions
  RegionAccessor<AccessorType::Generic> fa_ghostx_in = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS].get_field_accessor(
        *(task->regions[MINI_KBA_NON_GHOST_REQUIREMENTS].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_ghosty_in = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS+1].get_field_accessor(
        *(task->regions[MINI_KBA_NON_GHOST_REQUIREMENTS+1].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_ghostz_in = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS+2].get_field_accessor(
        *(task->regions[MINI_KBA_NON_GHOST_REQUIREMENTS+2].privilege_fields.begin()));
  ByteOffset ghostx_in_offsets[3], ghosty_in_offsets[3], ghostz_in_offsets[3];
  void *const ghostx_in_ptr = fa_ghostx_in.raw_rect_ptr<3>(ghostx_in_offsets);
  void *const ghosty_in_ptr = fa_ghosty_in.raw_rect_ptr<3>(ghosty_in_offsets);
  void *const ghostz_in_ptr = fa_ghostz_in.raw_rect_ptr<3>(ghostz_in_offsets);

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  // Figure out the origin point based on which corner we are
  const bool stride_x_positive = ((args->corner & 0x1) != 0);
  const bool stride_y_positive = ((args->corner & 0x2) != 0);
  const bool stride_z_positive = ((args->corner & 0x4) != 0);
  const coord_t origin_ints[3] = { 
    (stride_x_positive ? subgrid_bounds.lo[0] : subgrid_bounds.hi[0]),
    (stride_y_positive ? subgrid_bounds.lo[1] : subgrid_bounds.hi[1]),
    (stride_z_positive ? subgrid_bounds.lo[2] : subgrid_bounds.hi[2]) };
  const Point<3> origin(origin_ints);

  // Local arrays
  assert((Snap::num_angles % 2) == 0);
  const int num_vec_angles = Snap::num_angles/2;
  const size_t angle_buffer_size = num_vec_angles * sizeof(__m128d);
  __m128d *psi = (__m128d*)malloc(angle_buffer_size);
  __m128d *pc = (__m128d*)malloc(angle_buffer_size);
  __m128d *psii = (__m128d*)malloc(angle_buffer_size);
  __m128d *hv_x = (__m128d*)malloc(angle_buffer_size);
  __m128d *hv_y = (__m128d*)malloc(angle_buffer_size);
  __m128d *hv_z = (__m128d*)malloc(angle_buffer_size);
  __m128d *hv_t = (__m128d*)malloc(angle_buffer_size);
  __m128d *fx_hv_x = (__m128d*)malloc(angle_buffer_size);
  __m128d *fx_hv_y = (__m128d*)malloc(angle_buffer_size);
  __m128d *fx_hv_z = (__m128d*)malloc(angle_buffer_size);
  __m128d *fx_hv_t = (__m128d*)malloc(angle_buffer_size);

  const __m128d tolr = _mm_set1_pd(1.0e-12);

  // See note in the CPU implementation about why we do things this way
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  __m128d *yflux_pencil = (__m128d*)malloc(x_range * angle_buffer_size);
  __m128d *zflux_plane  = (__m128d*)malloc(y_range * x_range * angle_buffer_size);

  for (int z = 0; z < z_range; z++) {
    for (int y = 0; y < y_range; y++) {
      for (int x = 0; x < x_range; x++) {
        // Figure out the local point that we are working on    
        Point<3> local_point = origin;
        if (stride_x_positive)
          local_point.x[0] += x;
        else
          local_point.x[0] -= x;
        if (stride_y_positive)
          local_point.x[1] += y;
        else
          local_point.x[1] -= y;
        if (stride_z_positive)
          local_point.x[2] += z;
        else
          local_point.x[2] -= z;

        // Compute the angular source
        MomentQuad quad = *(qtot_ptr + qtot_offsets * local_point);
        for (int ang = 0; ang < num_vec_angles; ang++)
          psi[ang] = _mm_set1_pd(quad[0]);
        if (Snap::num_moments > 1) {
          const int corner_offset = 
            args->corner * Snap::num_angles * Snap::num_moments;
          for (unsigned l = 1; l < Snap::num_moments; l++) {
            const int moment_offset = corner_offset + l * Snap::num_angles;
            for (int ang = 0; ang < num_vec_angles; ang++) {
              psi[ang] = _mm_add_pd(psi[ang], _mm_mul_pd(
                    _mm_set_pd(Snap::ec[moment_offset+2*ang+1],
                               Snap::ec[moment_offset+2*ang]),
                    _mm_set1_pd(quad[l])));
            }
          }
        }

        // If we're doing MMS, there is an additional term
        if (Snap::source_layout == Snap::MMS_SOURCE)
        {
          __m128d *qim = (__m128d*)(qim_ptr + qim_offsets * local_point);
          for (int ang = 0; ang < num_vec_angles; ang++)
            psi[ang] = _mm_add_pd(psi[ang], qim[ang]);
        }

        // Compute the initial solution
        for (int ang = 0; ang < num_vec_angles; ang++)
          pc[ang] = psi[ang];
        // X ghost cells
        if (stride_x_positive) {
          // reading from x-1 
          Point<3> ghost_point = local_point;
          ghost_point.x[0] -= 1;
          if (x == 0) {
            // Ghost cell array
            memcpy(psii, get_sse_angle_ptr(ghostx_in_ptr, ghostx_in_offsets,
                  ghost_point, angle_buffer_size), angle_buffer_size);
          } 
          // Else nothing: psii already contains next flux
        } else {
          // reading from x+1
          Point<3> ghost_point = local_point;
          ghost_point.x[0] += 1;
          // Local coordinates here
          if (x == 0) {
            // Ghost cell array
            memcpy(psii, get_sse_angle_ptr(ghostx_in_ptr, ghostx_in_offsets,
                  ghost_point, angle_buffer_size), angle_buffer_size);
          }
          // Else nothing: psii already contains next flux
        }
        for (int ang = 0; ang < num_vec_angles; ang++)
          pc[ang] = _mm_add_pd(pc[ang], _mm_mul_pd( _mm_mul_pd(psii[ang], 
                  _mm_set_pd(Snap::mu[2*ang+1],Snap::mu[2*ang])), 
                  _mm_set1_pd(Snap::hi)));
        // Y ghost cells
        __m128d *psij;
        if (stride_y_positive) {
          // reading from y-1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] -= 1;
          if (y == 0) {
            // Ghost cell array
            psij = get_sse_angle_ptr(ghosty_in_ptr, ghosty_in_offsets,
                                     ghost_point, angle_buffer_size);
          } else {
            // Local array
            psij = yflux_pencil + x * num_vec_angles;
          }
        } else {
          // reading from y+1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] += 1;
          // Local coordinates here
          if (y == 0) {
            // Ghost cell array
            psij = get_sse_angle_ptr(ghosty_in_ptr, ghosty_in_offsets,
                                     ghost_point, angle_buffer_size);
          } else {
            // Local array
            psij = yflux_pencil + x * num_vec_angles;
          }
        }
        for (int ang = 0; ang < num_vec_angles; ang++)
          pc[ang] = _mm_add_pd(pc[ang], _mm_mul_pd( _mm_mul_pd(psij[ang],
                  _mm_set_pd(Snap::eta[2*ang+1], Snap::eta[2*ang])),
                  _mm_set1_pd(Snap::hj)));
        // Z ghost cells
        __m128d *psik;
        if (stride_z_positive) {
          // reading from z-1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] -= 1;
          if (z == 0) {
            // Ghost cell array
            psik = get_sse_angle_ptr(ghostz_in_ptr, ghostz_in_offsets,
                                     ghost_point, angle_buffer_size);
          } else {
            // Local array
            psik = zflux_plane + (y * x_range + x) * num_vec_angles;
          }
        } else {
          // reading from z+1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] += 1;
          // Local coordinates here
          if (z == 0) {
            // Ghost cell array
            psik = get_sse_angle_ptr(ghostz_in_ptr, ghostz_in_offsets,
                                     ghost_point, angle_buffer_size);
          } else {
            // Local array
            psik = zflux_plane + (y * x_range + x) * num_vec_angles;
          }
        }
        for (int ang = 0; ang < Snap::num_angles; ang++)
          pc[ang] = _mm_add_pd(pc[ang], _mm_mul_pd( _mm_mul_pd(psik[ang],
                  _mm_set_pd(Snap::xi[2*ang+1], Snap::xi[2*ang])),
                  _mm_set1_pd(Snap::hk)));
        // See if we're doing anything time dependent
        __m128d *time_flux_in = get_sse_angle_ptr(time_flux_in_ptr,
              time_flux_in_offsets, local_point, angle_buffer_size);
        if (vdelt != 0.0) 
        {
          for (int ang = 0; ang < num_vec_angles; ang++)
            pc[ang] = _mm_add_pd(pc[ang], _mm_mul_pd(
                  _mm_set1_pd(vdelt), time_flux_in[ang]));
        }
        // Multiple by the precomputed denominator inverse
        __m128d *dinv = get_sse_angle_ptr(dinv_ptr, dinv_offsets, 
                                          local_point, angle_buffer_size);
        for (int ang = 0; ang < num_vec_angles; ang++)
          pc[ang] = _mm_mul_pd(pc[ang], dinv[ang]);

        if (Snap::flux_fixup) {
          // DO THE FIXUP
          unsigned old_negative_fluxes = 0;
          for (int ang = 0; ang < num_vec_angles; ang++)
            hv_x[ang] = _mm_set1_pd(1.0);
          for (int ang = 0; ang < num_vec_angles; ang++)
            hv_y[ang] = _mm_set1_pd(1.0);
          for (int ang = 0; ang < num_vec_angles; ang++)
            hv_z[ang] = _mm_set1_pd(1.0);
          for (int ang = 0; ang < num_vec_angles; ang++)
            hv_t[ang] = _mm_set1_pd(1.0);

          const double t_xs = *(t_xs_ptr + t_xs_offsets * local_point);
          while (true) {
            unsigned negative_fluxes = 0;
            // Figure out how many negative fluxes we have
            for (int ang = 0; ang < num_vec_angles; ang++) {
              fx_hv_x[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), psii[ang]);
              __m128d ge = _mm_cmp_pd(fx_hv_x[ang], _mm_set1_pd(0.0), _CMP_GE_OS);
              // If not greater than or equal, set back to zero
              hv_x[ang] = _mm_and_pd(ge, hv_x[ang]);
              // Count how many negative fluxes we had
              __m128i negatives = _mm_andnot_si128(_mm_castpd_si128(ge), 
                                                   _mm_set_epi32(0, 1, 0, 1));
              negative_fluxes += _mm_extract_epi32(negatives, 0);
              negative_fluxes += _mm_extract_epi32(negatives, 2);
            }
            for (int ang = 0; ang < num_vec_angles; ang++) {
              fx_hv_y[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), psij[ang]);
              __m128d ge = _mm_cmp_pd(fx_hv_y[ang], _mm_set1_pd(0.0), _CMP_GE_OS);
              // If not greater than or equal set back to zero
              hv_y[ang] = _mm_and_pd(ge, hv_y[ang]);
              // Count how many negative fluxes we had
              __m128i negatives = _mm_andnot_si128(_mm_castpd_si128(ge), 
                                                   _mm_set_epi32(0, 1, 0, 1));
              negative_fluxes += _mm_extract_epi32(negatives, 0);
              negative_fluxes += _mm_extract_epi32(negatives, 2);
            }
            for (int ang = 0; ang < num_vec_angles; ang++) {
              fx_hv_z[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), psik[ang]);
              __m128d ge = _mm_cmp_pd(fx_hv_z[ang], _mm_set1_pd(0.0), _CMP_GE_OS);
              // If not greater than or equal set back to zero
              hv_z[ang] = _mm_and_pd(ge, hv_z[ang]);
              // Count how many negative fluxes we had
              __m128i negatives = _mm_andnot_si128(_mm_castpd_si128(ge), 
                                                   _mm_set_epi32(0, 1, 0, 1));
              negative_fluxes += _mm_extract_epi32(negatives, 0);
              negative_fluxes += _mm_extract_epi32(negatives, 2);
            }
            if (vdelt != 0.0) {
              for (int ang = 0; ang < num_vec_angles; ang++) {
                fx_hv_t[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), 
                                                      time_flux_in[ang]);
                __m128d ge = _mm_cmp_pd(fx_hv_t[ang], _mm_set1_pd(0.0), _CMP_GE_OS);
                // If not greater than or equal, set back to zero
                hv_t[ang] = _mm_and_pd(ge, hv_t[ang]);
                // Count how many negative fluxes we had
                __m128i negatives = _mm_andnot_si128(_mm_castpd_si128(ge), 
                                                     _mm_set_epi32(0, 1, 0, 1));
                negative_fluxes += _mm_extract_epi32(negatives, 0);
                negative_fluxes += _mm_extract_epi32(negatives, 2);
              }
            }
            if (negative_fluxes == old_negative_fluxes)
              break;
            old_negative_fluxes = negative_fluxes;
            if (vdelt != 0.0) {
              for (int ang = 0; ang < num_vec_angles; ang++) {
                __m128d sum = _mm_mul_pd(psii[ang], _mm_mul_pd(
                      _mm_set_pd(Snap::mu[2*ang+1], Snap::mu[2*ang]), 
                      _mm_mul_pd( _mm_set1_pd(Snap::hi), 
                        _mm_add_pd( _mm_set1_pd(1.0), hv_x[ang]))));
                sum = _mm_add_pd(sum, _mm_mul_pd(psij[ang], _mm_mul_pd(
                        _mm_set_pd(Snap::eta[2*ang+1], Snap::eta[2*ang]),
                        _mm_mul_pd( _mm_set1_pd(Snap::hj),
                          _mm_add_pd( _mm_set1_pd(1.0), hv_y[ang])))));
                sum = _mm_add_pd(sum, _mm_mul_pd(psik[ang], _mm_mul_pd(
                        _mm_set_pd(Snap::xi[2*ang+1], Snap::xi[2*ang]),
                        _mm_mul_pd( _mm_set1_pd(Snap::hk),
                          _mm_add_pd( _mm_set1_pd(1.0), hv_z[ang])))));
                sum = _mm_add_pd(sum, _mm_mul_pd(time_flux_in[ang], 
                      _mm_mul_pd( _mm_set1_pd(vdelt), _mm_add_pd(
                          _mm_set1_pd(1.0), hv_t[ang]))));
                pc[ang] = _mm_add_pd(psi[ang], _mm_mul_pd( _mm_set1_pd(0.5), sum));
                __m128d den = _mm_add_pd(_mm_set1_pd(t_xs), 
                    _mm_add_pd( _mm_add_pd( _mm_add_pd(
                      _mm_mul_pd( _mm_mul_pd( _mm_set_pd(Snap::mu[2*ang+1], 
                            Snap::mu[2*ang]), _mm_set1_pd(Snap::hi)), hv_x[ang]),
                      _mm_mul_pd( _mm_mul_pd( _mm_set_pd(Snap::eta[2*ang+1],
                            Snap::eta[2*ang]), _mm_set1_pd(Snap::hj)), hv_y[ang])),
                      _mm_mul_pd( _mm_mul_pd( _mm_set_pd(Snap::xi[2*ang+1],
                            Snap::xi[2*ang]), _mm_set1_pd(Snap::hk)), hv_z[ang])),
                      _mm_mul_pd(_mm_set1_pd(vdelt), hv_t[ang])));
                __m128d pc_ge = _mm_cmp_pd(pc[ang], _mm_set1_pd(0.0), _CMP_GE_OS);
                // Set the denominator back to zero if it is too small
                den = _mm_and_pd(den, pc_ge);
                __m128d den_ge = _mm_cmp_pd(den, tolr, _CMP_GE_OS);
                pc[ang] = _mm_or_pd(
                    _mm_and_pd(den_ge, _mm_div_pd(pc[ang], den)),
                    _mm_andnot_pd(den_ge, _mm_set1_pd(0.0)));
              }
            } else {
              for (int ang = 0; ang < num_vec_angles; ang++) {
                __m128d sum = _mm_mul_pd(psii[ang], _mm_mul_pd(
                      _mm_set_pd(Snap::mu[2*ang+1], Snap::mu[2*ang]), 
                      _mm_mul_pd( _mm_set1_pd(Snap::hi), 
                        _mm_add_pd( _mm_set1_pd(1.0), hv_x[ang]))));
                sum = _mm_add_pd(sum, _mm_mul_pd(psij[ang], _mm_mul_pd(
                        _mm_set_pd(Snap::eta[2*ang+1], Snap::eta[2*ang]),
                        _mm_mul_pd( _mm_set1_pd(Snap::hj),
                          _mm_add_pd( _mm_set1_pd(1.0), hv_y[ang])))));
                sum = _mm_add_pd(sum, _mm_mul_pd(psik[ang], _mm_mul_pd(
                        _mm_set_pd(Snap::xi[2*ang+1], Snap::xi[2*ang]),
                        _mm_mul_pd( _mm_set1_pd(Snap::hk),
                          _mm_add_pd( _mm_set1_pd(1.0), hv_z[ang])))));
                pc[ang] = _mm_add_pd(psi[ang], _mm_mul_pd( _mm_set1_pd(0.5), sum));
                __m128d den = _mm_add_pd(_mm_set1_pd(t_xs), _mm_add_pd( _mm_add_pd( 
                      _mm_mul_pd( _mm_mul_pd( _mm_set_pd(Snap::mu[2*ang+1], 
                            Snap::mu[2*ang]), _mm_set1_pd(Snap::hi)), hv_x[ang]),
                      _mm_mul_pd( _mm_mul_pd( _mm_set_pd(Snap::eta[2*ang+1],
                            Snap::eta[2*ang]), _mm_set1_pd(Snap::hj)), hv_y[ang])),
                      _mm_mul_pd( _mm_mul_pd( _mm_set_pd(Snap::xi[2*ang+1],
                            Snap::xi[2*ang]), _mm_set1_pd(Snap::hk)), hv_z[ang])));
                __m128d pc_ge = _mm_cmp_pd(pc[ang], _mm_set1_pd(0.0), _CMP_GE_OS);
                // Set the denominator back to zero if it is too small
                den = _mm_and_pd(den, pc_ge);
                __m128d den_ge = _mm_cmp_pd(den, tolr, _CMP_GE_OS);
                pc[ang] = _mm_or_pd(
                    _mm_and_pd(den_ge, _mm_div_pd(pc[ang], den)),
                    _mm_andnot_pd(den_ge, _mm_set1_pd(0.0)));
              }
            }
          }
          // Fixup done so compute the update values
          for (int ang = 0; ang < num_vec_angles; ang++)
            psii[ang] = _mm_mul_pd(fx_hv_x[ang], hv_x[ang]);
          for (int ang = 0; ang < num_vec_angles; ang++)
            psij[ang] = _mm_mul_pd(fx_hv_y[ang], hv_y[ang]);
          for (int ang = 0; ang < num_vec_angles; ang++)
            psik[ang] = _mm_mul_pd(fx_hv_z[ang], hv_z[ang]);
          if (vdelt != 0.0)
          {
            // Write out the outgoing temporal flux 
            __m128d *time_flux_out = 
              get_sse_angle_ptr(time_flux_out_ptr, time_flux_out_offsets,
                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm_stream_pd((double*)(time_flux_out+ang), 
                  _mm_mul_pd(fx_hv_t[ang], hv_t[ang]));
          }
        } else {
          // NO FIXUP
          for (int ang = 0; ang < num_vec_angles; ang++)
            psii[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), psii[ang]);
          for (int ang = 0; ang < num_vec_angles; ang++)
            psij[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), psij[ang]);
          for (int ang = 0; ang < num_vec_angles; ang++)
            psik[ang] = _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), psik[ang]);
          if (vdelt != 0.0) 
          {
            // Write out the outgoing temporal flux 
            __m128d *time_flux_out = 
              get_sse_angle_ptr(time_flux_out_ptr, time_flux_out_offsets,
                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm_stream_pd((double*)(time_flux_out+ang), 
                  _mm_sub_pd( _mm_mul_pd( _mm_set1_pd(2.0), pc[ang]), time_flux_in[ang]));
          }
        }
        // Write out the ghost regions
        if (stride_x_positive) {
          // Writing to x+1
          if (x == (Snap::nx_per_chunk-1)) {
            // We write out on our own region
            __m128d *target = get_sse_angle_ptr(ghostx_out_ptr, ghostx_out_offsets,
                                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm_stream_pd((double*)(target+ang), psii[ang]);
          } 
          // Else nothing: psii just gets caried over to next iteration
        } else {
          // Writing to x-1
          // Local coordinates here
          if (x == (Snap::nx_per_chunk-1)) {
            // Write out on our own region
            __m128d *target = get_sse_angle_ptr(ghostx_out_ptr, ghostx_out_offsets,
                                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm_stream_pd((double*)(target+ang), psii[ang]);
          } 
          // Else nothing: psii just gets carried over to next iteration 
        }
        // Y ghost
        if (stride_y_positive) {
          // Writing to y+1
          if (y == (Snap::ny_per_chunk-1)) {
            // Write out on our own region
            __m128d *target = get_sse_angle_ptr(ghosty_out_ptr, ghosty_out_offsets,
                                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm_stream_pd((double*)(target+ang), psij[ang]);
          } 
          // Else nothing: psij is already in place in the pencil
        } else {
          // Writing to y-1
          // Local coordinates here
          if (y == (Snap::ny_per_chunk-1)) {
            // Write out on our own region
            __m128d *target = get_sse_angle_ptr(ghosty_out_ptr, ghosty_out_offsets,
                                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm_stream_pd((double*)(target+ang), psij[ang]);
          } 
          // Else nothing: psij already in place in the pencil
        }
        // Z ghost
        if (stride_z_positive) {
          // Writing to z+1
          if (z == (Snap::nz_per_chunk-1)) {
            __m128d *target = get_sse_angle_ptr(ghostz_out_ptr, ghostz_out_offsets,
                                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm_stream_pd((double*)(target+ang), psik[ang]);
          } 
          // Else nothing: psik is already in place in the plane
        } else {
          // Writing to z-1
          // Local coordinates here
          if (z == (Snap::nz_per_chunk-1)) {
            // Write out on our own region
            __m128d *target = get_sse_angle_ptr(ghostz_out_ptr, ghostz_out_offsets,
                                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm_stream_pd((double*)(target+ang), psik[ang]);
          } 
          // Else nothing: psik is already in place in the plane
        }

        // Finally we apply reductions to the flux moments
        __m128d vec_total = _mm_set1_pd(0.0);
        for (int ang = 0; ang < num_vec_angles; ang++)
          vec_total = _mm_add_pd(vec_total, psi[ang]);
        double total = _mm_cvtsd_f64(_mm_hadd_pd(vec_total, vec_total));
        SumReduction::fold<false>(*(flux + flux_offsets * local_point), total);
        if (Snap::num_moments > 1) {
          MomentQuad quad;
          for (int l = 1; l < Snap::num_moments; l++) {
            unsigned offset = l * Snap::num_angles + 
              args->corner * Snap::num_angles * Snap::num_moments;
            vec_total = _mm_set1_pd(0.0);
            for (int ang = 0; ang < num_vec_angles; ang++)
              vec_total = _mm_add_pd(vec_total, _mm_mul_pd(psi[ang],
                    _mm_set_pd(Snap::ec[offset+2*ang+1], Snap::ec[offset+2*ang])));
            quad[l] = _mm_cvtsd_f64(_mm_hadd_pd(vec_total, vec_total));
          }
          QuadReduction::fold<false>(*(fluxm + fluxm_offsets * local_point), quad);
        }
      }
    }
  }

  free(psi);
  free(pc);
  free(psii);
  free(hv_x);
  free(hv_y);
  free(hv_z);
  free(hv_t);
  free(fx_hv_x);
  free(fx_hv_y);
  free(fx_hv_z);
  free(fx_hv_t);
  free(yflux_pencil);
  free(zflux_plane);
#endif
}

inline __m256d* get_avx_angle_ptr(void *ptr, const ByteOffset offsets[3],
                                  const Point<3> &point, size_t angle_buffer_size)
{
  return (__m256d*)(ptr + ((offsets * point) * angle_buffer_size));
}

inline __m256d* malloc_avx_aligned(size_t size)
{
  __m256d *result;
  posix_memalign((void**)&result, 32, size);
  return result;
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::avx_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.print("Running AVX Mini-KBA Sweep");

  assert(task->arglen == sizeof(MiniKBAArgs));
  const MiniKBAArgs *args = reinterpret_cast<const MiniKBAArgs*>(task->args);
    
  // This implementation of the sweep assumes three dimensions
  assert(Snap::num_dims == 3);

  RegionAccessor<AccessorType::Generic,MomentQuad> fa_qtot = 
    regions[0].get_field_accessor(
        SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<MomentQuad>();
  RegionAccessor<AccessorType::Generic,double> fa_flux = 
    regions[1].get_accessor().typeify<double>();
  RegionAccessor<AccessorType::Generic,MomentQuad> fa_fluxm = 
    regions[2].get_accessor().typeify<MomentQuad>();
  ByteOffset qtot_offsets[3], flux_offsets[3], fluxm_offsets[3];
  const MomentQuad *const qtot_ptr = fa_qtot.raw_rect_ptr<3>(qtot_offsets);
  double *const flux = fa_flux.raw_rect_ptr<3>(flux_offsets);
  MomentQuad *const fluxm = fa_fluxm.raw_rect_ptr<3>(fluxm_offsets);

  // No types here since the size of these fields are dependent
  // on the number of angles
  const double vdelt = regions[regions.size()-1].get_field_accessor(
      SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<double>().read(
      DomainPoint::from_point<1>(Point<1>::ZEROES()));
  RegionAccessor<AccessorType::Generic> fa_dinv = 
    regions[3].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic> fa_time_flux_in = 
    regions[4].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic> fa_time_flux_out = 
    regions[5].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group));
  RegionAccessor<AccessorType::Generic,double> fa_t_xs = 
    regions[6].get_field_accessor(SNAP_ENERGY_GROUP_FIELD(args->group)).typeify<double>();
  ByteOffset dinv_offsets[3], time_flux_in_offsets[3], 
             time_flux_out_offsets[3], t_xs_offsets[3];
  void *const dinv_ptr = fa_dinv.raw_rect_ptr<3>(dinv_offsets);
  void *const time_flux_in_ptr = 
    fa_time_flux_in.raw_rect_ptr<3>(time_flux_in_offsets);
  void *const time_flux_out_ptr = 
    fa_time_flux_out.raw_rect_ptr<3>(time_flux_out_offsets);
  const double *const t_xs_ptr = fa_t_xs.raw_rect_ptr<3>(t_xs_offsets);

  // Output ghost regions
  RegionAccessor<AccessorType::Generic> fa_ghostx_out = 
    regions[7].get_field_accessor(
        *(task->regions[7].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_ghosty_out = 
    regions[8].get_field_accessor(
        *(task->regions[8].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_ghostz_out = 
    regions[9].get_field_accessor(
        *(task->regions[9].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_qim = 
    regions[10].get_field_accessor(
        *(task->regions[10].privilege_fields.begin()));
  ByteOffset ghostx_out_offsets[3], ghosty_out_offsets[3], 
             ghostz_out_offsets[3], qim_offsets[3];
  void *const ghostx_out_ptr = fa_ghostx_out.raw_rect_ptr<3>(ghostx_out_offsets);
  void *const ghosty_out_ptr = fa_ghosty_out.raw_rect_ptr<3>(ghosty_out_offsets);
  void *const ghostz_out_ptr = fa_ghostz_out.raw_rect_ptr<3>(ghostz_out_offsets);
  void *const qim_ptr = fa_qim.raw_rect_ptr<3>(qim_offsets);

  // Input ghost regions
  RegionAccessor<AccessorType::Generic> fa_ghostx_in = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS].get_field_accessor(
        *(task->regions[MINI_KBA_NON_GHOST_REQUIREMENTS].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_ghosty_in = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS+1].get_field_accessor(
        *(task->regions[MINI_KBA_NON_GHOST_REQUIREMENTS+1].privilege_fields.begin()));
  RegionAccessor<AccessorType::Generic> fa_ghostz_in = 
    regions[MINI_KBA_NON_GHOST_REQUIREMENTS+2].get_field_accessor(
        *(task->regions[MINI_KBA_NON_GHOST_REQUIREMENTS+2].privilege_fields.begin()));
  ByteOffset ghostx_in_offsets[3], ghosty_in_offsets[3], ghostz_in_offsets[3];
  void *const ghostx_in_ptr = fa_ghostx_in.raw_rect_ptr<3>(ghostx_in_offsets);
  void *const ghosty_in_ptr = fa_ghosty_in.raw_rect_ptr<3>(ghosty_in_offsets);
  void *const ghostz_in_ptr = fa_ghostz_in.raw_rect_ptr<3>(ghostz_in_offsets);

  Domain dom = runtime->get_index_space_domain(ctx, 
          task->regions[0].region.get_index_space());
  Rect<3> subgrid_bounds = dom.get_rect<3>();

  // Figure out the origin point based on which corner we are
  const bool stride_x_positive = ((args->corner & 0x1) != 0);
  const bool stride_y_positive = ((args->corner & 0x2) != 0);
  const bool stride_z_positive = ((args->corner & 0x4) != 0);
  const coord_t origin_ints[3] = { 
    (stride_x_positive ? subgrid_bounds.lo[0] : subgrid_bounds.hi[0]),
    (stride_y_positive ? subgrid_bounds.lo[1] : subgrid_bounds.hi[1]),
    (stride_z_positive ? subgrid_bounds.lo[2] : subgrid_bounds.hi[2]) };
  const Point<3> origin(origin_ints);

  // Local arrays
  assert((Snap::num_angles % 4) == 0);
  const int num_vec_angles = Snap::num_angles/4;
  const size_t angle_buffer_size = num_vec_angles * sizeof(__m256d);
  __m256d *psi = malloc_avx_aligned(angle_buffer_size);
  __m256d *pc = malloc_avx_aligned(angle_buffer_size);
  __m256d *psii = malloc_avx_aligned(angle_buffer_size);
  __m256d *hv_x = malloc_avx_aligned(angle_buffer_size);
  __m256d *hv_y = malloc_avx_aligned(angle_buffer_size);
  __m256d *hv_z = malloc_avx_aligned(angle_buffer_size);
  __m256d *hv_t = malloc_avx_aligned(angle_buffer_size);
  __m256d *fx_hv_x = malloc_avx_aligned(angle_buffer_size);
  __m256d *fx_hv_y = malloc_avx_aligned(angle_buffer_size);
  __m256d *fx_hv_z = malloc_avx_aligned(angle_buffer_size);
  __m256d *fx_hv_t = malloc_avx_aligned(angle_buffer_size);

  const __m256d tolr = _mm256_set1_pd(1.0e-12);

  // See note in the CPU implementation about why we do things this way
  const int x_range = (subgrid_bounds.hi[0] - subgrid_bounds.lo[0]) + 1; 
  const int y_range = (subgrid_bounds.hi[1] - subgrid_bounds.lo[1]) + 1;
  const int z_range = (subgrid_bounds.hi[2] - subgrid_bounds.lo[2]) + 1;
  __m256d *yflux_pencil = malloc_avx_aligned(x_range * angle_buffer_size);
  __m256d *zflux_plane  = malloc_avx_aligned(y_range * x_range * angle_buffer_size);

  for (int z = 0; z < z_range; z++) {
    for (int y = 0; y < y_range; y++) {
      for (int x = 0; x < x_range; x++) {
        // Figure out the local point that we are working on    
        Point<3> local_point = origin;
        if (stride_x_positive)
          local_point.x[0] += x;
        else
          local_point.x[0] -= x;
        if (stride_y_positive)
          local_point.x[1] += y;
        else
          local_point.x[1] -= y;
        if (stride_z_positive)
          local_point.x[2] += z;
        else
          local_point.x[2] -= z;

        // Compute the angular source
        MomentQuad quad = *(qtot_ptr + qtot_offsets * local_point);
        for (int ang = 0; ang < num_vec_angles; ang++)
          psi[ang] = _mm256_set1_pd(quad[0]);
        if (Snap::num_moments > 1) {
          const int corner_offset = 
            args->corner * Snap::num_angles * Snap::num_moments;
          for (unsigned l = 1; l < Snap::num_moments; l++) {
            const int moment_offset = corner_offset + l * Snap::num_angles;
            for (int ang = 0; ang < num_vec_angles; ang++) {
              psi[ang] = _mm256_add_pd(psi[ang], _mm256_mul_pd(
                    _mm256_set_pd(Snap::ec[moment_offset+4*ang+3],
                                  Snap::ec[moment_offset+4*ang+2],
                                  Snap::ec[moment_offset+4*ang+1],
                                  Snap::ec[moment_offset+4*ang]),
                    _mm256_set1_pd(quad[l])));
            }
          }
        }

        // If we're doing MMS, there is an additional term
        if (Snap::source_layout == Snap::MMS_SOURCE)
        {
          __m256d *qim = (__m256d*)(qim_ptr + qim_offsets * local_point);
          for (int ang = 0; ang < num_vec_angles; ang++)
            psi[ang] = _mm256_add_pd(psi[ang], qim[ang]);
        }

        // Compute the initial solution
        for (int ang = 0; ang < num_vec_angles; ang++)
          pc[ang] = psi[ang];
        // X ghost cells
        if (stride_x_positive) {
          // reading from x-1 
          Point<3> ghost_point = local_point;
          ghost_point.x[0] -= 1;
          if (x == 0) {
            // Ghost cell array
            memcpy(psii, get_avx_angle_ptr(ghostx_in_ptr, ghostx_in_offsets,
                  ghost_point, angle_buffer_size), angle_buffer_size);
          } 
          // Else nothing: psii already contains next flux
        } else {
          // reading from x+1
          Point<3> ghost_point = local_point;
          ghost_point.x[0] += 1;
          // Local coordinates here
          if (x == 0) {
            // Ghost cell array
            memcpy(psii, get_avx_angle_ptr(ghostx_in_ptr, ghostx_in_offsets,
                  ghost_point, angle_buffer_size), angle_buffer_size);
          }
          // Else nothing: psii already contains next flux
        }
        for (int ang = 0; ang < num_vec_angles; ang++)
          pc[ang] = _mm256_add_pd(pc[ang], _mm256_mul_pd( _mm256_mul_pd(psii[ang], 
                  _mm256_set_pd(Snap::mu[4*ang+3], Snap::mu[4*ang+2],
                                Snap::mu[4*ang+1], Snap::mu[4*ang])), 
                  _mm256_set1_pd(Snap::hi)));
        // Y ghost cells
        __m256d *psij;
        if (stride_y_positive) {
          // reading from y-1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] -= 1;
          if (y == 0) {
            // Ghost cell array
            psij = get_avx_angle_ptr(ghosty_in_ptr, ghosty_in_offsets,
                                     ghost_point, angle_buffer_size);
          } else {
            // Local array
            psij = yflux_pencil + x * num_vec_angles;
          }
        } else {
          // reading from y+1
          Point<3> ghost_point = local_point;
          ghost_point.x[1] += 1;
          // Local coordinates here
          if (y == 0) {
            // Ghost cell array
            psij = get_avx_angle_ptr(ghosty_in_ptr, ghosty_in_offsets,
                                     ghost_point, angle_buffer_size);
          } else {
            // Local array
            psij = yflux_pencil + x * num_vec_angles;
          }
        }
        for (int ang = 0; ang < num_vec_angles; ang++)
          pc[ang] = _mm256_add_pd(pc[ang], _mm256_mul_pd( _mm256_mul_pd(psij[ang],
                  _mm256_set_pd(Snap::eta[4*ang+3], Snap::eta[4*ang+2],
                                Snap::eta[4*ang+1], Snap::eta[4*ang])),
                  _mm256_set1_pd(Snap::hj)));
        // Z ghost cells
        __m256d *psik;
        if (stride_z_positive) {
          // reading from z-1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] -= 1;
          if (z == 0) {
            // Ghost cell array
            psik = get_avx_angle_ptr(ghostz_in_ptr, ghostz_in_offsets,
                                     ghost_point, angle_buffer_size);
          } else {
            // Local array
            psik = zflux_plane + (y * x_range + x) * num_vec_angles;
          }
        } else {
          // reading from z+1
          Point<3> ghost_point = local_point;
          ghost_point.x[2] += 1;
          // Local coordinates here
          if (z == 0) {
            // Ghost cell array
            psik = get_avx_angle_ptr(ghostz_in_ptr, ghostz_in_offsets,
                                     ghost_point, angle_buffer_size);
          } else {
            // Local array
            psik = zflux_plane + (y * x_range + x) * num_vec_angles;
          }
        }
        for (int ang = 0; ang < Snap::num_angles; ang++)
          pc[ang] = _mm256_add_pd(pc[ang], _mm256_mul_pd( _mm256_mul_pd(psik[ang],
                  _mm256_set_pd(Snap::xi[4*ang+3], Snap::xi[4*ang+2],
                                Snap::xi[4*ang+1], Snap::xi[4*ang])),
                  _mm256_set1_pd(Snap::hk)));

        // See if we're doing anything time dependent
        __m256d *time_flux_in = get_avx_angle_ptr(time_flux_in_ptr,
              time_flux_in_offsets, local_point, angle_buffer_size);
        if (vdelt != 0.0) 
        {
          for (int ang = 0; ang < num_vec_angles; ang++)
            pc[ang] = _mm256_add_pd(pc[ang], _mm256_mul_pd(
                  _mm256_set1_pd(vdelt), time_flux_in[ang]));
        }
        // Multiple by the precomputed denominator inverse
        __m256d *dinv = get_avx_angle_ptr(dinv_ptr, dinv_offsets, 
                                          local_point, angle_buffer_size);
        for (int ang = 0; ang < num_vec_angles; ang++)
          pc[ang] = _mm256_mul_pd(pc[ang], dinv[ang]);

        if (Snap::flux_fixup) {
          // DO THE FIXUP
          unsigned old_negative_fluxes = 0;
          for (int ang = 0; ang < num_vec_angles; ang++)
            hv_x[ang] = _mm256_set1_pd(1.0);
          for (int ang = 0; ang < num_vec_angles; ang++)
            hv_y[ang] = _mm256_set1_pd(1.0);
          for (int ang = 0; ang < num_vec_angles; ang++)
            hv_z[ang] = _mm256_set1_pd(1.0);
          for (int ang = 0; ang < num_vec_angles; ang++)
            hv_t[ang] = _mm256_set1_pd(1.0);
          const double t_xs = *(t_xs_ptr + t_xs_offsets * local_point);
          while (true) {
            unsigned negative_fluxes = 0;
            // Figure out how many negative fluxes we have
            for (int ang = 0; ang < num_vec_angles; ang++) {
              fx_hv_x[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                    _mm256_set1_pd(2.0), pc[ang]), psii[ang]);
              __m256d ge = _mm256_cmp_pd(fx_hv_x[ang], 
                  _mm256_set1_pd(0.0), _CMP_GE_OS);
              // If not greater than or equal, set back to zero
              hv_x[ang] = _mm256_and_pd(ge, hv_x[ang]);
              // Count how many negative fluxes we had
              __m256i negatives = _mm256_andnot_si256(_mm256_castpd_si256(ge), 
                  _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1));
              negative_fluxes += _mm256_extract_epi32(negatives, 0);
              negative_fluxes += _mm256_extract_epi32(negatives, 2);
              negative_fluxes += _mm256_extract_epi32(negatives, 4);
              negative_fluxes += _mm256_extract_epi32(negatives, 6);
            }
            for (int ang = 0; ang < num_vec_angles; ang++) {
              fx_hv_y[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                    _mm256_set1_pd(2.0), pc[ang]), psij[ang]);
              __m256d ge = _mm256_cmp_pd(fx_hv_y[ang], 
                  _mm256_set1_pd(0.0), _CMP_GE_OS);
              // If not greater than or equal set back to zero
              hv_y[ang] = _mm256_and_pd(ge, hv_y[ang]);
              // Count how many negative fluxes we had
              __m256i negatives = _mm256_andnot_si256(_mm256_castpd_si256(ge), 
                  _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1));
              negative_fluxes += _mm256_extract_epi32(negatives, 0);
              negative_fluxes += _mm256_extract_epi32(negatives, 2);
              negative_fluxes += _mm256_extract_epi32(negatives, 4);
              negative_fluxes += _mm256_extract_epi32(negatives, 6);
            }
            for (int ang = 0; ang < num_vec_angles; ang++) {
              fx_hv_z[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                    _mm256_set1_pd(2.0), pc[ang]), psik[ang]);
              __m256d ge = _mm256_cmp_pd(fx_hv_z[ang], 
                  _mm256_set1_pd(0.0), _CMP_GE_OS);
              // If not greater than or equal set back to zero
              hv_z[ang] = _mm256_and_pd(ge, hv_z[ang]);
              // Count how many negative fluxes we had
              __m256i negatives = _mm256_andnot_si256(_mm256_castpd_si256(ge), 
                  _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1));
              negative_fluxes += _mm256_extract_epi32(negatives, 0);
              negative_fluxes += _mm256_extract_epi32(negatives, 2);
              negative_fluxes += _mm256_extract_epi32(negatives, 4);
              negative_fluxes += _mm256_extract_epi32(negatives, 6);
            }
            if (vdelt != 0.0) {
              for (int ang = 0; ang < num_vec_angles; ang++) {
                fx_hv_t[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                      _mm256_set1_pd(2.0), pc[ang]), time_flux_in[ang]);
                __m256d ge = _mm256_cmp_pd(fx_hv_t[ang], 
                    _mm256_set1_pd(0.0), _CMP_GE_OS);
                // If not greater than or equal, set back to zero
                hv_t[ang] = _mm256_and_pd(ge, hv_t[ang]);
                // Count how many negative fluxes we had
                __m256i negatives = _mm256_andnot_si256(_mm256_castpd_si256(ge), 
                    _mm256_set_epi32(0, 1, 0, 1, 0, 1, 0, 1));
                negative_fluxes += _mm256_extract_epi32(negatives, 0);
                negative_fluxes += _mm256_extract_epi32(negatives, 2);
                negative_fluxes += _mm256_extract_epi32(negatives, 4);
                negative_fluxes += _mm256_extract_epi32(negatives, 6);
              }
            }
            if (negative_fluxes == old_negative_fluxes)
              break;
            old_negative_fluxes = negative_fluxes;
            if (vdelt != 0.0) {
              for (int ang = 0; ang < num_vec_angles; ang++) {
                __m256d sum = _mm256_mul_pd(psii[ang], _mm256_mul_pd(
                      _mm256_set_pd(Snap::mu[4*ang+3], Snap::mu[4*ang+2],
                                    Snap::mu[4*ang+1], Snap::mu[4*ang]), 
                      _mm256_mul_pd( _mm256_set1_pd(Snap::hi), 
                        _mm256_add_pd( _mm256_set1_pd(1.0), hv_x[ang]))));
                sum = _mm256_add_pd(sum, _mm256_mul_pd(psij[ang], _mm256_mul_pd(
                        _mm256_set_pd(Snap::eta[4*ang+3], Snap::eta[4*ang+2],
                                      Snap::eta[4*ang+1], Snap::eta[4*ang]),
                        _mm256_mul_pd( _mm256_set1_pd(Snap::hj),
                          _mm256_add_pd( _mm256_set1_pd(1.0), hv_y[ang])))));
                sum = _mm256_add_pd(sum, _mm256_mul_pd(psik[ang], _mm256_mul_pd(
                        _mm256_set_pd(Snap::xi[4*ang+3], Snap::xi[4*ang+2],
                                      Snap::xi[4*ang+1], Snap::xi[4*ang]),
                        _mm256_mul_pd( _mm256_set1_pd(Snap::hk),
                          _mm256_add_pd( _mm256_set1_pd(1.0), hv_z[ang])))));
                sum = _mm256_add_pd(sum, _mm256_mul_pd(time_flux_in[ang], 
                      _mm256_mul_pd( _mm256_set1_pd(vdelt), _mm256_add_pd(
                          _mm256_set1_pd(1.0), hv_t[ang]))));
                pc[ang] = _mm256_add_pd(psi[ang], 
                    _mm256_mul_pd( _mm256_set1_pd(0.5), sum));
                __m256d den = _mm256_add_pd(_mm256_set1_pd(t_xs), 
                    _mm256_add_pd( _mm256_add_pd( _mm256_add_pd(
                      _mm256_mul_pd( _mm256_mul_pd( _mm256_set_pd(
                            Snap::mu[4*ang+3], Snap::mu[4*ang+2],
                            Snap::mu[4*ang+1], Snap::mu[4*ang]), 
                          _mm256_set1_pd(Snap::hi)), hv_x[ang]),
                      _mm256_mul_pd( _mm256_mul_pd( _mm256_set_pd(
                            Snap::eta[4*ang+3], Snap::eta[4*ang+2],
                            Snap::eta[4*ang+1], Snap::eta[4*ang]), 
                          _mm256_set1_pd(Snap::hj)), hv_y[ang])),
                      _mm256_mul_pd( _mm256_mul_pd( _mm256_set_pd(
                            Snap::xi[4*ang+3], Snap::xi[4*ang+2],
                            Snap::xi[4*ang+1], Snap::xi[4*ang]), 
                          _mm256_set1_pd(Snap::hk)), hv_z[ang])),
                      _mm256_mul_pd(_mm256_set1_pd(vdelt), hv_t[ang])));
                __m256d pc_ge = _mm256_cmp_pd(pc[ang], 
                    _mm256_set1_pd(0.0), _CMP_GE_OS);
                // Set the denominator back to zero if it is too small
                den = _mm256_and_pd(den, pc_ge);
                __m256d den_ge = _mm256_cmp_pd(den, tolr, _CMP_GE_OS);
                pc[ang] = _mm256_or_pd(
                    _mm256_and_pd(den_ge, _mm256_div_pd(pc[ang], den)),
                    _mm256_andnot_pd(den_ge, _mm256_set1_pd(0.0)));
              }
            } else {
              for (int ang = 0; ang < num_vec_angles; ang++) {
                __m256d sum = _mm256_mul_pd(psii[ang], _mm256_mul_pd(
                      _mm256_set_pd(Snap::mu[4*ang+3], Snap::mu[4*ang+2],
                                    Snap::mu[4*ang+1], Snap::mu[4*ang]), 
                      _mm256_mul_pd( _mm256_set1_pd(Snap::hi), 
                        _mm256_add_pd( _mm256_set1_pd(1.0), hv_x[ang]))));
                sum = _mm256_add_pd(sum, _mm256_mul_pd(psij[ang], _mm256_mul_pd(
                        _mm256_set_pd(Snap::eta[4*ang+3], Snap::eta[4*ang+2],
                                   Snap::eta[4*ang+1], Snap::eta[4*ang]),
                        _mm256_mul_pd( _mm256_set1_pd(Snap::hj),
                          _mm256_add_pd( _mm256_set1_pd(1.0), hv_y[ang])))));
                sum = _mm256_add_pd(sum, _mm256_mul_pd(psik[ang], _mm256_mul_pd(
                        _mm256_set_pd(Snap::xi[4*ang+3], Snap::xi[4*ang+2],
                                      Snap::xi[4*ang+1], Snap::xi[4*ang]),
                        _mm256_mul_pd( _mm256_set1_pd(Snap::hk),
                          _mm256_add_pd( _mm256_set1_pd(1.0), hv_z[ang])))));
                pc[ang] = _mm256_add_pd(psi[ang], 
                          _mm256_mul_pd( _mm256_set1_pd(0.5), sum));
                __m256d den = _mm256_add_pd(_mm256_set1_pd(t_xs), 
                    _mm256_add_pd( _mm256_add_pd( 
                      _mm256_mul_pd( _mm256_mul_pd( _mm256_set_pd(
                            Snap::mu[4*ang+3], Snap::mu[4*ang+2], 
                            Snap::mu[4*ang+1], Snap::mu[4*ang]), 
                          _mm256_set1_pd(Snap::hi)), hv_x[ang]),
                      _mm256_mul_pd( _mm256_mul_pd( _mm256_set_pd(
                            Snap::eta[4*ang+3], Snap::eta[4*ang+2],
                            Snap::eta[4*ang+1], Snap::eta[4*ang]), 
                          _mm256_set1_pd(Snap::hj)), hv_y[ang])),
                      _mm256_mul_pd( _mm256_mul_pd( _mm256_set_pd(
                            Snap::xi[4*ang+3], Snap::xi[4*ang+2],
                            Snap::xi[4*ang+1], Snap::xi[4*ang]), 
                          _mm256_set1_pd(Snap::hk)), hv_z[ang])));
                __m256d pc_ge = _mm256_cmp_pd(pc[ang], _mm256_set1_pd(0.0), _CMP_GE_OS);
                // Set the denominator back to zero if it is too small
                den = _mm256_and_pd(den, pc_ge);
                __m256d den_ge = _mm256_cmp_pd(den, tolr, _CMP_GE_OS);
                pc[ang] = _mm256_or_pd(
                    _mm256_and_pd(den_ge, _mm256_div_pd(pc[ang], den)),
                    _mm256_andnot_pd(den_ge, _mm256_set1_pd(0.0)));
              }
            }
          }
          // Fixup done so compute the update values
          for (int ang = 0; ang < num_vec_angles; ang++)
            psii[ang] = _mm256_mul_pd(fx_hv_x[ang], hv_x[ang]);
          for (int ang = 0; ang < num_vec_angles; ang++)
            psij[ang] = _mm256_mul_pd(fx_hv_y[ang], hv_y[ang]);
          for (int ang = 0; ang < num_vec_angles; ang++)
            psik[ang] = _mm256_mul_pd(fx_hv_z[ang], hv_z[ang]);
          if (vdelt != 0.0)
          {
            // Write out the outgoing temporal flux 
            __m256d *time_flux_out = 
              get_avx_angle_ptr(time_flux_out_ptr, time_flux_out_offsets,
                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm256_stream_pd((double*)(time_flux_out+ang), 
                  _mm256_mul_pd(fx_hv_t[ang], hv_t[ang]));
          }
        } else {
          // NO FIXUP
          for (int ang = 0; ang < num_vec_angles; ang++)
            psii[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                  _mm256_set1_pd(2.0), pc[ang]), psii[ang]);
          for (int ang = 0; ang < num_vec_angles; ang++)
            psij[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                  _mm256_set1_pd(2.0), pc[ang]), psij[ang]);
          for (int ang = 0; ang < num_vec_angles; ang++)
            psik[ang] = _mm256_sub_pd( _mm256_mul_pd( 
                  _mm256_set1_pd(2.0), pc[ang]), psik[ang]);
          if (vdelt != 0.0) 
          {
            // Write out the outgoing temporal flux 
            __m256d *time_flux_out = 
              get_avx_angle_ptr(time_flux_out_ptr, time_flux_out_offsets,
                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm256_stream_pd((double*)(time_flux_out+ang), 
                  _mm256_sub_pd( _mm256_mul_pd( _mm256_set1_pd(2.0), 
                      pc[ang]), time_flux_in[ang]));
          }
        }

        // Write out the ghost regions
        if (stride_x_positive) {
          // Writing to x+1
          if (x == (Snap::nx_per_chunk-1)) {
            // We write out on our own region
            __m256d *target = get_avx_angle_ptr(ghostx_out_ptr, ghostx_out_offsets,
                                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm256_stream_pd((double*)(target+ang), psii[ang]);
          } 
          // Else nothing: psii just gets caried over to next iteration
        } else {
          // Writing to x-1
          // Local coordinates here
          if (x == (Snap::nx_per_chunk-1)) {
            // Write out on our own region
            __m256d *target = get_avx_angle_ptr(ghostx_out_ptr, ghostx_out_offsets,
                                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm256_stream_pd((double*)(target+ang), psii[ang]);
          } 
          // Else nothing: psii just gets carried over to next iteration 
        }
        // Y ghost
        if (stride_y_positive) {
          // Writing to y+1
          if (y == (Snap::ny_per_chunk-1)) {
            // Write out on our own region
            __m256d *target = get_avx_angle_ptr(ghosty_out_ptr, ghosty_out_offsets,
                                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm256_stream_pd((double*)(target+ang), psij[ang]);
          } 
          // Else nothing: psij is already in place in the pencil
        } else {
          // Writing to y-1
          // Local coordinates here
          if (y == (Snap::ny_per_chunk-1)) {
            // Write out on our own region
            __m256d *target = get_avx_angle_ptr(ghosty_out_ptr, ghosty_out_offsets,
                                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm256_stream_pd((double*)(target+ang), psij[ang]);
          } 
          // Else nothing: psij already in place in the pencil
        }
        // Z ghost
        if (stride_z_positive) {
          // Writing to z+1
          if (z == (Snap::nz_per_chunk-1)) {
            __m256d *target = get_avx_angle_ptr(ghostz_out_ptr, ghostz_out_offsets,
                                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm256_stream_pd((double*)(target+ang), psik[ang]);
          } 
          // Else nothing: psik is already in place in the plane
        } else {
          // Writing to z-1
          // Local coordinates here
          if (z == (Snap::nz_per_chunk-1)) {
            // Write out on our own region
            __m256d *target = get_avx_angle_ptr(ghostz_out_ptr, ghostz_out_offsets,
                                                local_point, angle_buffer_size);
            for (int ang = 0; ang < num_vec_angles; ang++)
              _mm256_stream_pd((double*)(target+ang), psik[ang]);
          } 
          // Else nothing: psik is already in place in the plane
        }

        // Finally we apply reductions to the flux moments
        __m256d vec_total = _mm256_set1_pd(0.0);
        for (int ang = 0; ang < num_vec_angles; ang++)
          vec_total = _mm256_add_pd(vec_total, psi[ang]);
        vec_total = _mm256_hadd_pd(vec_total, vec_total);
        double total = _mm_cvtsd_f64( _mm256_extractf128_pd(
              _mm256_hadd_pd(vec_total, vec_total), 0));
        SumReduction::fold<false>(*(flux + flux_offsets * local_point), total);
        if (Snap::num_moments > 1) {
          MomentQuad quad;
          for (int l = 1; l < Snap::num_moments; l++) {
            unsigned offset = l * Snap::num_angles + 
              args->corner * Snap::num_angles * Snap::num_moments;
            vec_total = _mm256_set1_pd(0.0);
            for (int ang = 0; ang < num_vec_angles; ang++)
              vec_total = _mm256_add_pd(vec_total, _mm256_mul_pd(psi[ang],
                    _mm256_set_pd(Snap::ec[offset+4*ang+3], Snap::ec[offset+4*ang+2],
                                  Snap::ec[offset+4*ang+1], Snap::ec[offset+4*ang])));
            vec_total = _mm256_hadd_pd(vec_total, vec_total);
            quad[l] = _mm_cvtsd_f64( _mm256_extractf128_pd(
                  _mm256_hadd_pd(vec_total, vec_total), 0));
          }
          QuadReduction::fold<false>(*(fluxm + fluxm_offsets * local_point), quad);
        }
      }
    }
  }

  free(psi);
  free(pc);
  free(psii);
  free(hv_x);
  free(hv_y);
  free(hv_z);
  free(hv_t);
  free(fx_hv_x);
  free(fx_hv_y);
  free(fx_hv_z);
  free(fx_hv_t);
  free(yflux_pencil);
  free(zflux_plane);
#endif
}

//------------------------------------------------------------------------------
/*static*/ void MiniKBATask::gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  assert(false);
#endif
}

