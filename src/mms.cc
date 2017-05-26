/* Copyright 2017 NVIDIA Corporation
 *
 * The U.S. Department of Energy funded the development of this software 
 * under subcontract B609478 with Lawrence Livermore National Security, LLC
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
#include "mms.h"

#include <cmath>

#ifndef MAX
#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x,y) (((x) < (y)) ? (x) : (y))
#endif

extern Legion::Logger log_snap;

template<bool COS>
void mms_trigint(const int lc, const double d, const double del, 
                 const double *cb, double *fn);

template<>
void mms_trigint<true/*COS*/>(const int lc, const double d, const double del, 
                              const double *cb, double *fn)
{
  memset(fn, 0, lc * sizeof(double));
  const double denom = d * del;
  for (int i = 0; i < lc; i++) {
    fn[i] = cos(d * cb[i]) - cos(d * cb[i+1]);
    fn[i] /= denom;
  }
}

template<>
void mms_trigint<false/*COS*/>(const int lc, const double d, const double del, 
                               const double *cb, double *fn)
{
  memset(fn, 0, lc * sizeof(double));
  for (int i = 0; i < lc; i++) {
    fn[i] = sin(d * cb[i+1]) - sin(d * cb[i]);
    fn[i] /= del;
  }
}

//------------------------------------------------------------------------------
MMSInitFlux::MMSInitFlux(const Snap &snap, const SnapArray<3> &ref_flux, 
                         const SnapArray<3> &ref_fluxm)
  : SnapTask<MMSInitFlux, Snap::MMS_INIT_FLUX_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
  ref_flux.add_projection_requirement(READ_WRITE, *this);
  ref_fluxm.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void MMSInitFlux::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  TaskLayoutConstraintSet layout_constraints;
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MMSInitFlux::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running MMS Init Flux");

  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));

  double *tx = (double*)malloc(Snap::nx_per_chunk * sizeof(double));
  double *ty = (double*)malloc(Snap::ny_per_chunk * sizeof(double));
  double *tz = (double*)malloc(Snap::nz_per_chunk * sizeof(double));
  double *ib = (double*)malloc((Snap::nx_per_chunk+1) * sizeof(double));
  double *jb = (double*)malloc((Snap::ny_per_chunk+1) * sizeof(double));
  double *kb = (double*)malloc((Snap::nz_per_chunk+1) * sizeof(double));

  const double a = PI / Snap::lx;
  const double b = PI / Snap::ly;
  const double c = PI / Snap::lz;
  const double dx = Snap::lx / double(Snap::nx);
  const double dy = Snap::ly / double(Snap::ny);
  const double dz = Snap::lz / double(Snap::nz);
  ib[0] = dom.bounds.lo[0] * dx;
  for (int i = 1; i <= Snap::nx_per_chunk; i++)
    ib[i] = ib[i-1] + dx;
  jb[0] = dom.bounds.lo[1] * dy;
  for (int j = 1; j <= Snap::ny_per_chunk; j++)
    jb[j] = jb[j-1] + dx;
  kb[0] = dom.bounds.lo[2] * dz;
  for (int k = 1; k <= Snap::nz_per_chunk; k++)
    kb[k] = kb[k-1] + dx;

  mms_trigint<true/*COS*/>(Snap::nx_per_chunk, a, dx, ib, tx);
  if (Snap::num_dims > 1) {
    mms_trigint<true/*COS*/>(Snap::ny_per_chunk, b, dy, jb, ty);
    if (Snap::num_dims > 2) {
      mms_trigint<true/*COS*/>(Snap::nz_per_chunk, c, dz, kb, tz);
    } else {
      for (int k = 0; k < Snap::nz_per_chunk; k++)
        tz[k] = 1.0;
    }
  } else {
    for (int j = 0; j < Snap::ny_per_chunk; j++)
      ty[j] = 1.0;
    for (int k = 0; k < Snap::nz_per_chunk; k++)
      tz[k] = 1.0;
  }

  unsigned g = 1;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, g++)
  {
    Accessor<double,3> fa_flux(regions[0], *it);
    for (DomainIterator<3> itr(dom); itr(); itr++) {    
      const Point<3> p = itr;
      int i = p[0] - dom.bounds.lo[0];
      assert(i < Snap::nx_per_chunk);
      int j = p[1] - dom.bounds.lo[1];
      assert(j < Snap::ny_per_chunk);
      int k = p[2] - dom.bounds.lo[2];
      assert(k < Snap::nz_per_chunk);
      double value = (double(g) * tx[i] * ty[j] * tz[k]);
      fa_flux[itr] = value;
    }
  }

  double p[3] = { 0.0, 0.0, 0.0 };
  for (int c = 0; c < Snap::num_corners; c++) {
    for (int l = 1; l < Snap::num_moments; l++) {
      unsigned offset = (l + c * Snap::num_moments) * Snap::num_angles;
      for (int ang = 0; ang < Snap::num_angles; ang++)
        p[l-1] += Snap::w[ang] * Snap::ec[offset + ang];
    }
  }

  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, g++)
  {
    Accessor<double,3> fa_flux(regions[0], *it);
    Accessor<MomentTriple,3> fa_fluxm(regions[1], *it);
    for (DomainIterator<3> itr(dom); itr(); itr++) {
      double flux = fa_flux[itr];
      MomentTriple result;
      for (int l = 0; l < 3; l++)
        result[l] = p[l] * flux;
      fa_fluxm[itr] = result;
    }
  }

  free(tx);
  free(ty);
  free(tz);
#endif
}

//------------------------------------------------------------------------------
MMSInitSource::MMSInitSource(const Snap &snap, const SnapArray<3> &ref_flux,
                         const SnapArray<3> &ref_fluxm, const SnapArray<3> &mat,
                         const SnapArray<1> &sigt, const SnapArray<2> &slgg,
                         const SnapArray<3> &qim,int c)
  : SnapTask<MMSInitSource, Snap::MMS_INIT_SOURCE_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED), corner(c)
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&corner, sizeof(corner));
  ref_flux.add_projection_requirement(READ_ONLY, *this);
  ref_fluxm.add_projection_requirement(READ_ONLY, *this);
  mat.add_projection_requirement(READ_ONLY, *this);
  sigt.add_region_requirement(READ_ONLY, *this); 
  slgg.add_region_requirement(READ_ONLY, *this);
  qim.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void MMSInitSource::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  TaskLayoutConstraintSet layout_constraints;
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MMSInitSource::cpu_implementation(const Task *task,
    const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running MMS Init Source");

  assert(task->arglen == sizeof(int));
  const int corner = *((int*)task->args);
  const double is = (0x1 & corner) ? 1.0 : -1.0;
  const double js = (0x2 & corner) ? 1.0 : -1.0;
  const double ks = (0x4 & corner) ? 1.0 : -1.0;

  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));

  const double a = PI / Snap::lx;
  const double b = PI / Snap::ly;
  const double c = PI / Snap::lz;
  const double dx = Snap::lx / double(Snap::nx);
  const double dy = Snap::ly / double(Snap::ny);
  const double dz = Snap::lz / double(Snap::nz);

  double *ib = (double*)malloc((Snap::nx_per_chunk+1) * sizeof(double));
  double *jb = (double*)malloc((Snap::ny_per_chunk+1) * sizeof(double));
  double *kb = (double*)malloc((Snap::nz_per_chunk+1) * sizeof(double));

  ib[0] = dom.bounds.lo[0] * dx;
  for (int i = 1; i <= Snap::nx_per_chunk; i++)
    ib[i] = ib[i-1] + dx;
  jb[0] = dom.bounds.lo[1] * dy;
  for (int j = 1; j <= Snap::ny_per_chunk; j++)
    jb[j] = jb[j-1] + dx;
  kb[0] = dom.bounds.lo[2] * dz;
  for (int k = 1; k <= Snap::nz_per_chunk; k++)
    kb[k] = kb[k-1] + dx; 

  double *cx = (double*)malloc(Snap::nx_per_chunk * sizeof(double));
  double *sx = (double*)malloc(Snap::nx_per_chunk * sizeof(double));
  double *cy = (double*)malloc(Snap::ny_per_chunk * sizeof(double));
  double *sy = (double*)malloc(Snap::ny_per_chunk * sizeof(double));
  double *cz = (double*)malloc(Snap::nz_per_chunk * sizeof(double));
  double *sz = (double*)malloc(Snap::nz_per_chunk * sizeof(double));
  
  mms_trigint<true/*COS*/>(Snap::nx_per_chunk, a, dx, ib, cx);
  mms_trigint<false/*SIN*/>(Snap::nx_per_chunk, a, dx, ib, sx);
  if (Snap::num_dims > 1) {
    mms_trigint<true/*COS*/>(Snap::ny_per_chunk, b, dy, jb, cy);
    mms_trigint<false/*SIN*/>(Snap::ny_per_chunk, b, dy, jb, sy);
    if (Snap::num_dims > 2) {
      mms_trigint<true/*COS*/>(Snap::nz_per_chunk, c, dz, kb, cz);
      mms_trigint<false/*SIN*/>(Snap::nz_per_chunk, c, dz, kb, sz);
    } else {
      for (int k = 0; k < Snap::nz_per_chunk; k++)
        cz[k] = 1.0;
      for (int k = 0; k < Snap::nz_per_chunk; k++)
        sz[k] = 0.0;
    }
  } else {
    for (int j = 0; j < Snap::ny_per_chunk; j++)
      cy[j] = 1.0;
    for (int j = 0; j < Snap::ny_per_chunk; j++)
      sy[j] = 0.0;
    for (int k = 0; k < Snap::nz_per_chunk; k++)
      cz[k] = 1.0;
    for (int k = 0; k < Snap::nz_per_chunk; k++)
      sz[k] = 0.0;
  }

  const size_t angle_buffer_size = Snap::num_angles * sizeof(double);
  double *angle_buffer = (double*)malloc(angle_buffer_size);

  Accessor<int,3> fa_mat(regions[2], Snap::FID_SINGLE);

  unsigned g_idx = 0;
  std::vector<Accessor<double,3> > 
    fa_fluxes(task->regions[0].privilege_fields.size());
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, g_idx++)
    fa_fluxes[g_idx] = Accessor<double,3>(regions[0], *it);
  g_idx = 0;
  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++, g_idx++)
  {
    Accessor<double,3> &fa_flux = fa_fluxes[g_idx]; 
    Accessor<MomentTriple,3> fa_fluxm(regions[1], *it);
    Accessor<double,1> fa_sigt(regions[3], *it);
    Accessor<MomentQuad,2> fa_slgg(regions[4], *it);
    Accessor<double,3> fa_qim(regions[5], *it);

    for (DomainIterator<3> itr(dom); itr(); itr++) {
      Point<3> p = itr;
      const int i = p[0] - dom.bounds.lo[0];
      const int j = p[1] - dom.bounds.lo[1];
      const int k = p[2] - dom.bounds.lo[2];

      const int mat = fa_mat[itr];
      const double sigt = fa_sigt[mat];
      const double ref_flux = fa_flux[itr];
      const double flux_update = sigt * ref_flux;

      const MomentTriple ref_fluxm = fa_fluxm[itr];

      memcpy(angle_buffer, fa_qim.ptr(itr), angle_buffer_size);
      for (int ang = 0; ang < Snap::num_angles; ang++) {
        angle_buffer[ang] += (double(g_idx+1) * is * Snap::mu[ang] * sx[i] * cy[j] * cz[k]);
        angle_buffer[ang] += flux_update;
        if (Snap::num_dims > 1)
          angle_buffer[ang] += (double(g_idx+1) * js * Snap::eta[ang] * cx[i] * sy[j] * cz[k]);
        if (Snap::num_dims > 2)
          angle_buffer[ang] += (double(g_idx+1) * ks * Snap::xi[ang] * cx[i] * cy[j] * sz[k]);
        unsigned gp_idx = 0;
        for (std::set<FieldID>::const_iterator gp = 
              task->regions[0].privilege_fields.begin(); gp !=
              task->regions[0].privilege_fields.end(); gp++, gp_idx++) {
          Accessor<double,3> &fa_flux_gp = fa_fluxes[gp_idx];
          const double flux_gp = fa_flux_gp[itr];
          const MomentQuad quad = fa_slgg[mat][gp_idx];
          angle_buffer[ang] -= (quad[0] * flux_gp);
          int lm = 1;
          for (int l = 1; l < Snap::num_moments; l++) {
            for (int ll = 0; ll < Snap::lma[l]; ll++) {
              const int offset = corner * Snap::num_angles * Snap::num_moments + 
                                  lm * Snap::num_angles + ang;
              assert((lm-1) < 3);
              angle_buffer[ang] -= (Snap::ec[offset] * quad[l] * ref_fluxm[lm-1]);
              lm = lm + 1;
            }
          }
        }
      }
      memcpy(fa_qim.ptr(itr), angle_buffer, angle_buffer_size);
    }
  }

  free(ib);
  free(jb);
  free(kb);
  free(cx);
  free(sx);
  free(cy);
  free(sy);
  free(cz);
  free(sz);
  free(angle_buffer);
#endif
}

//------------------------------------------------------------------------------
MMSInitTimeDependent::MMSInitTimeDependent(const Snap &snap, 
                                           const SnapArray<1> &v,
                                           const SnapArray<3> &ref_flux, 
                                           const SnapArray<3> &qi)
  : SnapTask<MMSInitTimeDependent, Snap::MMS_INIT_TIME_DEPENDENT_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
  v.add_region_requirement(READ_ONLY, *this);
  ref_flux.add_projection_requirement(READ_WRITE, *this);
  qi.add_projection_requirement(WRITE_DISCARD, *this);
}

//------------------------------------------------------------------------------
/*static*/ void MMSInitTimeDependent::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  TaskLayoutConstraintSet layout_constraints;
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MMSInitTimeDependent::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running MMS Init Time Dependent");

  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[1].region.get_index_space()));

  const double t_scale = Snap::total_sim_time - 0.5 * Snap::dt;

  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    Accessor<double,1> fa_v(regions[0], *it);
    Accessor<double,3> fa_flux(regions[1], *it);
    Accessor<double,3> fa_qi(regions[2], *it);

    const double vg = fa_v[0];

    for (DomainIterator<3> itr(dom); itr(); itr++) {
      const double ref_flux = fa_flux[itr];
      // compute the source
      fa_qi[itr] = ref_flux / vg;
      // Then scale the flux 
      fa_flux[itr] = ref_flux * t_scale;
    }
  }
#endif
}

//------------------------------------------------------------------------------
MMSScale::MMSScale(const Snap &snap, const SnapArray<3> &qim, double f)
  : SnapTask<MMSScale, Snap::MMS_SCALE_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED), scale_factor(f)
//------------------------------------------------------------------------------
{
  global_arg = TaskArgument(&scale_factor, sizeof(scale_factor));
  qim.add_projection_requirement(READ_WRITE, *this);
}

//------------------------------------------------------------------------------
/*static*/ void MMSScale::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  TaskLayoutConstraintSet layout_constraints;
  register_cpu_variant<cpu_implementation>(execution_constraints,
                                           layout_constraints,
                                           true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ void MMSScale::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
#ifndef NO_COMPUTE
  log_snap.info("Running MMS Scale");

  assert(task->arglen == sizeof(double));
  const double scale_factor = *((double*)task->args);

  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));

  const size_t angle_buffer_size = Snap::num_angles * sizeof(double);
  double *angle_buffer = (double*)malloc(angle_buffer_size);

  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    Accessor<double,3> fa_qim(regions[0], *it);
    for (DomainIterator<3> itr(dom); itr(); itr++)
    {
      memcpy(angle_buffer, fa_qim.ptr(itr), angle_buffer_size);
      for (int ang = 0; ang < Snap::num_angles; ang++)
        angle_buffer[ang] *= scale_factor;
      memcpy(fa_qim.ptr(itr), angle_buffer, angle_buffer_size);
    }
  }
  free(angle_buffer);
#endif
}

//------------------------------------------------------------------------------
MMSCompare::MMSCompare(const Snap &snap, const SnapArray<3> &flux, 
                       const SnapArray<3> &ref_flux)
  : SnapTask<MMSCompare, Snap::MMS_COMPARE_TASK_ID>(
      snap, snap.get_launch_bounds(), Predicate::TRUE_PRED)
//------------------------------------------------------------------------------
{
  flux.add_projection_requirement(READ_ONLY, *this);
  ref_flux.add_projection_requirement(READ_ONLY, *this);
}

//------------------------------------------------------------------------------
/*static*/ void MMSCompare::preregister_cpu_variants(void)
//------------------------------------------------------------------------------
{
  ExecutionConstraintSet execution_constraints;
  TaskLayoutConstraintSet layout_constraints;
  register_cpu_variant<MomentTriple,cpu_implementation>(execution_constraints,
                                                        layout_constraints,
                                                        true/*leaf*/);
}

//------------------------------------------------------------------------------
/*static*/ MomentTriple MMSCompare::cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime)
//------------------------------------------------------------------------------
{
  MomentTriple result;
#ifndef NO_COMPUTE
  log_snap.info("Running MMS Compare");

  double min = INFINITY;
  double max = -INFINITY;
  double sum = 0.0;

  Domain<3> dom = runtime->get_index_space_domain(ctx, 
          IndexSpace<3>(task->regions[0].region.get_index_space()));

  const double tolr = 1.0e-12;

  for (std::set<FieldID>::const_iterator it = 
        task->regions[0].privilege_fields.begin(); it !=
        task->regions[0].privilege_fields.end(); it++)
  {
    Accessor<double,3> fa_flux(regions[0], *it);
    Accessor<double,3> fa_ref_flux(regions[1], *it);
    for (DomainIterator<3> itr(dom); itr(); itr++) {
      const double flux = fa_flux[itr];
      double ref_flux = fa_ref_flux[itr];

      double df = 1.0;
      if (ref_flux < tolr) {
        ref_flux = 1.0;
        df = 0.0;
      }
      //printf("(%lld,%lld,%lld): %.8g %.8g\n", 
      //        itr.p[0], itr.p[1], itr.p[2], flux, ref_flux);
      df = fabs(flux / ref_flux - df);
      if (df > max)
        max = df;
      if (df < min)
        min = df;
      sum += df;
    }
  }
  result[0] = max;
  result[1] = min;
  result[2] = sum;
#endif
  return result;
}

const MomentTriple MMSReduction::identity = MomentTriple(-INFINITY, INFINITY, 0.0);

//------------------------------------------------------------------------------
template<>
void MMSReduction::apply<true>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  if (rhs[0] > lhs[0])
    lhs[0] = rhs[0];
  if (rhs[1] < lhs[1])
    lhs[1] = rhs[1];
  lhs[2] += rhs[2];
}

//------------------------------------------------------------------------------
template<>
void MMSReduction::apply<false>(LHS &lhs, RHS rhs)
//------------------------------------------------------------------------------
{
  union { long as_int; double as_float; } oldval, newval;

  long *target = (long *)&lhs[0];
  do {
    oldval.as_int = *target;
    newval.as_float = MAX(oldval.as_float, rhs[0]);
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));

  target = (long *)&lhs[1];
  do {
    oldval.as_int = *target;
    newval.as_float = MIN(oldval.as_float, rhs[1]);
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));

  target = (long *)&lhs[2];
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs[2];
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

//------------------------------------------------------------------------------
template<>
void MMSReduction::fold<true>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  if (rhs2[0] > rhs1[0])
    rhs1[0] = rhs2[0];
  if (rhs2[1] < rhs1[1])
    rhs1[1] = rhs2[1];
  rhs1[2] += rhs2[2];
}

//------------------------------------------------------------------------------
template<>
void MMSReduction::fold<false>(RHS &rhs1, RHS rhs2)
//------------------------------------------------------------------------------
{
  union { long as_int; double as_float; } oldval, newval;

  long *target = (long *)&rhs1[0];
  do {
    oldval.as_int = *target;
    newval.as_float = MAX(oldval.as_float, rhs2[0]);
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));

  target = (long *)&rhs1[1];
  do {
    oldval.as_int = *target;
    newval.as_float = MIN(oldval.as_float, rhs2[1]);
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));

  target = (long *)&rhs1[2];
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs2[2];
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

