/* Copyright 2017 NVIDIA Corporation
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

#ifndef __EXPXS_H__
#define __EXPXS_H__

#include "snap.h"
#include "legion.h"

using namespace Legion;

class ExpandCrossSection : public SnapTask<ExpandCrossSection,
                                           Snap::EXPAND_CROSS_SECTION_TASK_ID> {
public:
  ExpandCrossSection(const Snap &snap, const SnapArray &sig, 
                     const SnapArray &mat, const SnapArray &xs, 
                     int group_start, int group_stop);
public:
  const int group_start;
  const int group_stop;
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void fast_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class ExpandScatteringCrossSection : public SnapTask<ExpandScatteringCrossSection,
                                    Snap::EXPAND_SCATTERING_CROSS_SECTION_TASK_ID> {
public:
  ExpandScatteringCrossSection(const Snap &snap, const SnapArray &slgg,
                               const SnapArray &mat, const SnapArray &s_xs, 
                               int group_start, int group_stop);
public:
  const int group_start;
  const int group_stop;
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void fast_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

class CalculateGeometryParam : public SnapTask<CalculateGeometryParam,
                                               Snap::CALCULATE_GEOMETRY_PARAM_TASK_ID> {
public:
  CalculateGeometryParam(const Snap &snap, const SnapArray &t_xs, 
                         const SnapArray &vdelt, const SnapArray &dinv, 
                         int group_start, int group_stop);
public:
  const int group_start;
  const int group_stop;
public:
  static void preregister_cpu_variants(void);
  static void preregister_gpu_variants(void);
public:
  static void cpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void fast_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
  static void gpu_implementation(const Task *task,
      const std::vector<PhysicalRegion> &regions, Context ctx, Runtime *runtime);
};

#endif // __EXPXS_H__

