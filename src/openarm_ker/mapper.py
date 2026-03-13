# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Joint mapper."""

import yaml
import numpy as np


class Mapper:
    """Joint mapper."""

    def __init__(
        self,
        mappingyaml_path: str,
        leader_joint_names: list[str],
        mapping_key: str,
    ):
        """Initialize joint mapper."""
        # Load YAML configuration
        with open(mappingyaml_path) as f:
            full_config = yaml.safe_load(f)

        if mapping_key not in full_config:
            raise ValueError(f"Section '{mapping_key}' not found in YAML.")

        mapping_list = full_config[mapping_key]
        self.num_joints = len(mapping_list)

        # Allocate NumPy arrays
        self.indices = np.zeros(self.num_joints, dtype=int)
        self.scales = np.ones(self.num_joints, dtype=float)
        self.offsets = np.zeros(self.num_joints, dtype=float)

        # Allocate Limit arrays
        self.limits_min = np.full(self.num_joints, -100.0, dtype=float)
        self.limits_max = np.full(self.num_joints, 100.0, dtype=float)

        leader_name_to_idx = {name: i for i, name in enumerate(leader_joint_names)}
        self.follower_names = []

        # --- Process Mappings ---
        for i, m in enumerate(mapping_list):
            l_name = m["leader"]
            f_name = m["follower"]

            if l_name not in leader_name_to_idx:
                raise ValueError(f"Leader joint '{l_name}' not found.")

            self.indices[i] = leader_name_to_idx[l_name]
            self.scales[i] = m.get("sign", 1.0) * m.get("scale", 1.0)
            self.offsets[i] = m.get("offset", 0.0)
            self.follower_names.append(f_name)

            self.open_range = m.get("open_range")
            self.leader_range = m.get("leader_range")

            mech_limit = m.get("mech_limits")

            # 0.25 radian margin. If target exceeds this, follower will be inverted
            if mech_limit and isinstance(mech_limit, list) and len(mech_limit) == 2:
                self.limits_min[i] = mech_limit[0] - 0.35
                self.limits_max[i] = mech_limit[1] + 0.35
            else:
                raise ValueError(f"Invalid mech_limits for joint '{f_name}'.")

        self.TWO_PI = 2 * np.pi

    def __map_range(self, in_min, in_max, out_min, out_max, val):
        """Map value from one range to another."""
        return (val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min

    def map(self, leader_position: np.ndarray) -> np.ndarray:
        """Execute mapping.

        Logic:
        1. Apply linear transform FIRST (y = x * scale - offset).
        2. Then check if the RESULT is within limits.
        3. Wrap output values (add/sub 2pi) if they are outside limits.
        """
        # 1. Linear Transformation (Calculate Follower Command directly)
        # target_vals = leader_position[self.indices] * self.scales - self.offsets
        target_vals = (leader_position[self.indices] - self.offsets) * self.scales

        follower_gripper_pos = self.__map_range(
            in_max=self.leader_range[1],
            in_min=self.leader_range[0],
            out_max=self.open_range[1],
            out_min=self.open_range[0],
            val=target_vals[-1],
        )
        target_vals[-1] = follower_gripper_pos

        # 2. Output Wrapping (Check result against limits) to joint1 ~ joint7
        for i in range(self.num_joints - 1):
            if target_vals[i] < self.limits_min[i]:
                target_vals[i] += self.TWO_PI
            elif target_vals[i] > self.limits_max[i]:
                target_vals[i] -= self.TWO_PI

        return target_vals
