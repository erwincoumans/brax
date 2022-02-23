# Copyright 2021 The Brax Authors.
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

# pylint:disable=g-multiple-import
"""Functionality for brax bodies."""

#from flax import struct
#import jax
#import jax.numpy as jnp
import numpy as jnp

from brax.physics import config_pb2
from brax.physics import math
from brax.physics.base import P, QP, euler_to_quat, vec_to_np

#@struct.dataclass
class Body(object):
  """A body is a solid, non-deformable object with some mass and shape.

  Attributes:
    idx: Index of where body is found in the system.
    inertia: (3, 3) Inverse Inertia matrix represented in body frame.
    mass: Mass of the body.
    active: whether the body is effected by physics calculations
  """
  idx: jnp.ndarray
  inertia: jnp.ndarray
  mass: jnp.ndarray
  active: jnp.ndarray

  def __init__(self, idx, inertia, mass, active):
    self.idx = idx
    self.inertia = inertia
    self.mass = mass
    self.active = active
    
  @classmethod
  def from_config(cls, config: config_pb2.Config) -> 'Body':
    """Returns Body from a brax config."""
    #bodies = []
    b = Body([],[],[],[])
    
    for idx, body in enumerate(config.bodies):
      frozen = jnp.sum(
          vec_to_np(body.frozen.position) + vec_to_np(body.frozen.rotation))
      #bodies.append(
      #    cls(
      #        idx=jnp.array(idx),
      #        inertia=jnp.linalg.inv(jnp.diag(vec_to_np(body.inertia))),
      #        mass=jnp.array(body.mass),
      #        active=jnp.array(jnp.sum(frozen) != 6),
      #    ))
      b.idx.append(idx)
      b.inertia.append(jnp.linalg.inv(jnp.diag(vec_to_np(body.inertia))))
      b.mass.append(body.mass)
      b.active.append(jnp.sum(frozen) != 6)
          
    #print("bodies=",bodies)
    #print("Body from_config=",b.idx)
    return b
    #return jax.tree_multimap((lambda *args: jnp.stack(args)), *bodies)
    #return map((lambda *args: jnp.stack(args)), *bodies)

  def impulse(self, qp: QP, impulse: jnp.ndarray, pos: jnp.ndarray, index: int) -> P:
    """Calculates updates to state information based on an impulse.

    Args:
      qp: State data of the system
      impulse: Impulse vector
      pos: Location of the impulse relative to the body's center of mass

    Returns:
      dP: An impulse to apply to this body
    """
    dvel = impulse / self.mass[index]
    dang = jnp.matmul(self.inertia[index], jnp.cross(pos - qp.pos[index], impulse))
    return P(vel=dvel, ang=dang)

def min_z(qp: QP, body: config_pb2.Body) -> float:
  """Returns the lowest z of all the colliders in a body."""
  result = float('inf')

  for col in body.colliders:
    if col.HasField('sphere'):
      sphere_pos = math.rotate(vec_to_np(col.position), qp.rot)
      min_z = qp.pos[2] + sphere_pos[2] - col.sphere.radius
      result = jnp.min(jnp.array([result, min_z]))
    elif col.HasField('capsule'):
      axis = math.rotate(jnp.array([0., 0., 1.]), euler_to_quat(col.rotation))
      length = col.capsule.length / 2 - col.capsule.radius
      for end in (-1, 1):
        sphere_pos = vec_to_np(col.position) + end * axis * length
        sphere_pos = math.rotate(sphere_pos, qp.rot)
        min_z = qp.pos[2] + sphere_pos[2] - col.capsule.radius
        result = jnp.min(jnp.array([result, min_z]))
    elif col.HasField('box'):
      corners = [(i % 2 * 2 - 1, 2 * (i // 4) - 1, i // 2 % 2 * 2 - 1)
                 for i in range(8)]
      corners = jnp.array(corners, dtype=jnp.float32)
      for corner in corners:
        corner = corner * vec_to_np(col.box.halfsize)
        corner = math.rotate(corner, euler_to_quat(col.rotation))
        corner = corner + vec_to_np(col.position)
        corner = math.rotate(corner, qp.rot) + qp.pos
        result = jnp.min(jnp.array([result, corner[2]]))
    else:
      # ignore planes and other stuff
      result = jnp.min(jnp.array([result, 0.0]))

  return result
  
  
  
def take_bodies(objects, i: jnp.ndarray, axis=0):
  """Returns objects sliced by i."""
  b = Body([],[],[],[])
  #print("dir objects=",dir(objects))
  #print("objects=",objects)
  for idx in i:
    if idx in objects.idx:
      i_ = objects.idx.index(idx)
      b.idx.append(objects.idx[i_])
      b.inertia.append(objects.inertia[i_])
      b.mass.append(objects.mass[i_])
      b.active.append(objects.active[i_])
  return b  