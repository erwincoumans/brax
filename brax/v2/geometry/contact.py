# Copyright 2022 The Brax Authors.
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
"""Calculations for generating contacts."""

from typing import Iterator, Optional, Tuple

from brax.v2 import math
from brax.v2.base import (
    Capsule,
    Contact,
    Convex,
    Geometry,
    Mesh,
    Plane,
    Sphere,
    System,
    Transform,
)
from brax.v2.geometry import math as geom_math
from brax.v2.geometry import mesh as geom_mesh
import jax
from jax import numpy as jp


def _combine(
    geom_a: Geometry, geom_b: Geometry
) -> Tuple[float, float, Tuple[int, int]]:
  # default is to take maximum, but can override
  friction = jp.maximum(geom_a.friction, geom_b.friction)
  elasticity = jp.maximum(geom_a.elasticity, geom_b.elasticity)
  link_idx = (
      geom_a.link_idx,
      geom_b.link_idx if geom_b.link_idx is not None else -1,
  )
  return friction, elasticity, link_idx  # pytype: disable=bad-return-type  # jax-ndarray


def _sphere_plane(sphere: Sphere, plane: Plane) -> Contact:
  """Calculates one contact between a sphere and a plane."""
  n = math.rotate(jp.array([0.0, 0.0, 1.0]), plane.transform.rot)
  t = jp.dot(sphere.transform.pos - plane.transform.pos, n)
  penetration = sphere.radius - t
  # halfway between contact points on sphere and on plane
  pos = sphere.transform.pos - n * (sphere.radius - 0.5 * penetration)
  c = Contact(pos, n, penetration, *_combine(sphere, plane))  # pytype: disable=wrong-arg-types  # jax-ndarray
  # add a batch dimension of size 1
  return jax.tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _sphere_sphere(s_a: Sphere, s_b: Sphere) -> Contact:
  """Calculates one contact between two spheres."""
  n, dist = math.normalize(s_a.transform.pos - s_b.transform.pos)
  penetration = s_a.radius + s_b.radius - dist
  s_a_pos = s_a.transform.pos - n * s_a.radius
  s_b_pos = s_b.transform.pos + n * s_b.radius
  pos = (s_a_pos + s_b_pos) * 0.5
  c = Contact(pos, n, penetration, *_combine(s_a, s_b))  # pytype: disable=wrong-arg-types  # jax-ndarray
  # add a batch dimension of size 1
  return jax.tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _sphere_capsule(sphere: Sphere, capsule: Capsule) -> Contact:
  """Calculates one contact between a sphere and a capsule."""
  segment = jp.array([0.0, 0.0, capsule.length * 0.5])
  segment = math.rotate(segment, capsule.transform.rot)
  pt = geom_math.closest_segment_point(
      capsule.transform.pos - segment,
      capsule.transform.pos + segment,
      sphere.transform.pos,
  )
  n, dist = math.normalize(sphere.transform.pos - pt)
  penetration = sphere.radius + capsule.radius - dist

  sphere_pos = sphere.transform.pos - n * sphere.radius
  cap_pos = pt + n * capsule.radius
  pos = (sphere_pos + cap_pos) * 0.5

  c = Contact(pos, n, penetration, *_combine(sphere, capsule))  # pytype: disable=wrong-arg-types  # jax-ndarray
  # add a batch dimension of size 1
  return jax.tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _sphere_convex(sphere: Sphere, convex: Convex) -> Contact:
  """Calculates contacts between a sphere and a convex object."""
  # Get convex transformed normals, faces, and vertices.
  normals = geom_mesh.get_face_norm(convex.vert, convex.face)
  faces = jp.take(convex.vert, convex.face, axis=0)

  @jax.vmap
  def transform_faces(face, normal):
    face = convex.transform.pos + jax.vmap(math.rotate, in_axes=[0, None])(
        face, convex.transform.rot
    )
    normal = math.rotate(normal, convex.transform.rot)
    return face, normal

  faces, normals = transform_faces(faces, normals)

  # Get support from face normals.
  @jax.vmap
  def get_support(faces, normal):
    sphere_pos = sphere.transform.pos - normal * sphere.radius
    return jp.dot(sphere_pos - faces[0], normal)

  support = get_support(faces, normals)

  # Pick the face with minimal penetration as long as it has support.
  support = jp.where(support >= 0, -1e12, support)
  best_idx = support.argmax()
  face = faces[best_idx]
  normal = normals[best_idx]

  # Get closest point between the polygon face and the sphere center point.
  # Project the sphere center point onto poly plane. If it's inside polygon
  # edge normals, then we're done.
  pt = geom_math.project_pt_onto_plane(sphere.transform.pos, face[0], normal)
  edge_p0 = jp.roll(face, 1, axis=0)
  edge_p1 = face
  edge_normals = jax.vmap(jp.cross, in_axes=[0, None])(
      edge_p1 - edge_p0,
      normal,
  )
  edge_dist = jax.vmap(
      lambda plane_pt, plane_norm: (pt - plane_pt).dot(plane_norm)
  )(edge_p0, edge_normals)
  inside = jp.all(edge_dist <= 0)  # lte to handle degenerate edges

  # If the point is outside edge normals, project onto the closest edge plane
  # that the point is in front of.
  degenerate_edge = jp.all(edge_normals == 0, axis=1)
  behind = edge_dist < 0.0
  edge_dist = jp.where(degenerate_edge | behind, 1e12, edge_dist)
  idx = edge_dist.argmin()
  edge_pt = geom_math.closest_segment_point(edge_p0[idx], edge_p1[idx], pt)

  pt = jp.where(inside, pt, edge_pt)

  # Get the normal, penetration, and contact position.
  n, d = math.normalize(sphere.transform.pos - pt)
  spt = sphere.transform.pos - n * sphere.radius
  penetration = sphere.radius - d
  pos = (pt + spt) * 0.5

  c = Contact(pos, n, penetration, *_combine(sphere, convex))  # pytype: disable=wrong-arg-types  # jax-ndarray
  return jax.tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _sphere_mesh(sphere: Sphere, mesh: Mesh) -> Contact:
  """Calculates contacts between a sphere and a mesh."""

  @jax.vmap
  def sphere_face(face):
    pt = mesh.transform.pos + jax.vmap(math.rotate, in_axes=[0, None])(
        face, mesh.transform.rot
    )
    p0, p1, p2 = pt[0, :], pt[1, :], pt[2, :]

    tri_p = geom_math.closest_triangle_point(p0, p1, p2, sphere.transform.pos)
    n = sphere.transform.pos - tri_p
    n, dist = math.normalize(n)
    penetration = sphere.radius - dist
    sph_p = sphere.transform.pos - n * sphere.radius
    pos = (tri_p + sph_p) * 0.5
    return Contact(pos, n, penetration, *_combine(sphere, mesh))  # pytype: disable=wrong-arg-types  # jax-ndarray

  return sphere_face(jp.take(mesh.vert, mesh.face, axis=0))


def _capsule_plane(capsule: Capsule, plane: Plane) -> Contact:
  """Calculates two contacts between a capsule and a plane."""
  segment = jp.array([0.0, 0.0, capsule.length * 0.5])
  segment = math.rotate(segment, capsule.transform.rot)

  results = []
  for off in [segment, -segment]:
    sphere = Sphere(
        link_idx=capsule.link_idx,
        transform=Transform.create(pos=capsule.transform.pos + off),
        friction=capsule.friction,
        elasticity=capsule.elasticity,
        radius=capsule.radius,
    )
    results.append(_sphere_plane(sphere, plane))

  return jax.tree_map(lambda *x: jp.concatenate(x), *results)


def _capsule_capsule(cap_a: Capsule, cap_b: Capsule) -> Contact:
  """Calculates one contact between two capsules."""
  seg_a = jp.array([0.0, 0.0, cap_a.length * 0.5])
  seg_a = math.rotate(seg_a, cap_a.transform.rot)
  seg_b = jp.array([0.0, 0.0, cap_b.length * 0.5])
  seg_b = math.rotate(seg_b, cap_b.transform.rot)
  pt_a, pt_b = geom_math.closest_segment_to_segment_points(
      cap_a.transform.pos - seg_a,
      cap_a.transform.pos + seg_a,
      cap_b.transform.pos - seg_b,
      cap_b.transform.pos + seg_b,
  )
  n, dist = math.normalize(pt_a - pt_b)
  penetration = cap_a.radius + cap_b.radius - dist

  cap_a_pos = pt_a - n * cap_a.radius
  cap_b_pos = pt_b + n * cap_b.radius
  pos = (cap_a_pos + cap_b_pos) * 0.5

  c = Contact(pos, n, penetration, *_combine(cap_a, cap_b))  # pytype: disable=wrong-arg-types  # jax-ndarray
  # add a batch dimension of size 1
  return jax.tree_map(lambda x: jp.expand_dims(x, axis=0), c)


def _capsule_convex(capsule: Capsule, convex: Convex) -> Contact:
  """Calculates contacts between a capsule and a convex object."""
  # Get convex transformed normals, faces, and vertices.
  normals = geom_mesh.get_face_norm(convex.vert, convex.face)
  faces = jp.take(convex.vert, convex.face, axis=0)

  @jax.vmap
  def transform_faces(faces, normals):
    faces = convex.transform.pos + jax.vmap(math.rotate, in_axes=[0, None])(
        faces, convex.transform.rot
    )
    normals = math.rotate(normals, convex.transform.rot)
    return faces, normals

  faces, normals = transform_faces(faces, normals)

  seg = jp.array([0.0, 0.0, capsule.length * 0.5])
  seg = math.rotate(seg, capsule.transform.rot)
  cap_pts = jp.array([
      capsule.transform.pos - seg,
      capsule.transform.pos + seg,
  ])

  # Get support from face normals.
  @jax.vmap
  def get_support(face, normal):
    pts = cap_pts - normal * capsule.radius
    sup = jax.vmap(lambda x: jp.dot(x - face[0], normal))(pts)
    return sup.min()

  support = get_support(faces, normals)
  has_support = jp.all(support < 0)

  # Pick the face with minimal penetration as long as it has support.
  support = jp.where(support >= 0, -1e12, support)
  best_idx = support.argmax()
  face = faces[best_idx]
  normal = normals[best_idx]

  # Clip the edge against side planes and create two contact points against the
  # face.
  edge_p0 = jp.roll(face, 1, axis=0)
  edge_p1 = face
  edge_normals = jax.vmap(jp.cross, in_axes=[0, None])(
      edge_p1 - edge_p0,
      normal,
  )
  cap_pts_clipped, mask = geom_math.clip_edge_to_planes(
      cap_pts[0], cap_pts[1], edge_p0, edge_normals
  )
  cap_pts_clipped = cap_pts_clipped - normal * capsule.radius
  face_pts = jax.vmap(geom_math.project_pt_onto_plane, in_axes=[0, None, None])(
      cap_pts_clipped, face[0], normal
  )
  # Create variables for the face contact.
  pos = (cap_pts_clipped + face_pts) * 0.5
  norm = jp.stack([normal] * 2, 0)
  penetration = jp.where(
      mask & has_support, jp.dot(face_pts - cap_pts_clipped, normal), -1
  )

  # Get a potential edge contact.
  # TODO handle deep edge penetration more gracefully, since edge_axis
  # can point in the wrong direction for deep penetration.
  edge_closest, cap_closest = jax.vmap(
      geom_math.closest_segment_to_segment_points, in_axes=[0, 0, None, None]
  )(edge_p0, edge_p1, cap_pts[0], cap_pts[1])
  e_idx = ((edge_closest - cap_closest) ** 2).sum(axis=1).argmin()
  cap_closest_pt, edge_closest_pt = cap_closest[e_idx], edge_closest[e_idx]
  edge_axis = cap_closest_pt - edge_closest_pt
  edge_axis, edge_dist = math.normalize(edge_axis)
  edge_pos = (
      edge_closest_pt + (cap_closest_pt - edge_axis * capsule.radius)
  ) * 0.5
  edge_norm = edge_axis
  edge_penetration = capsule.radius - edge_dist
  has_edge_contact = edge_penetration > 0

  # Create the contact.
  pos = jp.where(has_edge_contact, pos.at[0].set(edge_pos), pos)
  norm = jp.where(has_edge_contact, norm.at[0].set(edge_norm), norm)
  penetration = jp.where(
      has_edge_contact, penetration.at[0].set(edge_penetration), penetration
  )
  friction, elasticity, link_idx = jax.tree_map(
      lambda x: jp.repeat(x, 2), _combine(capsule, convex)
  )
  return Contact(pos, norm, penetration, friction, elasticity, link_idx)


def _capsule_mesh(capsule: Capsule, mesh: Mesh) -> Contact:
  """Calculates contacts between a capsule and a mesh."""

  @jax.vmap
  def capsule_face(face, face_norm):
    seg = jp.array([0.0, 0.0, capsule.length * 0.5])
    seg = math.rotate(seg, capsule.transform.rot)
    end_a, end_b = capsule.transform.pos - seg, capsule.transform.pos + seg

    tri_norm = math.rotate(face_norm, mesh.transform.rot)
    pt = mesh.transform.pos + jax.vmap(math.rotate, in_axes=[0, None])(
        face, mesh.transform.rot
    )
    p0, p1, p2 = pt[..., 0, :], pt[..., 1, :], pt[..., 2, :]

    seg_p, tri_p = geom_math.closest_segment_triangle_points(
        end_a, end_b, p0, p1, p2, tri_norm
    )
    n = seg_p - tri_p
    n, dist = math.normalize(n)
    penetration = capsule.radius - dist
    cap_p = seg_p - n * capsule.radius
    pos = (tri_p + cap_p) * 0.5
    return Contact(pos, n, penetration, *_combine(capsule, mesh))  # pytype: disable=wrong-arg-types  # jax-ndarray

  face_vert = jp.take(mesh.vert, mesh.face, axis=0)
  face_norm = geom_mesh.get_face_norm(mesh.vert, mesh.face)
  return capsule_face(face_vert, face_norm)


def _convex_plane(convex: Convex, plane: Plane) -> Contact:
  """Calculates contacts between a convex object and a plane."""

  @jax.vmap
  def transform_verts(vertices):
    return convex.transform.pos + math.rotate(vertices, convex.transform.rot)

  vertices = transform_verts(convex.vert)
  n = math.rotate(jp.array([0.0, 0.0, 1.0]), plane.transform.rot)
  support = jax.vmap(jp.dot, in_axes=[None, 0])(
      n, plane.transform.pos - vertices
  )
  idx = geom_math.manifold_points(vertices, support > 0, n)

  pos = vertices[idx]
  normal = jp.stack([n] * 4, axis=0)
  unique = jp.tril(idx == idx[:, None]).sum(axis=1) == 1
  penetration = jp.where(unique, support[idx], -1)
  friction, elasticity, link_idx = jax.tree_map(
      lambda x: jp.repeat(x, 4), _combine(convex, plane)
  )
  return Contact(pos, normal, penetration, friction, elasticity, link_idx)


def _convex_convex(convex_a: Convex, convex_b: Convex) -> Contact:
  """Calculates contacts between two convex objects."""
  # pad face vertices so that we can broadcast between geom_i and geom_j
  sa, sb = convex_a.face.shape[-1], convex_b.face.shape[-1]
  if sa < sb:
    face = jp.pad(convex_a.face, ((0, sb - sa)), 'edge')
    convex_a = convex_a.replace(face=face)
  elif sb < sa:
    face = jp.pad(convex_b.face, ((0, sa - sb)), 'edge')
    convex_b = convex_b.replace(face=face)

  normals_a = geom_mesh.get_face_norm(convex_a.vert, convex_a.face)
  normals_b = geom_mesh.get_face_norm(convex_b.vert, convex_b.face)
  faces_a = jp.take(convex_a.vert, convex_a.face, axis=0)
  faces_b = jp.take(convex_b.vert, convex_b.face, axis=0)

  def transform_faces(convex, faces, normals):
    faces = convex.transform.pos + jax.vmap(math.rotate, in_axes=[0, None])(
        faces, convex.transform.rot
    )
    normals = math.rotate(normals, convex.transform.rot)
    return faces, normals

  v_transform_faces = jax.vmap(transform_faces, in_axes=[None, 0, 0])
  faces_a, normals_a = v_transform_faces(convex_a, faces_a, normals_a)
  faces_b, normals_b = v_transform_faces(convex_b, faces_b, normals_b)

  def transform_verts(convex, vertices):
    vertices = convex.transform.pos + math.rotate(
        vertices, convex.transform.rot
    )
    return vertices

  v_transform_verts = jax.vmap(transform_verts, in_axes=[None, 0])
  vertices_a = v_transform_verts(convex_a, convex_a.vert)
  vertices_b = v_transform_verts(convex_b, convex_b.vert)

  unique_edges_a = jp.take(vertices_a, convex_a.unique_edge, axis=0)
  unique_edges_b = jp.take(vertices_b, convex_b.unique_edge, axis=0)

  c = geom_math.sat_hull_hull(
      faces_a,
      faces_b,
      vertices_a,
      vertices_b,
      normals_a,
      normals_b,
      unique_edges_a,
      unique_edges_b,
  )
  friction, elasticity, link_idx = jax.tree_map(
      lambda x: jp.repeat(x, 4), _combine(convex_a, convex_b)
  )

  return Contact(
      c.pos,
      c.normal,
      c.penetration,
      friction,
      elasticity,
      link_idx,
  )


def _mesh_plane(mesh: Mesh, plane: Plane) -> Contact:
  """Calculates contacts between a mesh and a plane."""

  @jax.vmap
  def point_plane(vert):
    n = math.rotate(jp.array([0.0, 0.0, 1.0]), plane.transform.rot)
    pos = mesh.transform.pos + math.rotate(vert, mesh.transform.rot)
    penetration = jp.dot(plane.transform.pos - pos, n)
    return Contact(pos, n, penetration, *_combine(mesh, plane))  # pytype: disable=wrong-arg-types  # jax-ndarray

  return point_plane(mesh.vert)


_TYPE_FUN = {
    (Sphere, Plane): _sphere_plane,
    (Sphere, Sphere): _sphere_sphere,
    (Sphere, Capsule): _sphere_capsule,
    (Sphere, Convex): _sphere_convex,
    (Sphere, Mesh): _sphere_mesh,
    (Capsule, Plane): _capsule_plane,
    (Capsule, Capsule): _capsule_capsule,
    (Capsule, Convex): _capsule_convex,
    (Capsule, Mesh): _capsule_mesh,
    (Convex, Convex): _convex_convex,
    (Convex, Plane): _convex_plane,
    (Mesh, Plane): _mesh_plane,
}


def _geom_pairs(sys: System) -> Iterator[Tuple[Geometry, Geometry]]:
  for i in range(len(sys.geoms)):
    for j in range(i, len(sys.geoms)):
      mask_i, mask_j = sys.geom_masks[i], sys.geom_masks[j]
      if (mask_i & mask_j >> 32) | (mask_i >> 32 & mask_j) == 0:
        continue
      geom_i, geom_j = sys.geoms[i], sys.geoms[j]
      if i == j and geom_i.link_idx is None:
        continue
      if i == j and geom_j.transform.pos.shape[0] == 1:
        continue
      yield (geom_i, geom_j)


def contact(sys: System, x: Transform) -> Optional[Contact]:
  """Calculates contacts in the system.

  Args:
    sys: system defining the kinematic tree and other properties
    x: link transforms in world frame

  Returns:
    Contact pytree, one row for each element in sys.contacts

  Raises:
    RuntimeError: if sys.contacts has an invalid type pair
  """

  contacts = []
  for geom_i, geom_j in _geom_pairs(sys):
    key = (type(geom_i), type(geom_j))
    fun = _TYPE_FUN.get(key)
    if fun is None:
      geom_i, geom_j = geom_j, geom_i
      fun = _TYPE_FUN.get((key[1], key[0]))
      if fun is None:
        raise RuntimeError(f'unrecognized collider pair: {key}')
    tx_i = x.take(geom_i.link_idx).vmap().do(geom_i.transform)

    if geom_i is geom_j:
      geom_i = geom_i.replace(transform=tx_i)
      choose_i, choose_j = jp.triu_indices(geom_i.link_idx.shape[0], 1)
      geom_i, geom_j = geom_i.take(choose_i), geom_i.take(choose_j)
      c = jax.vmap(fun)(geom_i, geom_j)  # type: ignore
      c = jax.tree_map(jp.concatenate, c)
    else:
      geom_i = geom_i.replace(transform=tx_i)
      # OK for geom j to have no parent links, e.g. static terrain
      if geom_j.link_idx is not None:
        tx_j = x.take(geom_j.link_idx).vmap().do(geom_j.transform)
        geom_j = geom_j.replace(transform=tx_j)
      vvfun = jax.vmap(jax.vmap(fun, in_axes=(0, None)), in_axes=(None, 0))
      c = vvfun(geom_i, geom_j)  # type: ignore
      c = jax.tree_map(lambda x: jp.concatenate(jp.concatenate(x)), c)

    contacts.append(c)

  if not contacts:
    return None

  # ignore penetration of two geoms within the same link
  c = jax.tree_map(lambda *x: jp.concatenate(x), *contacts)
  penetration = jp.where(c.link_idx[0] != c.link_idx[1], c.penetration, -1)
  c = c.replace(penetration=penetration)

  return c
