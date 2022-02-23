
import math
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from meshcat.animation import Animation
vis = meshcat.Visualizer().open()

#rendering sphere
RADIUS=0.5
sphere = g.Sphere([RADIUS])
vis['sphere'].set_object(sphere,g.MeshPhongMaterial(color=0x5555ff, wireframe=True))

ground = g.Box([10,10,0.01])
vis['ground'].set_object(ground,g.MeshLambertMaterial( color=0xffffff, wireframe=True))


import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import brax
from google.protobuf import text_format
from brax.physics.base import validate_config


#import jax
#with jax.disable_jit():
if 1:
  _SYSTEM_CONFIG = """
bodies {
  name: "$ MainBallBody"
  colliders {
    capsule {
      radius: 0.5
      length: 1.001
    }
  }
  inertia { x: 1.0 y: 1.0 z: 1.0 }
  mass: 1
}


bodies {
  name: "Ground"
  colliders {
    plane {}
  }
  inertia { x: .2 y: .2 z: .2 }
  mass: 1
  frozen { all: true }
}

friction: 1.0
gravity { y:0.4 z: -10 }
angular_damping: 0
baumgarte_erp: 0.1

collide_include {
  first: "$ MainBallBody"
  second: "Ground"
}

dt: 0.0167
substeps: 2

defaults {
  qps {
    name: "$ MainBallBody"
    pos {
      x: 0, z: 1.5
    }
    
    
  }
} 
  
  """

  config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
  print(config)


  print(dir(config))

  print(config.dt)

  #config.dt *= action_repeat
  #config.substeps *= action_repeat
  s = brax.System(config)

  config2 = validate_config(config)
  num_bodies = len(config.bodies)
  print("num_bodies=",num_bodies)
  num_joints = len(config.joints)
  print("num_joints=",num_joints)
  print("num_joint_dof=",s.num_joint_dof)

  a = s.default_qp(0)
  print(dir(a))
  print("default_qp.pos=", a.pos)
  print("default_qp.rot=", a.rot)
  import numpy as np
  action = np.array([0]*8)
  
  
  def matrix(pos, rot):
    q = np.array([rot[0],rot[1],rot[2],rot[3]], dtype=np.float64, copy=True)
    n = np.dot(q, q)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], pos[0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], pos[1]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], pos[2]],
        [0.0                , 0.0                ,0.0                 , 1.0]])
  
  anim = Animation()
  a1 = a
  for frame_index in range(500):
    print("frame_index=",frame_index)
    a1 = s.step(a1, action)[0]
    mat4 = matrix(a1.pos[0], a1.rot[0])
    
    with anim.at_frame(vis, frame_index) as frame:
      frame["sphere"].set_transform(mat4)
    
  # `set_animation` actually sends the animation to the
  # viewer. By default, the viewer will play the animation
  # right away. To avoid that, you can also pass `play=False`. 
  vis.set_animation(anim)#, play=False)
  
  print("a1=",a1)
  print("a1.pos=",a1.pos)
  print("a1.rot=",a1.rot)
  print("a1.vel=",a1.vel)
  print("a1.ang=",a1.ang)
  print("dir(a1)=",dir(a1))
