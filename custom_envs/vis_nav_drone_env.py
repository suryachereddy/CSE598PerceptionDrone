import gym
from gym import spaces
import airsim
import cv2
from PIL import Image
import open3d as o3d
import copy
import numpy as np
from pathlib import Path
import time
import os
from enum import Enum
from multiprocessing.managers import SyncManager
from multiprocessing import process
import multiprocessing.managers as m
import cloudpickle
import random
import multiprocessing as mp

#pretrained model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(device)
resnet=torch.nn.Sequential(*(list(resnet.children())[:-1]))
for param in resnet.parameters():
    param.requires_grad = False
resnet.eval()

CAPTURE_RATE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

# # Good example for use with discretized action space
# class ACT(Enum):
#   move_to = 1
#   rot_yaw_to = 2
#   rot_yaw = 3
#   up = 4
#   down = 5
#   left = 6
#   right = 7
#   forward = 8
#   backward = 9

# Good example for use with discretized action space
class ACT(Enum):
  up = 0
  down = 1
  left = 2
  right = 3
  forward = 4
  backward = 5


np.set_printoptions(precision=2)

CAPTURE_TRAIN_RGBD = True
DEADBAND = 0.5
TARGET_MIN_DIST = .1
ROT_DEADBAND = 0.00008
TARGET_CENTER_ROT_DEADBAND = 0.001

milliseconds = int(round(time.time() * 1000))
TRAIN_RGBD_DATASET_PATH = f'data{os.sep}train{os.sep}drone_rgbd_{milliseconds}{os.sep}'
DRONE_PROXY_AUTH_KEY = b'abasdknasdfasdf'
DRONE_PROXY_IP_ADDR = '127.0.0.1'
DRONE_PROXY_PORT = 50001

TARGET_ID = 25
NAV_STEP_DURATION = 0.1

NUM_AVAIL_DRONES = 1
lock = mp.Lock()


class DroneProxyManager(SyncManager):
  pass


def create_drone_proxy_client(host, port, authkey):
  DroneProxyManager.register('get_drone_proxy')
  proxy_manager = DroneProxyManager(address=(host, port), authkey=authkey)
  proxy_manager.connect()

  return proxy_manager


class DroneRetrieveProxy(object):
  drone = None
  drones = None
  avail_drones = {}

  @staticmethod
  def get_drone():
    if DroneRetrieveProxy.drones is None:
      DroneRetrieveProxy.drones = []
      for i in range(NUM_AVAIL_DRONES):
        DroneRetrieveProxy.drones.append(airsim.MultirotorClient())
        DroneRetrieveProxy.avail_drones[i] = True

    lock.acquire()

    drone = None
    drone_id = None
    for k in DroneRetrieveProxy.avail_drones.keys():
        if DroneRetrieveProxy.avail_drones[k]:
            drone = DroneRetrieveProxy.drones[k]
            drone_id = k
            DroneRetrieveProxy.avail_drones[k] = False
            break

    assert(drone is not None)

    lock.release()

    return drone, drone_id

  @staticmethod
  def release_drone(drone_id):
    lock.acquire()

    DroneRetrieveProxy.avail_drones[drone_id] = True

    lock.release()


def get_avail_drone():
  return [cloudpickle.dumps(DroneRetrieveProxy)]


def RebuildProxy(func, token, serializer, kargs):
  incref = (
          kargs.pop('incref', True) and
          not getattr(process.current_process(), '_inheriting', False)
  )
  return func(token, serializer, incref=incref, **kargs)

m.RebuildProxy = RebuildProxy


class VisNavDroneEnv(gym.Env):
  metadata = {"render.modes": ["rgb_array"]}

  def __init__(self, content=None, step_length=None,
               image_shape=None, embed_shape=None, drone=None, random=False):

      self.random = random
      if CAPTURE_TRAIN_RGBD:
        Path(TRAIN_RGBD_DATASET_PATH).mkdir(parents=True, exist_ok=True)
        self.rgbd_sequence = 0

      if content is not None:
        self.step_length, self.image_shape = content
      else:
        self.step_length = step_length
        self.image_shape = image_shape

      if embed_shape is not None:
        self.embed_shape = embed_shape

      self.drone = None
      self.drone_id = None

      self.state = {}
      self.dproxy_client = None
      self.max_steps = 100
      _dtype = np.float64
      # self.observation_space = gym.spaces.Dict({'obs': spaces.Box(low=0, high=1, shape=self.image_shape, dtype=np.uint8),
      #                                                                   # 'embed': gym.spaces.Box(low=0, high=1, shape=self.embed_shape,
      #                                                                   #                         dtype=_dtype)
      #                                                                   })
      self.observation_space = gym.spaces.Dict(
                                                spaces = {
                                                  "img": spaces.Box(low=0, high=1, shape=(512,1,1), dtype=np.uint8),

                                                  "sem_img": spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8),
                                                  "vec": spaces.Box(low=0, high=1, shape=self.embed_shape, dtype=np.float32),
                                                }
      ) 
      #self.observation_space = spaces.Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
                                                                                                                                                                                                 
      self.capture_cnt = 0
      self.cur_step = 0

      # self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Box(0, 1, (3,))))
      self.action_space = spaces.Discrete(6)
      #self.action_space = spaces.Box(-1.0,1.0,(3,))
      # TODO: get other signals

      self.image_requests = [airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True),
                             airsim.ImageRequest(0, airsim.ImageType.Segmentation),
                             airsim.ImageRequest(0, airsim.ImageType.Scene)]

      self.orig_pose = None

  def __deepcopy__(self, memo):
    return VisNavDroneEnv(copy.deepcopy((self.step_length, self.image_shape), memo=memo),
                          drone=None)

  def __del__(self):
      if self.dproxy_client is not None:
          drone_released = cloudpickle.loads(self.dproxy_client.get_drone_proxy().pop()).release_drone(self.drone_id)
          assert(drone_released)

  @staticmethod
  def draw_registration_result(source, target=None):
    geometries = [source]
    if target is not None:
      geometries.append(target)

    o3d.visualization.draw_geometries(geometries)

  def get_target_info(self):
    depth, semantic, rgb = self.get_obs()

    # iterate through semantic rgb array to find the pixel with value of [212, 51, 60] which is the color of the target
    target_rgb = np.array([212, 51, 60])

    # reverse the array as semantic image has gbr instead of rgb
    target_rgb = np.flip(target_rgb, axis=0)

    target_idx = np.where(np.all(semantic == target_rgb, axis=-1))

    target_id_not = np.where(np.all(semantic != target_rgb, axis=-1))

    semantic_mask = semantic.copy()
    semantic_mask[target_idx] = np.array([255, 255, 255])
    semantic_mask[target_id_not] = np.array([0, 0, 0])

    from PIL import Image
    img2 = Image.fromarray(semantic_mask.reshape(semantic.shape[0], semantic.shape[1], 3), 'RGB')
    img2.show()

    # Keep only 1 channel
    semantic_mask, _, _ = cv2.split(semantic_mask)
    semantic_mask = semantic_mask.reshape(semantic.shape[0], semantic.shape[1], 1)

    

    done = False

    self.cur_step += 1

    reward = 0
    rgb=torch.from_numpy(rgb).permute(2,0,1).float().div(255)
    rgb=rgb.to(device)
    rgb=resnet(rgb.unsqueeze(0))
    rgbrep=rgb.cpu().detach().numpy()
    

    _obs =  {
            "img": rgbrep, 
            "sem_img": semantic_mask, 
            "vec": depth
            } #TODO: REPLACE me

    target_depth = np.mean(depth[target_idx])

    return target_idx,target_depth, _obs, done

  def step(self, action):
    #time.sleep(self.step_length)

    self.drone.simPause(True)
    
    target_idx, target_depth, _obs, done = self.get_target_info()

    # if target not in view give reward of -100 and end the episode
    if np.shape(target_idx)[-1] == 0:
      done = True
      
      reward = -350
      self.cur_step = 0

      print('Target not found')
      self.drone.simPause(False)
      return _obs, reward, done, self.state

    # if target is in view
    else:

      # if target is within 0.5 meter of the drone give reward of +1000 and end the episode
      if(target_depth < 0.5):
        print("Within range of Target")
        done = True
        self.cur_step = 0
        reward = 1000
        self.drone.simPause(False)
        return _obs, reward, done, self.state

      # if target is not within 0.5 meter of the drone
      else:
        # print(target_depth)
        offset_weight = 1/10
        # moment = cv2.moments(semantic_mask)
        # target_center_x = int(moment["m10"]/moment["m00"])
        # target_center_y = int(moment["m01"]/moment["m00"])
        # offset = np.sqrt((self.image_shape[1]//2-target_center_x)**2 +
        #                   (self.image_shape[0]//2-target_center_y)**2)
        # reward at very timestep is a function of the distance to the target and the closeness target to center of the image
        # The 0.1 and 0.4 coeffecients are selected with trial and error
        #reward = ((-target_depth) - (offset_weight*offset)*0.1)*0.4
        #reward = 1/target_depth
        reward = -1
    
        # if greater than the maximum time steps end the episode
        if self.cur_step >= self.max_steps:
          done = True
          self.cur_step = 0
          print("done")
          self.drone.simPause(False)
          return _obs, reward, done, self.state

        # if collision happens end the episode
        if self.drone.simGetCollisionInfo().has_collided:
          reward=-350
          done=True
          self.cur_step=0
          self.drone.simPause(False)
          print("collided")
          return _obs, reward, done, self.state

        
    

    # TODO: reward shaping here

    # self.drone.moveByVelocityAsync(0.0, -0.0, -2.0, duration=NAV_STEP_DURATION,
    #                           drivetrain=airsim.DrivetrainType.ForwardOnly,
    #                           yaw_mode=airsim.YawMode(False, 0))  # TODO: REMOVE ME

    #reward = None  # TODO: REPLACE ME
    #done = False   # TODO: REPLACE ME

    # TODO: embedding fusion setup here.. other side is model inference side
    
    self.drone.simPause(False)
    if action == ACT.forward.value:
      self.drone.moveByVelocityAsync(-1.0, 0.0, 0.0, duration=NAV_STEP_DURATION,
                                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                yaw_mode=airsim.YawMode(False, 0)) #.join()

    elif action == ACT.backward.value:
      self.drone.moveByVelocityAsync(1.0, 0.0, 0.0, duration=NAV_STEP_DURATION,
                                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                yaw_mode=airsim.YawMode(False, 0)) #.join()

    elif action == ACT.right.value:
      self.drone.moveByVelocityAsync(0.0, -1.0, 0.0, duration=NAV_STEP_DURATION,
                                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                yaw_mode=airsim.YawMode(False, 0)) #.join()

    elif action == ACT.left.value:
      self.drone.moveByVelocityAsync(0.0, 1.0, 0.0, duration=NAV_STEP_DURATION,
                                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                yaw_mode=airsim.YawMode(False, 0)) #.join()

    elif action == ACT.up.value:
      self.drone.moveByVelocityAsync(0.0, 0.0, -1.0, duration=NAV_STEP_DURATION,
                                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                yaw_mode=airsim.YawMode(False, 0)) #.join()

    elif action == ACT.down.value:
      self.drone.moveByVelocityAsync(0.0, 0.0, 1.0, duration=NAV_STEP_DURATION,
                                drivetrain=airsim.DrivetrainType.ForwardOnly,
                                yaw_mode=airsim.YawMode(False, 0)) #.join()
    # self.drone.moveByVelocityAsync(float(action[0]),float(action[1]),float(action[2]), duration=NAV_STEP_DURATION,
    #                             drivetrain=airsim.DrivetrainType.ForwardOnly,
    #                             yaw_mode=airsim.YawMode(False, 0)).join()
    # print reward and target depth on same line
    print("reward: ", reward, "target depth: ", target_depth)
    return _obs, reward, done, self.state
    #return rgb, reward, done, self.state

  def reset(self):
    self.currenttarget=0
    if self.drone is None:
      if self.dproxy_client is None:
        self.dproxy_client = create_drone_proxy_client(DRONE_PROXY_IP_ADDR, DRONE_PROXY_PORT, DRONE_PROXY_AUTH_KEY)

      self.drone, self.drone_id = cloudpickle.loads(self.dproxy_client.get_drone_proxy().pop()).get_drone()

      self.objects = self.drone.simListSceneObjects()
      self.target = 'SM_AI_vol1_06_teapot_4'
      found = self.drone.simSetSegmentationObjectID(self.target, TARGET_ID)
      assert(found)

    self._init_flight()

    self.drone.simPause(True)
    
    
    target_idx, target_depth, _obs, done = self.get_target_info()

    return _obs
    #return rgb

  def render(self):
    return self.get_obs()

  def get_obs(self):

    # get depth, semantic, rgb
    responses = self.drone.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False),
                             airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False),
                             airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
    depth_res = responses[0]
    semantic_res = responses[1]
    rgb_res = responses[2]

    # get depth, semantic, rgb

    depth = np.asarray(depth_res.image_data_float, dtype=np.float32).reshape(depth_res.height, depth_res.width,1)
    
    semantic = np.fromstring(semantic_res.image_data_uint8, dtype=np.uint8).reshape(semantic_res.height, semantic_res.width, 3)

    rgb = np.fromstring(rgb_res.image_data_uint8, dtype=np.uint8).reshape(rgb_res.height, rgb_res.width, 3)
    print(depth.shape)
    # from PIL import Image
    # img1 = Image.fromarray(depth, 'L')
    # img1.show()
    # img2 = Image.fromarray(semantic, 'BGR')
    # img2.show()
    # img3 = Image.fromarray(rgb, 'BGR')
    # img3.show()
  
    # print(img_rgb.shape)

    return depth, semantic, rgb

  def _init_flight(self):
    self.drone.reset()
    self.drone.confirmConnection()
    self.drone.enableApiControl(True)
    self.drone.armDisarm(True)

    is_landed = self.drone.getMultirotorState().landed_state
    if is_landed == airsim.LandedState.Landed:
        self.drone.takeoffAsync(0.1).join()
    else:
        self.drone.hoverAsync().join()

    # TODO: random teleportion here
    pose = self.drone.simGetVehiclePose()

    pose.position.x_val += random.uniform(-2, 2)
    pose.position.y_val += random.uniform(-2, 2)
    pose.position.z_val += random.uniform(-2, 2)

    self.drone.simSetVehiclePose(pose, True)

    