import numpy as np
import torch
import torchvision
from PIL import Image

from image_agent.models import load_model
GOAL_POS = np.float32([[0, 75], [0, -75]])  # (0 and 2 coor) Blue, Red

# Steps duration of lost status
LOST_STATUS_STEPS = 10
LOST_COOLDOWN_STEPS = 10
START_STEPS = 40
LAST_PUCK_DURATION = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def norm(vector):
    # return np.sqrt(np.sum(np.square(vector)))
    return np.linalg.norm(vector)

class Team:
    agent_type = 'image'
        
    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        #self.actcall = 0
        self.model = load_model('det.th').to(device)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)),
                                                         torchvision.transforms.ToTensor()])
    def status_reset(self):
            # Timing variables
            self.step = 0
            self.timer = 0

            # Status variables
            self.puck_last_pos = 0
            self.step_lost = 0
            self.step_back = 0
            self.normal = True
            self.lost_cooldown = 0

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        print('New Match called')

        self.status_reset()
        return ['wilber'] * num_players

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # TODO: Change me. I'm just cruising straight
        #print('Act called', self.actcall)
        #self.actcall = self.actcall + 1
        
        result = []
		
        for index in range(self.num_players):
          # Get positions
          #print(player_state[index])
          #print('Player' , index, player_state[index]['kart'])
          front = np.float32(player_state[index]['kart']['front'])[[0, 2]]
          #print('Player' , index)
          kart = np.float32(player_state[index]['kart']['location'])[[0, 2]]
          #print(kart)
          # Reset variables status if scored a goal
          if norm(player_state[index]['kart']['velocity']) < 1:
            if self.timer == 0:
              self.timer = self.step
            elif self.step - self.timer > 20:
              self.status_reset()
          else:
            self.timer = 0
          # Get predicted puck position
          img = self.transform(Image.fromarray(player_image[index])).to(device)
          pred = self.model.detect(img, max_pool_ks=7, min_score=0.2, max_det=15)  # (score, cx, cy, w, h)
          puck_visible = len(pred) > 0
          if puck_visible:
            
            # Average predictions
            puck_pos = np.mean([cx[1] for cx in pred])
            puck_pos = puck_pos / 64 - 1  # [0, 128] -> [-1, 1]
            puck_size = np.mean([cx[2] for cx in pred])
            puck_size = puck_size / 128  # [0, 128] -> [0, 1]

            # If vary large change, ignore this step
            if self.normal and np.abs(puck_pos - self.puck_last_pos) > 0.5:
              puck_pos = self.puck_last_pos
              self.normal = False
            else:
              self.normal = True

            # Update status variables
            self.puck_last_pos = puck_pos
            self.step_lost = self.step
          elif self.step - self.step_lost < LAST_PUCK_DURATION:
            self.normal = False
            puck_pos = self.puck_last_pos
          else:
            puck_pos = None
            self.step_back = LOST_STATUS_STEPS

          # Opposite goal theta
          u = front - kart
          u = u / np.linalg.norm(u)
          v = GOAL_POS[self.team] - kart
          dist_opp_goal = norm(v)
          v = v / np.linalg.norm(v)

          theta_goal = np.arccos(np.dot(u, v))
          signed_theta_opp_goal_deg = np.degrees(-np.sign(np.cross(u, v)) * theta_goal)

          # Self goal theta
          v = GOAL_POS[self.team - 1] - kart
          dist_own_goal = norm(v)
          v = v / np.linalg.norm(v)

          theta_goal = np.arccos(np.dot(u, v))
          signed_theta_self_goal_deg = np.degrees(-np.sign(np.cross(u, v)) * theta_goal)

          # ideas: if closer to goal, more important to have angle of goal
          # todo ideas: width can be used to know how close is the puck
          # ideas: make the relation of closer and importance of the goal non linear (change when close not as impactful)
          dist_opp_goal = ((np.clip(dist_opp_goal, 10, 100) - 10) / 90) + 1  # [1, 2]
          if self.step_back == 0 and (self.lost_cooldown == 0 or puck_visible):
            if 20 < np.abs(signed_theta_opp_goal_deg) < 120:
              importance_dist = 1 / dist_opp_goal ** 3
              aim_point = puck_pos + np.sign(puck_pos - signed_theta_opp_goal_deg / 100) * 0.3 * importance_dist
            else:
              aim_point = puck_pos
            # print(f"{aim_point}, {puck_pos}")
            # aim_point = puck_pos
            if self.step_lost == self.step:
              # If have vision of the puck
              acceleration = 0.75 if norm(player_state[index]['kart']['velocity']) < 15 else 0
              brake = False
            else:
              # If no vision of the puck
              acceleration = 0
              brake = False
          elif self.lost_cooldown > 0:
            # If already in own goal, start going towards opposite goal
            aim_point = signed_theta_opp_goal_deg / 100
            acceleration = 0.5
            brake = False
            self.lost_cooldown -= 1
          else:
            # If in lost status, back towards own goal
            if dist_own_goal > 10:
              aim_point = signed_theta_self_goal_deg / 100  # [0, 1] aprox
              acceleration = 0
              brake = True
              self.step_back -= 1
            else:
              self.lost_cooldown = LOST_COOLDOWN_STEPS
              self.step_back = 0
              aim_point = signed_theta_opp_goal_deg / 100
              acceleration = 0.5
              brake = False

          if index == 1 and self.step < 25:
            # If second car, wait more until start
            acceleration = 0
            brake = False

          # Steer and drift
          steer = np.clip(aim_point * 15, -1, 1)
          drift = np.abs(aim_point) > 0.2
          self.step += 1

          # print(f"{acceleration}, {aim_point}, {steer}, {puck_visible}")
          if self.step < 25:
            steer = signed_theta_opp_goal_deg
          if self.step < START_STEPS:
            acceleration = 1
          #result.append([dict(acceleration=acceleration, steer=steer, brake=brake, drift=drift)])
        return [dict(acceleration=1, steer=0)] * self.num_players        
        #return result
