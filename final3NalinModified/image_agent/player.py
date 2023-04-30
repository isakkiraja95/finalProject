import numpy as np
import torch
from torch.serialization import load
import torchvision
import time
from PIL import Image
from os import path

#from image_agent.models import load_model, Detector

GOALS = np.float32([[0, 75], [0, -75]])

LOST_STATUS_STEPS = 10
LOST_COOLDOWN_STEPS = 10
START_STEPS = 25
LAST_PUCK_DURATION = 4
MIN_SCORE = 0.2
MAX_DET = 15
MAX_DEV = 0.7
MIN_ANGLE = 20
MAX_ANGLE = 120
TARGET_SPEED = 15
STEER_YIELD = 15
DRIFT_THRESH = 0.7
TURN_CONE = 100

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
    
class Team:
    agent_type = 'image'
        
    def __init__(self):
        #self.initialize_vars()
        
        self.timer = []
        self.puck_prev = []
        self.last_seen = []
        self.recover_steps = []
        self.use_puck = []
        self.cooldown = []
        
        self.model = torch.load(path.join(path.dirname(path.abspath(__file__)), 'detector.pt')).to(device)
        self.model.eval()
        
        #To Reduce the image size suitable for the model
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)),
                                                         torchvision.transforms.ToTensor()])

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
        self.step = 0
        for index in range(num_players):            
            self.timer.append(0)
            self.puck_prev.append(0)
            self.last_seen.append(0)
            self.recover_steps.append(0)
            self.use_puck.append(True)
            self.cooldown.append(0)
        
        print(f"New Match : {time.strftime('%H-%M-%S')}")
        return ['wilber'] * num_players

    def initialize_vars(self , index):      
        self.timer[index] = 0
        self.puck_prev[index] = 0
        self.last_seen[index] = 0
        self.recover_steps[index] = 0
        self.use_puck[index] = True
        self.cooldown[index] = 0

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

        p = []
        
        for index in range(self.num_players):
            player_info = player_state[index]
            image = player_image[index]

            # Detect Puck in the image
            img = self.transform(Image.fromarray(image)).to(device)
            pred = self.model.detect(img, max_pool_ks=7, min_score=MIN_SCORE, max_det=MAX_DET)
            puck_found = len(pred) > 0

            # try and detect if goal scored so we can reset (only needs to be done for one of the players)
            if index == 0 and np.linalg.norm(player_info['kart']['velocity']) < 1:
                if self.timer[index] == 0:
                    self.timer[index] = self.step
                elif self.step - self.timer[index] > 20:
                    self.initialize_vars(index)
            else:
                self.timer[index] = 0

            # Calculate Player's location in the game
            front = np.float32(player_info['kart']['front'])[[0, 2]]
            loc = np.float32(player_info['kart']['location'])[[0, 2]]

            # execute when we find puck on screen
            if puck_found:
                # takes avg of peaks
                puck_loc = np.mean([cx[1] for cx in pred])
                puck_loc = puck_loc / 64 - 1

                # ignores puck detections whose change is too much so that we ignore bad detections
                if self.use_puck[index] and np.abs(puck_loc - self.puck_prev[index]) > MAX_DEV:
                    puck_loc = self.puck_prev[index]
                    self.use_puck[index] = False
                else:
                    self.use_puck[index] = True

                # update vars
                self.puck_prev[index] = puck_loc
                self.last_seen[index] = self.step
            # if puck not seen then use prev location or start lost actions
            elif self.step - self.last_seen[index] < LAST_PUCK_DURATION:
                self.use_puck[index] = False
                puck_loc = self.puck_prev[index]
            else:
                puck_loc = None
                self.recover_steps[index] = LOST_STATUS_STEPS

            # calcualate direction vector
            dir = front - loc
            dir = dir / np.linalg.norm(dir)

            # calculate angle to own goal
            goal_dir = GOALS[self.team - 1] - loc
            dist_own_goal = np.linalg.norm(goal_dir)
            goal_dir = goal_dir / np.linalg.norm(goal_dir)

            goal_angle = np.arccos(np.clip(np.dot(dir, goal_dir), -1, 1))
            signed_own_goal_deg = np.degrees(
                -np.sign(np.cross(dir, goal_dir)) * goal_angle)

            # calculate angle to opp goal
            goal_dir = GOALS[self.team] - loc
            goal_dist = np.linalg.norm(goal_dir)
            goal_dir = goal_dir / np.linalg.norm(goal_dir)

            goal_angle = np.arccos(np.clip(np.dot(dir, goal_dir), -1, 1))
            signed_goal_angle = np.degrees(
                -np.sign(np.cross(dir, goal_dir)) * goal_angle)

            # restrict dist between [1,2] so we can use a weight function
            goal_dist = (
                (np.clip(goal_dist, 10, 100) - 10) / 90) + 1

            # set aim point if not cooldown or in recovery
            if (self.cooldown[index] == 0 or puck_found) and self.recover_steps[index] == 0:
                # if angle isn't extreme then weight our attack angle by dist
                if MIN_ANGLE < np.abs(signed_goal_angle) < MAX_ANGLE:
                    distW = 1 / goal_dist ** 3
                    aim_point = puck_loc + \
                        np.sign(puck_loc - signed_goal_angle /
                                TURN_CONE) * 0.3 * distW
                # if two tight then just chase puck
                else:
                    aim_point = puck_loc
                # sets the speed as const if found
                if self.last_seen[index] == self.step:
                    brake = False
                    acceleration = 0.75 if np.linalg.norm(
                        player_info['kart']['velocity']) < TARGET_SPEED else 0
                else:
                    brake = False
                    acceleration = 0
            # cooldown actions
            elif self.cooldown[index] > 0:
                self.cooldown[index] -= 1
                brake = False
                acceleration = 0.5
                aim_point = signed_goal_angle / TURN_CONE
            # recovery actions
            else:
                # if not a goal keep backing up
                if dist_own_goal > 10:
                    aim_point = signed_own_goal_deg / TURN_CONE
                    acceleration = 0
                    brake = True
                    self.recover_steps[index] -= 1
                # if at goal then cooldown on reversing
                else:
                    self.cooldown[index] = LOST_COOLDOWN_STEPS
                    aim_point = signed_goal_angle / TURN_CONE
                    acceleration = 0.5
                    brake = False
                    self.recover_steps[index] = 0

            # set steering/drift
            steer = np.clip(aim_point * STEER_YIELD, -1, 1)
            drift = np.abs(aim_point) > DRIFT_THRESH
            
            # Boost speed with Nitro
            nitro = False
            
            # For starters go in constant pace towards goal
            if self.step < START_STEPS:
                acceleration = 1
                steer = signed_goal_angle
                
            result = {
                'steer': steer,
                'acceleration': acceleration,
                'brake': brake,
                'drift': drift,
                'nitro': nitro, 
                'rescue': False
            }
            p.append(result)
        
        self.step += 1

        return p
        #return [dict(acceleration=1, steer=0)] * self.num_players
