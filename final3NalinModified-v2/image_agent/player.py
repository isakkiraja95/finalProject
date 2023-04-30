import numpy as np
import torch
import torchvision
import time

from torch.serialization import load
from PIL import Image
from os import path

LOST_STATUS_STEPS = 10
LOST_COOLDOWN_STEPS = 10
START_STEPS = 25
LAST_PUCK_DURATION = 4
MAX_DEV = 0.7
MIN_ANGLE = 20
MAX_ANGLE = 120
TARGET_SPEED = 15
STEER_YIELD = 15
TURN_CONE = 100

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
    
class Team:
    agent_type = 'image'
        
    def __init__(self):
        self.timevalidity = []
        self.backtrace = []
        self.puck_valid = []
        self.puck_prev = []
        self.last_seen = []
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
            self.timevalidity.append(0)
            self.backtrace.append(0)
            self.puck_valid.append(True)
            self.puck_prev.append(0)
            self.last_seen.append(0)
            self.cooldown.append(0)
        
        print(f"New Match : {time.strftime('%H-%M-%S')}")
        return ['wilber'] * num_players

    def resetCurrentState(self , index):      
        self.timevalidity[index] = 0
        self.puck_prev[index] = 0
        self.last_seen[index] = 0
        self.backtrace[index] = 0
        self.puck_valid[index] = True
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
            
            # Calculate Player's current_kart features in the game
            kart_front = torch.tensor(player_info['kart']['front'], dtype=torch.float32)[[0, 2]]
            kart_center = torch.tensor(player_info['kart']['location'], dtype=torch.float32)[[0, 2]]
            kart_direction = (kart_front-kart_center)
            kart_direction = kart_direction / torch.norm(kart_direction )
            kart_angle = torch.atan2(kart_direction[1], kart_direction[0])
            kart_velocity = np.linalg.norm(player_info['kart']['velocity'])

            # try and detect if goal scored so we can reset (only needs to be done for one of the players)
            if index == 0 and kart_velocity < 1:
                if self.timevalidity[index] == 0:
                    self.timevalidity[index] = self.step
                elif self.step - self.timevalidity[index] > 20:
                    self.resetCurrentState(index)
            else:
                self.timevalidity[index] = 0

            # Detect Puck in the image
            img = self.transform(Image.fromarray(image)).to(device)
            pred_prob = self.model.detect(img, max_pool_ks=7, min_score=0.2, max_det=15)
            puck_found = len(pred_prob) > 0
            
            # execute when we find puck on screen
            if puck_found:
                # takes avg of peaks
                puck_location = np.mean([cx[1] for cx in pred_prob])
                puck_location = puck_location / 64 - 1

                # ignores puck detections whose change is too much so that we ignore bad detections
                if self.puck_valid[index] and np.abs(puck_location - self.puck_prev[index]) > MAX_DEV:
                    puck_location = self.puck_prev[index]
                    self.puck_valid[index] = False
                else:
                    self.puck_valid[index] = True

                # update vars
                self.puck_prev[index] = puck_location
                self.last_seen[index] = self.step
            # if puck not seen then use prev kart_center or start lost actions
            elif self.step - self.last_seen[index] < LAST_PUCK_DURATION:
                self.puck_valid[index] = False
                puck_location = self.puck_prev[index]
            else:
                puck_location = None
                self.backtrace[index] = LOST_STATUS_STEPS

            # Obtain Angle to Own Goal
            own_goal_line_center = torch.tensor(soccer_state['goal_line'][(self.team+1)%2], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
            own_goal_line_direction = (own_goal_line_center-kart_center)
            own_goal_line_distance = np.linalg.norm(own_goal_line_direction)
            own_goal_line_direction = own_goal_line_direction / own_goal_distance
            #own_goal_line_angle = torch.atan2(kart_direction, own_goal_line_direction)
            
            own_goal_line_angle = np.arccos(np.clip(np.dot(kart_direction, own_goal_line_direction), -1, 1))
            own_goal_line_degree = np.degrees(-np.sign(np.cross(kart_direction, own_goal_line_direction)) * own_goal_line_angle)

            # Obtain Angle to Opp Goal
            opp_goal_line_center = torch.tensor(soccer_state['goal_line'][(self.team)%2], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
            opp_goal_line_direction = (opp_goal_line_center-kart_center)
            opp_goal_line_distance = np.linalg.norm(opp_goal_line_direction)
            opp_goal_line_direction = opp_goal_line_direction / opp_goal_line_distance
            #opp_goal_line_angle = torch.atan2(kart_direction, opp_goal_line_direction)
            
            opp_goal_line_angle = np.arccos(np.clip(np.dot(kart_direction, opp_goal_line_direction), -1, 1))
            opp_goal_line_degree = np.degrees(-np.sign(np.cross(kart_direction, opp_goal_line_direction)) * opp_goal_line_angle)
            
            # restrict dist between [1,2] so we can use a weight function
            opp_goal_line_distance = ((np.clip(opp_goal_line_distance, 10, 100) - 10) / 90) + 1

            # set aim point if not cooldown or in recovery
            if (self.cooldown[index] == 0 or puck_found) and self.backtrace[index] == 0:
                # if angle isn't extreme then weight our attack angle by dist
                if MIN_ANGLE < np.abs(opp_goal_line_degree) < MAX_ANGLE:
                    distW = 1 / goal_dist ** 3
                    aim_point = puck_location + \
                        np.sign(puck_location - opp_goal_line_degree /
                                TURN_CONE) * 0.3 * distW
                # if two tight then just chase puck
                else:
                    aim_point = puck_location
                # sets the speed as const if found
                if self.last_seen[index] == self.step:
                    brake = False
                    acceleration = 0.75 if kart_velocity < TARGET_SPEED else 0
                else:
                    brake = False
                    acceleration = 0
            # cooldown actions
            elif self.cooldown[index] > 0:
                self.cooldown[index] -= 1
                brake = False
                acceleration = 0.5
                aim_point = opp_goal_line_degree / TURN_CONE
            # recovery actions
            else:
                # if not a goal keep backing up
                if own_goal_line_distance > 10:
                    aim_point = own_goal_line_degree / TURN_CONE
                    acceleration = 0
                    brake = True
                    self.backtrace[index] -= 1
                # if at goal then cooldown on reversing
                else:
                    self.cooldown[index] = LOST_COOLDOWN_STEPS
                    aim_point = opp_goal_line_degree / TURN_CONE
                    acceleration = 0.5
                    brake = False
                    self.backtrace[index] = 0

            # set steering/drift
            steer = np.clip(aim_point * STEER_YIELD, -1, 1)
            
            #Enable drift if the destination is 70% more deviated angle
            drift = np.abs(aim_point) > 0.7
            
            # Boost speed with Nitro
            nitro = False
            
            # For starters go in constant pace towards goal
            if self.step < START_STEPS:
                acceleration = 1
                steer = opp_goal_line_degree
                
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
