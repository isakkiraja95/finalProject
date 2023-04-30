import numpy as np
import torch
import torchvision
import time

from torch.serialization import load
from PIL import Image
from os import path

TURN_ANGLE = 100

goal_state = np.float32([[0, 75], [0, -75]])

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
    
class Team:
    agent_type = 'image'
        
    def __init__(self):
        self.timevalidity = []
        self.backtrace = []
        self.puck_valid = []
        self.puck_prev_location = []
        self.puck_last_detected = []
        self.waitTime = []
        
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
            self.puck_prev_location.append(0)
            self.puck_last_detected.append(0)
            self.waitTime.append(0)
        
        print(f"New Match : {time.strftime('%H-%M-%S')}")
        return ['wilber'] * num_players

    def resetCurrentState(self , index):      
        self.timevalidity[index] = 0
        self.puck_prev_location[index] = 0
        self.puck_last_detected[index] = 0
        self.backtrace[index] = 0
        self.puck_valid[index] = True
        self.waitTime[index] = 0

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
            kart_front = np.float32(player_info['kart']['front'])[[0, 2]]
            kart_center = np.float32(player_info['kart']['location'])[[0, 2]]
            kart_direction = (kart_front-kart_center)
            kart_direction = kart_direction / np.linalg.norm(kart_direction )
            
            kart_velocity = np.linalg.norm(player_info['kart']['velocity'])

            # If Goal Scored, Reset the current state
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
            
            if puck_found:
                # Extract Puck's Peak location
                puck_curr_location = np.mean([x[1] for x in pred_prob])
                puck_curr_location = puck_curr_location / 64 - 1

                # The model is unable to find the puck properly or it moved too far to make a conclusion. In such cases ignore it
                if self.puck_valid[index] and np.abs(puck_curr_location - self.puck_prev_location[index]) > 0.7:
                    self.puck_valid[index] = False
                    puck_curr_location = self.puck_prev_location[index]
                else:
                    self.puck_valid[index] = True

                self.puck_last_detected[index] = self.step
                self.puck_prev_location[index] = puck_curr_location
            elif self.step - self.puck_last_detected[index] < 5:
                self.puck_valid[index] = False
                puck_curr_location = self.puck_prev_location[index]
            else:
                puck_curr_location = None
                self.backtrace[index] = 10
            
            # Obtain Angle to Own Goal
            own_goal_line_center = goal_state[self.team - 1]
            own_goal_line_direction = (own_goal_line_center-kart_center)
            own_goal_line_distance = np.linalg.norm(own_goal_line_direction)
            own_goal_line_direction = own_goal_line_direction / own_goal_line_distance
            #own_goal_line_angle = torch.atan2(kart_direction, own_goal_line_direction)
            
            own_goal_line_angle = np.arccos(np.clip(np.dot(kart_direction, own_goal_line_direction), -1, 1))
            own_goal_line_degree = np.degrees(-np.sign(np.cross(kart_direction, own_goal_line_direction)) * own_goal_line_angle)

            
            # Obtain Angle to Opp Goal
            opp_goal_line_center = goal_state[self.team]
            opp_goal_line_direction = (opp_goal_line_center-kart_center)
            opp_goal_line_distance = np.linalg.norm(opp_goal_line_direction)
            opp_goal_line_direction = opp_goal_line_direction / opp_goal_line_distance
            #opp_goal_line_angle = torch.atan2(kart_direction, opp_goal_line_direction)
            
            opp_goal_line_angle = np.arccos(np.clip(np.dot(kart_direction, opp_goal_line_direction), -1, 1))
            opp_goal_line_degree = np.degrees(-np.sign(np.cross(kart_direction, opp_goal_line_direction)) * opp_goal_line_angle)
            opp_goal_line_distance = ((np.clip(opp_goal_line_distance, 10, 100) - 10) / 90) + 1

            # Find a target_location if not waitTime or in recovery
            if (self.waitTime[index] == 0 or puck_found) and self.backtrace[index] == 0:
                if 20 < np.abs(opp_goal_line_degree) and np.abs(opp_goal_line_degree) < 120:
                    weighted_distance = 1 / opp_goal_line_distance ** 3
                    target_location = puck_curr_location + np.sign(puck_curr_location - opp_goal_line_degree /TURN_ANGLE) * 0.3 * weighted_distance
                else:
                    target_location = puck_curr_location
                
                if self.puck_last_detected[index] == self.step:
                    brake = False
                    acceleration = 0
                    if kart_velocity < 20:
                        acceleration = 0.75
                else:
                    brake = False
                    acceleration = 0
            elif self.waitTime[index] > 0:
                self.waitTime[index] -= 1
                brake = False
                acceleration = 0.5
                target_location = opp_goal_line_degree / TURN_ANGLE
            else:
                if own_goal_line_distance > 10:
                    target_location = own_goal_line_degree / TURN_ANGLE
                    acceleration = 0
                    brake = True
                    self.backtrace[index] -= 1
                else:
                    self.waitTime[index] = 10
                    target_location = opp_goal_line_degree / TURN_ANGLE
                    acceleration = 0.5
                    brake = False
                    self.backtrace[index] = 0

            # Update Steering based on target location
            steer = np.clip(target_location * 20, -1, 1)
            
            #Enable drift if the destination is 70% more deviated angle
            drift = np.abs(target_location) > 0.7
            
            nitro = False
            
               #Go straight for the intial few steps
            if self.step < 25:
                acceleration = 1
                steer = opp_goal_line_degree
                
                if self.step > 5 and self.step < 20: 
                # Boost speed with Nitro to get a head start
                    nitro = True    
                    
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
        #print(self.step)
        return p
        #return [dict(acceleration=1, steer=0)] * self.num_players
