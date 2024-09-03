# Self Driving Car

# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import datetime

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Importing the TD3 object from your AI implementation
from ai import TD3, ReplayBuffer

# Configurations
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1583')
Config.set('graphics', 'height', '731')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Initialize the TD3 agent with continuous action space
state_dim = 5
action_dim = 1
max_action = 5  # Example max action value, adjust as necessary
brain = TD3(state_dim, action_dim, max_action)

# Hyperparameters
batch_size = 256
initial_buffer = 20000
reset_car_after = 1000
save_interval = 1000
# num_episodes = 1000x
# max_timesteps_per_episode = 200


# last_reward = 0
new_reward = 0
scores = []
im = CoreImage("./images_new/MASK1.png")

targets_list = [(1470, 70), (165, 227), (812, 560)]
initialization_grid = (6,2)
total_width = 1583
total_height = 731

# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global is_train

    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images_new/mask.png").convert('L')
    sand = np.asarray(img)/255

    is_train = True

    random_target = random.choice(targets_list)
    goal_x = random_target[0]
    goal_y = random_target[1]
    first_update = False

# Initializing the last distance
last_distance = 0

# Creating the car class
class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 10.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 10.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 10.
        

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

class Target(Widget):
    pass

def calculate_center_points(total_width, total_height, grid_size):
    grid_width, grid_height = grid_size

    # Calculate the width and height of each cell in the grid
    cell_width = total_width / grid_width
    cell_height = total_height / grid_height

    # Calculate the center points
    center_points = []
    for j in range(grid_height):
        for i in range(grid_width):
            center_x = round((i + 0.5) * cell_width)
            center_y = round((j + 0.5) * cell_height)
            center_points.append((center_x, center_y))
    
    return center_points

# Creating the game class
class Game(Widget):

    car = ObjectProperty(None)
    target = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    

    
    def __init__(self, **kwargs):
        super(Game, self).__init__(**kwargs)
        self.replay_buffer = ReplayBuffer(max_size=500000)
        self.timesteps = 0
        self.total_timesteps = 0
        self.update_interval = 1  # Update the policy every after every batch

        self.last_signal = [0] * state_dim
        self.last_action = [0] * action_dim
        self.last_reward = 0
        self.last_done = False

        self.max_training_iteration = 500000
        self.trn_it = 0
        self.episode_last_step_reward = 0
        self.episode_rewards = []

        # Define possible starting positions
        random_location = calculate_center_points(total_width, total_height,
                                                   initialization_grid)
        self.starting_positions = random_location #+ targets_list

    def serve_car(self, x = None, y = None):
        if x is not None and y is not None:
            self.car.center = (x, y)
        else:
            self.car.center = random.choice(self.starting_positions)
        # self.car.center = self.center
        self.car.velocity = Vector(6, 0)
    
    def update(self, dt):
        global brain
        global new_reward #last_reward
        global scores, last_distance
        global goal_x, goal_y, longueur, largeur
        global is_train

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        # Reinitialize car position every 1000 timesteps
        if (self.total_timesteps < initial_buffer and 
        self.total_timesteps % reset_car_after == 0):
            self.serve_car()

        # Get current state
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.
        # last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        new_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        self.replay_buffer.add((self.last_signal, 
                           self.last_action,
                            new_signal, 
                            self.last_reward, 
                            self.last_done))
        # Select action using the TD3 agent
        # action = brain.select_action(np.array(last_signal))
        action = brain.select_action(np.array(new_signal))
        rotation = float(action[0])  # Convert the continuous action to a rotation angle
        self.car.move(rotation)

        self.last_action = action
        self.last_signal = new_signal 
        

        # Calculate reward
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        
        # Update visuals
        self.target.pos = (goal_x, goal_y)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            # print(1, goal_x, goal_y, distance, rotation, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            # new_reward = -5
            new_reward = -1
            
        else:
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            new_reward = -0.2
            # print(0, goal_x, goal_y, distance, rotation, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))

            if distance < last_distance:
                # new_reward += 2 
                new_reward = 0.1

        if self.car.x < 5:
            self.car.x = 5
            # new_reward += -2
            new_reward = -1
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            # new_reward += -2
            new_reward = -1
        if self.car.y < 5:
            self.car.y = 5
            # new_reward += -2
            new_reward = -1
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            # new_reward += -2
            new_reward = -1

        if distance < 25:
            # new_reward += 5
            random_target = random.choice(targets_list)
            while goal_x == random_target[0] and goal_y == random_target[1]:
                random_target = random.choice(targets_list)
            goal_x = random_target[0]
            goal_y = random_target[1]
        # print(new_reward)
        self.last_reward = new_reward
        # Store the transition in replay buffer
        # new_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        done = distance < 25
        self.last_done = done
        if done:
            self.episode_rewards.append(self.episode_last_step_reward)
            self.episode_last_step_reward = 0
        else:
            self.episode_last_step_reward += new_reward            
        
        last_distance = distance
        # Train the TD3 agent
        # if len(replay_buffer.storage) > batch_size:
        #     brain.train(replay_buffer, batch_size)
        self.timesteps += 1
        self.total_timesteps += 1

        if (is_train and 
            self.total_timesteps > initial_buffer 
            and self.timesteps % self.update_interval == 0):
            # print("Training ...")
            # print(f"storage size {len(self.replay_buffer.storage)}, pointer {self.replay_buffer.ptr},positive samples {len([i for i, t in enumerate(self.replay_buffer.storage) if t[-2] >= self.replay_buffer.postive_sample_threshold])}")
            brain.train(self.replay_buffer, batch_size=batch_size)
            if self.trn_it % save_interval == 0:
                print(f"{datetime.datetime.now()}: {self.trn_it} - prev eps rewards: {self.episode_rewards}, curr eps reward {self.episode_last_step_reward}")
                print(f"storage size {len(self.replay_buffer.storage)}, pointer {self.replay_buffer.ptr},positive samples {len([i for i, t in enumerate(self.replay_buffer.storage) if t[-2] >= self.replay_buffer.postive_sample_threshold])}")
                brain.save()
            self.timesteps = 0
            self.trn_it += 1
            if self.trn_it >= self.max_training_iteration: 
                is_train = False
                brain.save()
 
# Adding the painting tools
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images_new/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20*density + 1)
            sand[int(touch.x)-10:int(touch.x)+10,int(touch.y)-10:int(touch.y)+10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)
class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car(x = 812, y = 560)
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        # clearbtn = Button(text = 'clear')
        # clearbtn.bind(on_release = self.clear_canvas)
        trnbtn = Button(text = 'train or test')
        trnbtn.bind(on_release = self.train_test_toggle)
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        savebtn.bind(on_release = self.save)
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        # parent.add_widget(clearbtn)
        parent.add_widget(trnbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent
    def train_test_toggle(self, obj):
        global is_train
        if is_train: is_train = False
        else: is_train = True

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))
    
    def save(self, obj):
        print("saving brain...")
        brain.save()
        # plt.plot(scores)
        # plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
