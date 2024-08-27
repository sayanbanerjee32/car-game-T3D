# Self Driving Car

# Importing the libraries
import numpy as np
# from random import random, randint
import random
# import matplotlib.pyplot as plt
import time

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

# Importing the Dqn object from our AI in ai.py
# from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1583')
Config.set('graphics', 'height', '731')
# Config.set('graphics', 'width', '1429')
# Config.set('graphics', 'height', '660')


# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# commented for T3D
# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
# brain = Dqn(5,3,0.9)
# action2rotation = [0,5,-5]
last_reward = 0
scores = []
# im = CoreImage("./images/MASK1.png")
im = CoreImage("./images_new/MASK1.png")
# textureMask = CoreImage(source="./kivytest/simplemask1.png")

targets_list = [(1470, 70),
                (165, 227),
                (812, 560)]

# Initializing the map
first_update = True

# For T3D
max_steps = 2500
current_step = 0
done = True
on_road = -1
on_road_count = 0
off_road_count = 0
boundary_hit_count = 0
goal_hit_count = 0
train_episode_num = 0
eval_episode_num = 0
mode="eval"

def init():
    global sand
    global goal_x
    global goal_y
    global first_update

    # for T3D
    global done
    global max_steps
    global current_step

    sand = np.zeros((longueur,largeur))
    # img = PILImage.open("./images/mask.png").convert('L')
    img = PILImage.open("./images_new/mask.png").convert('L')
    sand = np.asarray(img)/255
    # goal_x = 1420
    # goal_y = 622
    random_target = random.choice(targets_list)
    goal_x = random_target[0]
    goal_y = random_target[1]
    first_update = False
    # global swap
    # swap = 0

    # for T3D
    done = False
    current_step = 0


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

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    target = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    
    # for T3D
    def serve_car(self, start_event, reset_q, mode_q, state_q, action_q, next_state_reward_done_tuple_q):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

        # for T3D
        self.car.start_event = start_event
        self.car.reset_q = reset_q
        self.car.mode_q = mode_q
        self.car.state_q = state_q
        self.car.action_q = action_q
        self.car.next_state_reward_done_tuple_q = next_state_reward_done_tuple_q
    
    def get_state(self):
        # Sensor readings
        sensor1_signal = self.car.signal1
        sensor2_signal = self.car.signal2
        sensor3_signal = self.car.signal3

        # Calculate orientation to goal
        goal_direction_vector = Vector(goal_x - self.car.x, goal_y - self.car.y)
        car_orientation = Vector(*self.car.velocity).angle(goal_direction_vector) / 180.0

        # Compile the state representation
        state = np.array([
            sensor1_signal,  # Distance to obstacle from sensor 1
            sensor2_signal,  # Distance to obstacle from sensor 2
            sensor3_signal,  # Distance to obstacle from sensor 3
            car_orientation,  # Orientation towards the goal
            -car_orientation  # Symmetric orientation (can help with symmetry in learning)
        ])

        # Return the state as a numpy array
        return state

    
    # def serve_target(self):
    #     self.target.center = self.center

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        # global swap
        

        longueur = self.width
        largeur = self.height
        
        # for T3D
        self.car.start_event.wait()
        if done == True:
            reset = self.car.reset_q.get()
            if reset == True:
                print("first_update is set to True")
                first_update = True
				
            (mode, train_episode_num, eval_episode_num) = self.car.mode_q.get()
            print("mode: ", mode, " train_episode_num: ",  train_episode_num, " eval_episode_num:", eval_episode_num ) 
            if mode == "Train":
                max_steps = 2500
                # traversal_log = train_traversal_log
            elif mode == "Eval": 
                max_steps = 500
                # traversal_log = eval_traversal_log
            else:
                max_steps = 2500
                # traversal_log = full_eval_traversal_log				
        
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        # For T3D
        # action = brain.update(last_reward, last_signal)
        # scores.append(brain.score())
        # rotation = action2rotation[action]
        action_array = self.car.action_q.get()
        rotation = action_array[0]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        
        self.target.pos = (goal_x, goal_y)
        
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3


        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = -0.2
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < last_distance:
                last_reward = 0.1
            # else:
            #     last_reward = last_reward +(-0.2)

        if self.car.x < 5:
            self.car.x = 5
            last_reward = -1
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            last_reward = -1
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -1
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -1

        if distance < 25:
            random_target = random.choice(targets_list)
            while goal_x == random_target[0] and goal_y == random_target[1]:
                random_target = random.choice(targets_list)
            goal_x = random_target[0]
            goal_y = random_target[1]
            
            # if swap == 1:
            #     # goal_x = 1420
            #     # goal_y = 622
            #     goal_x = 1470
            #     goal_y = 70
            #     swap = 0
            # else:
            #     # goal_x = 9
            #     # goal_y = 85
            #     goal_x = 165
            #     goal_y = 227
            #     swap = 1
        last_distance = distance

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
            # img.save("./images/sand.jpg")
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
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):
    ## for T3D
    def __init__(self, start_event, reset_q, mode_q, state_q, action_q, next_state_reward_done_tuple_q):
        super(CarApp, self).__init__()
        self.start_event = start_event
        self.reset_q = reset_q
        self.mode_q = mode_q
        self.state_q = state_q
        self.action_q = action_q
        self.next_state_reward_done_tuple_q = next_state_reward_done_tuple_q

    def build(self):
        parent = Game()
        ## for T3D
        parent.serve_car(self.start_event, self.reset_q, self.mode_q,
                         self.state_q, self.action_q, self.next_state_reward_done_tuple_q)

        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

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
