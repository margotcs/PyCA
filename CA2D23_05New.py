# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:47:35 2024

@author: mcost
"""


"""
Created on Mon Apr 22 14:36:50 2024

@author: mcost
"""

from Automaton import Automaton
import torch, pygame, random
from matplotlib.colors import hsv_to_rgb

class CA2D23_05New(Automaton):


    def __init__(self, size, random=False):
        super().__init__(size)

        #self.s_num = self.get_num_rule(s_num) # Translate string to number form
        #self.b_num = self.get_num_rule(b_num) # Translate string to number form
        self.random = random

        #ca ca initialise avec un 2D world ou chaque element c'est un entier...
        # c'est bien ce que je veux c'est que chaque element puisse etre 0,1 ou 2. 
        #ici il faudra que je rajoute vers ou ils vont etc -> je peux faire un self.directions, un self.birth
        #self.death etc. 

        
        # juste below c'est la ligne de baricelli :
        #self.world = torch.randint(-n_species,n_species+1,(self.h,self.w,2),dtype=torch.int,device=device)
        
        self.infectedWorld = True
        self.reproduction = True
        self.death = True
        self.nbChannels = 4
        self.gestationTime = 1
        self.lifeTime = 25
        self.world = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.int)
        self.reset()


    def reset(self):
        """
            Resets the automaton to the initial state.
        """
        self._worldmap = torch.zeros((3,self.h,self.w))

        #self.world[:, :, 1] = 1 #this puts ones on the second channel (the one corresponding to directions)

        self.world[:, :, 0] = 0  # Initialize the first channel with zeros

        # Set specific regions of the first channel to different values
        #self.world[3:13, :10, 0] = 2
        self.world[200:200 + 10, 100:100 + 10, 0] = 1
        self.world[100:100 + 10, 200:200 + 10, 0] = 2

        # Set a 2x2 region in the center of the first channel to random integer values between 0 and 2
        #self.world[self.w//2-1:self.w//2+1, self.h//2-1:self.h//2+1, 0] = torch.randint(0, 3, (2, 2))
        
        # Set a 10*10 region to random integers between 0 and 2 
        self.world[5:15, 5:15, 0] = torch.randint(0, 3, (10, 10))
        
        #set directions :
        setDirMask =  torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        setDirMask[:,:,1] = self.world[:,:,0] != 0
        """
        # give north directions to all non empty cells 
        
        self.world[setDirMask] = 2
        """
        random_directions = torch.randint(1, 5, (self.h, self.w, self.nbChannels))
        random_directions = random_directions.to(self.world.dtype)
        self.world[setDirMask] = random_directions[setDirMask]
        
        
        #initialize birth clock to gestationTime for all non empty cells :
        setBirthMask =  torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        setBirthMask[:,:,2] = self.world[:,:,0] != 0
        self.world[setBirthMask] = self.gestationTime
        
        #initialize death clock : give lifeTime to all non empty cells : 
        setDeathMask =  torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        setDeathMask[:,:,3] = self.world[:,:,0] != 0
        self.world[setDeathMask] = self.lifeTime   
            
        """#truc de base : 
        if(self.random):
            self.world = self.get_init_mat(0.5)
        else:
            self.world = torch.zeros_like(self.world,dtype=torch.int)
            self.world[self.w//2-1:self.w//2+1,self.h//2-1:self.h//2+1]=torch.ranint(0,2,(2,2))
        """
    def draw(self):
        """
            Updates the worldmap with the current state of the automaton.
        """
        # ce qu'il y a de base : 
        #self._worldmap = self.world[None,:,:].expand(3,-1,-1).to(dtype=torch.float)
        
        #ca je l'ai pris de Baricelli : 
        self._worldmap=self.get_color_world() # (3,H,W)
        
    def get_color_world(self): #vient de Baricelli
        """
            Return colorized sliced world

            Returns : (3,H,W) tensor of floats
        """
    
        colorize=torch.zeros((self.h,self.w,3),dtype=torch.float) # (H,W,3)
        #attention c'est en hsv
        colorize[..., 0] = torch.abs(self.world[:, :, 0]) / 3
        colorize[..., 1] = torch.abs(self.world[:, :, 0]) / 3
        colorize[..., 2] = torch.abs(self.world[:, :, 0]) / 3

        
        # Green channel: Assign intensity based on the value of self.world
        """
        colorize[..., 0] = torch.where(self.world == 1, 0.5, 0.0)  # Assign 1.0 for value 2, 0.0 otherwise
        colorize[..., 1] = torch.where(self.world == 1, 1.0, 0.0)
        colorize[..., 2] = torch.where(self.world == 1, 1.0, 0.0)  # Assign 1.0 for value 0, 0.0 otherwise

        # Blue channel: Assign intensity based on the value of self.world
        colorize[..., 0] = torch.where(self.world == 0, 0.0, 0.0)  # Assign 1.0 for value 2, 0.0 otherwise
        colorize[..., 1] = torch.where(self.world == 0, 0.0, 0.0)
        colorize[..., 2] = torch.where(self.world == 0, 0.0, 0.0)  # Assign 1.0 for value 0, 0.0 otherwise
        """
        # Convert HSV to RGB and permute dimensions
        colorize = torch.tensor(hsv_to_rgb(colorize.cpu().numpy())).permute(2, 0, 1)  # (3, H, W)


        return colorize
    
    def process_event(self, event, camera=None):
        if(event.type == pygame.KEYDOWN):
            if(event.key == pygame.K_DELETE):
                self.reset() 
            if(event.key == pygame.K_n):
                # Picks a random rule
                b_rule = torch.randint(0,2**8,(1,)).item()
                s_rule = torch.randint(0,2**8,(1,)).item()
                self.change_num(s_rule,b_rule)
                print('rule : ', (s_rule,b_rule))

    def change_num(self,s_num : int, b_num : int):
        """
            Changes the rule of the automaton to the one specified by s_num and b_num
        """
        self.s_num = s_num
        self.b_num = b_num
        self.reset()
    
    def step(self):
        
        ## first we see if they get infected or not##
        
        # Generate tensors for all 8 neighbors
        n, s = self.world.roll(1, 0), self.world.roll(-1, 0) #ca ca fait haut puis bas 
        w, e = self.world.roll(-1, 1), self.world.roll(1, 1) #ca ca fait gauche puis droite 
        # roll( shift, dimension along which we shift)
        # sw, se = w.roll(1, 1), e.roll(1, 1)
        #nw, ne = w.roll(-1, 1), e.roll(-1, 1)
        
        if self.infectedWorld:
            infection = torch.zeros_like(self.world[:,:,0]);
            infection[w[:,:,0] == 2] = 1
            infection[e[:,:,0] == 2] = 1
            infection[n[:,:,0] == 2] = 1
            infection[s[:,:,0] == 2] = 1 
            
            visuInfection = infection.numpy()
            visuWorld = self.world.numpy()
            random_proba = torch.rand(self.h,self.w) #random between 0 and 1 same size as self.world
            infectedNeighboursMask = (infection == 1) & (self.world[:,:,0] == 1)
            visuNeighbourInfect = infectedNeighboursMask.numpy()
            combinedMask = (random_proba < 1/8) & infectedNeighboursMask
            visuCombinedMask = combinedMask.numpy()
            #new_infected = torch.zeros_like(self.world[:,:,0])
            #new_infected[combinedMask] = 1
            #visunew_infected = new_infected.numpy()
            maskInfected = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
            maskInfected[:, :, 0] = combinedMask
            #visuWorld = self.world.numpy()
            self.world[maskInfected] = 2
            visuWorld = self.world.numpy()
            test = 1
        
        # choose which ones will move
        movingDirection = random.randint(1, 4)
        #movingDirection = 2;
        if movingDirection == 1:
            #move to the north :
            
            shiftedWorld = self.world.roll(-1,0) # south tensor
            pregMask = self.move(shiftedWorld, 1)
        elif movingDirection == 2:
            #move to the east :
            shiftedWorld = self.world.roll(1,1) #roll +1 in the dimension 1: roll to the right. Like this at the same emplacement we see what's on the west
            pregMask = self.move(shiftedWorld, 2)
   
        elif movingDirection == 3:
            #move to the south
            shiftedWorld = self.world.roll(1,0) # north tensor
            pregMask = self.move(shiftedWorld, 3)
            
        elif movingDirection == 4:
            #move to the west :
            shiftedWorld = self.world.roll(-1,1) #roll -1 in the dimension 1: roll to the left. Like this at the same emplacement we see what's on the east
            pregMask = self.move(shiftedWorld, 4)
   
           #visuWorld = self.world.numpy()
            
        else:
            raise ValueError("Something went wrong with movingDirection")
            
            
            
        """ #old move
        # move to north :
        s = self.world.roll(-1, 0) #need to redo it because now infection happened
        # en fait ca devrait pas etre (1,0) le s ? ...
        direction = 1
        pregMask0, pregMask1, pregMask2 = self.moveOld(s, direction) 
        """
      #### part about death ## 
      
        if self.death:
            
            # decrease le life time de 1, pour toutes les non empty cells 
            agingMask = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
            agingMask[:,:,3] = (self.world[:,:,0] != 0) 
            self.world[agingMask] -= 1
            if (self.world[:,:,3] < 0).any():
                raise ValueError("Something went wrong with the death mechanism")
            
            #if some cells have their life at zero, make them die.
            visuWorld = self.world.numpy()
            dyingMask = (self.world[:,:,3] == 0) & (self.world[:,:,0] !=0)
            # above mask selects all non empty cells that have zero life time left.
            visudyingMask = dyingMask.numpy()
            self.deathMechanism(dyingMask)
            
      #### part about birth ##
        if self.reproduction:
            #les pregMask sont calculés pendant le deplacement, vu que seulement
            #ceux qui bougent ont un bébé
            self.birth(pregMask) #function that makes the actual birth

            visuWorld = self.world.numpy()
            # a la fin : decrease la birth clock de 1, pour toutes les non empty cells 
            # mais si une non empty cell a deja sa birth clock a zero, alors il faut la remettre a gestTime
            diminishBirthClock = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
            diminishBirthClock[:,:,2] = (self.world[:,:,0] != 0) & (self.world[:,:,2] > 0)
            restartBirthClock = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
            restartBirthClock[:,:,2] = (self.world[:,:,0] != 0) & (self.world[:,:,2] <= 0)

            visudimin = diminishBirthClock.numpy()
            visuRestart = restartBirthClock.numpy()
            self.world[diminishBirthClock] -= 1
            self.world[restartBirthClock] = self.gestationTime
            #self.world[birthClockMask] = self.world[birthClockMask] -1 
            visuWorld = self.world.numpy()
            test = 1
        
        
            
    
    
    def birth(self, pregMask):
        
        #from a (h,w) mask, prepare the ones adapted to each channels:
        zeros = torch.zeros(self.h, self.w, dtype=torch.bool)
        pregMask0 = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        pregMask0[:,:,:] = torch.stack([pregMask, zeros, zeros, zeros], dim=2)
        pregMask1 = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        pregMask1[:,:,:] = torch.stack([zeros, pregMask,zeros, zeros], dim=2)
        pregMask2 = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        pregMask2[:,:,:] = torch.stack([zeros,zeros, pregMask, zeros], dim=2)
        pregMask3 = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        pregMask3[:,:,:] = torch.stack([zeros, zeros, zeros, pregMask], dim=2)

            
        # set the type of the rodent which will be bornt
        self.world[pregMask0] = 1 #birth of a non infected rodent
        #set its direction
        random_directions = torch.randint(1, 5, (self.h, self.w, self.nbChannels))
        random_directions = random_directions.to(self.world.dtype)
        self.world[pregMask1] = random_directions[pregMask1]
        
        #set its birth clock (starts far from being pregnant)
        self.world[pregMask2] = self.gestationTime 
        
        #build the lifetime mask from pregMaskTest :
        visupregMask3 = pregMask3.numpy()
        self.world[pregMask3] = self.lifeTime
        
    def deathMechanism(self, dyingMask):
        
        #dyingMask is a (h,w) filter. We want to stack it to fill up all channels 
        dyingMaskFull = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        
        dyingMaskFull[:,:,:] = torch.stack([dyingMask * self.nbChannels], dim=2) 
        #le dim = 2 vient du fait qu'on veut les concatener sur la deuxieme dimension (celle ou j'ai mes channels)
        visudyingMaskFull = dyingMaskFull.numpy()
        self.world[dyingMaskFull] = 0 #set values of all channels at zero for the dying cell
        visuWorld = self.world.numpy()
        test=1


    def move(self, shiftedWorld, direction):
          #movement of rodents
          # for destinations the filter must have same values on all channel because I want to replace the whole individual
          # and not only change its state infected or not 
          destinations = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
          destinations[:, :, :] = torch.stack([(self.world[:, :, 0] == 0) & (shiftedWorld[:, :, 1] == direction)] * self.nbChannels, dim=2) 
          #le dim = 2 vient du fait qu'on veut les concatener sur la deuxieme dimension (celle ou j'ai mes channels)
          #the line above stack the (h,w) filters to fill up all channels in northDestinations
          visudesti = destinations.numpy()
          visuDepart = shiftedWorld.numpy()
          visuWorld = self.world.numpy()
          if direction == 1:
              departures = destinations.roll(1,0)  #+1 roll in the zeroth dimension (my y axis, +1 goes below in the y axis)     
          elif direction == 2:
              departures = destinations.roll(-1,1)
          elif direction == 3:
              departures = destinations.roll(-1,0)
          elif direction == 4:
              departures = destinations.roll(1,1) #+1 roll in the 1st dimension (my x axis, +1 goes to the right)
          visuDepart = departures.numpy()
          
        #### before making the move, find out which cells are pregnant and are going to move, for the birthing part 
       
          if self.reproduction:
              #la on check que ils sont au bout de la birth clock, et que ils vont partir :
            
              pregMask = (self.world[:,:,2] == 0) & departures[:,:,0]
              
              #northDepartures is on all three channels the same, so we can select the first one
              #visuPrefMask = pregMask0.numpy()
              
          visuWorld = self.world.numpy()
        #### make the move 
          #visuDestinations = northDestinations.numpy()
          #visuS = s.numpy()
          self.world[destinations] = shiftedWorld[destinations]
          self.world[departures] = 0 #empty completely the cell (included direction)
          visuWorld = self.world.numpy()
          test = 1
          
          return pregMask
      
    def get_init_mat(self,rand_portion):
        """
            Get initialization matrix for CA

            Params : 
            rand_portion : float, portion of the screen filled with noise.
        """
        batched_size = torch.tensor([self.h,self.w])
        randsize = (batched_size*rand_portion).to(dtype=torch.int16) # size of the random square
        randstarts = (batched_size*(1-rand_portion)/2).to(dtype=torch.int16) # Where to start the index for the random square

        randsquare = torch.where(torch.randn(*randsize.tolist())>0,1,0) # Creates random square of 0s and 1s

        init_mat = torch.zeros((self.h,self.w),dtype=torch.int16)
        init_mat[randstarts[0]:randstarts[0]+randsize[0],
        randstarts[1]:randstarts[1]+randsize[1]] = randsquare
        init_mat = init_mat.to(torch.int16)


        return init_mat # (B,H,W)