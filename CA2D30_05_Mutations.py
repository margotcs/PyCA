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
import h5py

class CA2D30_05_Mutations(Automaton):


    def __init__(self, size, initialisation = 'init1'):
        super().__init__(size)

        #self.s_num = self.get_num_rule(s_num) # Translate string to number form
        #self.b_num = self.get_num_rule(b_num) # Translate string to number form
        self.initialisation = initialisation

        #ca ca initialise avec un 2D world ou chaque element c'est un entier...
        # c'est bien ce que je veux c'est que chaque element puisse etre 0,1 ou 2. 
        #ici il faudra que je rajoute vers ou ils vont etc -> je peux faire un self.directions, un self.birth
        #self.death etc. 

        
        # juste below c'est la ligne de baricelli :
        #self.world = torch.randint(-n_species,n_species+1,(self.h,self.w,2),dtype=torch.int,device=device)
        
        self.infectedWorld = True
        self.reproduction = True
        self.death = True
        self.isMutation = True
        if self.isMutation:
            self.probaMutSpread = 0.001 #between zero and one, if zero no mutation ever
            self.probaMutLifeTime = 0.005#0.1 # same as above, proba that one non empty cell has a different lifetime
        else:
            self.probaMutSpread = 0 #between zero and one, if zero no mutation ever
            self.probaMutLifeTime = 0 # same as above, proba that one non empty cell has a different lifetime
                
        self.lifeMutMax = 250 #maximum value life can take after mutation
        
        self.nbChannels = 5
        self.gestationTime = 1
        self.lifeTimeMax = 20#25
        self.world = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.int)
        self.reset()

        self.infectedCellsNumber = []
        self.healthyCellsNumber = []
        self.timeCounter = 0

    def reset(self):
        """
            Resets the automaton to the initial state.
        """
        self._worldmap = torch.zeros((3,self.h,self.w))
        
        if self.initialisation == 'init1':

            self.world[:, :, 0] = 0  # Initialize the first channel with zeros
    
            # Set specific regions of the first channel to different values
            #self.world[3:13, :10, 0] = 2
            self.world[200:200 + 10, 100:100 + 10, 0] = 1
            self.world[100:100 + 10, 200:200 + 10, 0] = 2
    
            # Set a 2x2 region in the center of the first channel to random integer values between 0 and 2
            #self.world[self.w//2-1:self.w//2+1, self.h//2-1:self.h//2+1, 0] = torch.randint(0, 3, (2, 2))
            
            # Set a 10*10 region to random integers between 0 and 2 
            self.world[5:15, 5:15, 0] = torch.randint(0, 3, (10, 10))
        
        if self.initialisation == 'random':
            self.world[:,:,0] = torch.randint(0,3,(self.h,self.w))
            
        if self.initialisation == 'random_circles':
            self.world[:, :, 0] = 0
            num_circles = random.randint(3,10)
            for _ in range(num_circles):
                # Random center
                center = (random.randint(0, self.w-1), random.randint(0, self.h-1))
                # Random radius
                radius = random.randint(2, 15)  # You can adjust the range of the radius
                # Draw the circle with a random value between 1 and 2 (assuming you want 0 for the background)
                self.draw_circle(center, radius, random.randint(1, 2))
                
        if self.initialisation == 'little_circles':
            self.world[:, :, 0] = 0
            num_circles = 12
            for _ in range(num_circles):
                # Random center
                center = (random.randint(0, self.w-1), random.randint(0, self.h-1))
                # Random radius
                radius = random.randint(3, 5)  # You can adjust the range of the radius
                # Draw the circle with a random value between 1 and 2 (assuming you want 0 for the background)
                self.draw_circle(center, radius, random.randint(1, 2))
        
        if self.initialisation == 'big_circle':
            self.world[:, :, 0] = 0
            # Random center
            center = (random.randint(0, self.w-1), random.randint(0, self.h-1))
            radius = 25  # You can adjust the range of the radius
            # Draw the circle with a random value between 1 and 2 (assuming you want 0 for the background)
            self.draw_circle(center, radius, random.randint(1, 2))
            
        
        #set directions :
        setDirMask =  torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        setDirMask[:,:,1] = self.world[:,:,0] != 0

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
        self.world[setDeathMask] = self.lifeTimeMax   
        
        #initialize life time Max : give at first the normal maximum lifetime 
        #to all non empty cells:
        setLifeTimeMaxMask = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        setLifeTimeMaxMask[:,:,4] = self.world[:,:,0] != 0
        self.world[setLifeTimeMaxMask] = self.lifeTimeMax

    def draw(self):
        """
            Updates the worldmap with the current state of the automaton.
        """

        self._worldmap=self.get_color_world() # (3,H,W)
        
    def get_color_world(self): #vient de Baricelli
        """
            Return colorized sliced world

            Returns : (3,H,W) tensor of floats
        """
    
        colorize=torch.zeros((self.h,self.w,3),dtype=torch.float) # (H,W,3)
        #hsv colors 
        
        # Assign very light green for empty cell 
        colorize[self.world[:, :, 0] == 0] = torch.tensor([152/360, 0.17, 0.9], dtype=torch.float)

        # Assign blue for self.world[:, :, 0] == 1
        colorize[self.world[:, :, 0] == 1] = torch.tensor([174 / 360, 0.94, 0.70], dtype=torch.float)
                                                        
        # Assign red for self.world[:, :, 0] == 2 
        colorize[self.world[:, :, 0] == 2] = torch.tensor([354/360, 0.94, 0.7], dtype=torch.float)
        
        
        # Convert HSV to RGB and permute dimensions
        colorize = torch.tensor(hsv_to_rgb(colorize.cpu().numpy())).permute(2, 0, 1)  # (3, H, W)


        return colorize
    
    def process_event(self, event, camera=None):
        if(event.type == pygame.KEYDOWN):
            if(event.key == pygame.K_DELETE):
                self.reset() 
            """
            if(event.key == pygame.K_n):
                # Picks a random rule
                b_rule = torch.randint(0,2**8,(1,)).item()
                s_rule = torch.randint(0,2**8,(1,)).item()
                self.change_num(s_rule,b_rule)
                print('rule : ', (s_rule,b_rule))
            """
    
    def step(self):
        
        ## first we see if they get infected or not##
        
        # Generate tensors for all 8 neighbors
        n, s = self.world.roll(1, 0), self.world.roll(-1, 0) #ca ca fait haut puis bas 
        w, e = self.world.roll(-1, 1), self.world.roll(1, 1) #ca ca fait gauche puis droite 
        # roll( shift, dimension along which we shift)

        
        if self.infectedWorld:
            
            probaSpreading = 1/8 #if no mutation happens rodent will get infected
            #because of their neighbours with proba 1/8
            
            #at each step there is a very small proba that the 
            #epidemic will change infection rate
            
            random_probaMut = random.random()
            if random_probaMut < self.probaMutSpread:
                probaSpreading = random.random() #we change probaSpreading
                #by a random number between zero and 1 (orinally it's 1/8)
                print("Mutation in virus spreading, new proba = ", probaSpreading)
                    
            infection = torch.zeros_like(self.world[:,:,0]);
            infection[w[:,:,0] == 2] = 1
            infection[e[:,:,0] == 2] = 1
            infection[n[:,:,0] == 2] = 1
            infection[s[:,:,0] == 2] = 1 
            
            visuInfection = infection.numpy()
            visuWorld = self.world.numpy()
            random_proba = torch.rand(self.h,self.w) #random between 0 and 1 same size as self.world
            infectedNeighboursMask = (infection == 1) & (self.world[:,:,0] == 1)
            combinedMask = (random_proba < probaSpreading) & infectedNeighboursMask
            maskInfected = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
            maskInfected[:, :, 0] = combinedMask
            self.world[maskInfected] = 2
            visuWorld = self.world.numpy()
            test = 1
        
        # choose which ones will move
        movingDirection = random.randint(1, 4)
        #before moving : save the life time max of each cell :
        lifeTimeMaxBeforeMoving = self.world[:,:,4].detach().clone();
        
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
       
        else:
            raise ValueError("Something went wrong with movingDirection")
        
      #### part about death ## 
      
        if self.death:
            
            # decrease le life time de 1, pour toutes les non empty cells 
            agingMask = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
            agingMask[:,:,3] = (self.world[:,:,0] != 0) 
            visuWorld = self.world.numpy()
            self.world[agingMask] -= 1
            visuWorld = self.world.numpy()
            if (self.world[:,:,3] < 0).any():
                raise ValueError("Something went wrong with the death mechanism")
            
            #if some cells have their life at zero, make them die.
            visuWorld = self.world.numpy()
            dyingMask = (self.world[:,:,3] == 0) & (self.world[:,:,0] !=0)
            # above mask selects all non empty cells that have zero life time left.
            self.deathMechanism(dyingMask)
      
      #### part about birth ##
        if self.reproduction:
            #les pregMask sont calculés pendant le deplacement, vu que seulement
            #ceux qui bougent ont un bébé
            self.birth(pregMask, lifeTimeMaxBeforeMoving) #function that makes the actual birth

            
            # a la fin : decrease la birth clock de 1, pour toutes les non empty cells 
            # mais si une non empty cell a deja sa birth clock a zero, alors il faut la remettre a gestTime
            diminishBirthClock = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
            diminishBirthClock[:,:,2] = (self.world[:,:,0] != 0) & (self.world[:,:,2] > 0)
            restartBirthClock = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
            restartBirthClock[:,:,2] = (self.world[:,:,0] != 0) & (self.world[:,:,2] <= 0)

            self.world[diminishBirthClock] -= 1
            self.world[restartBirthClock] = self.gestationTime
            
      #### saving how many infected, healthy, etc.
        #if self.timeCounter %10 == 0:
        healthyCells, infectedCells = self.computeWorldCharacteristics()
        self.healthyCellsNumber.append(healthyCells)
        self.infectedCellsNumber.append(infectedCells)
        self.timeCounter += 1
          

    
    def birth(self, pregMask, lifeTimeMaxBeforeMoving):
        
        #from a (h,w) mask, prepare the ones adapted to each channels:
        zeros = torch.zeros(self.h, self.w, dtype=torch.bool)
        pregMask0 = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        pregMask0[:,:,:] = torch.stack([pregMask, zeros, zeros, zeros, zeros], dim=2)
        pregMask1 = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        pregMask1[:,:,:] = torch.stack([zeros, pregMask,zeros, zeros, zeros], dim=2)
        pregMask2 = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        pregMask2[:,:,:] = torch.stack([zeros,zeros, pregMask, zeros, zeros], dim=2)
        pregMask3 = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        pregMask3[:,:,:] = torch.stack([zeros, zeros, zeros, pregMask, zeros], dim=2)
        pregMask4 = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        pregMask4[:,:,:] = torch.stack([zeros, zeros, zeros, zeros, pregMask], dim=2)

        visuPregMask = pregMask.numpy() 
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
        self.world[pregMask3] = lifeTimeMaxBeforeMoving[pregMask]
        
        #set the maximum life time (same as the parent)
        self.world[pregMask4] = lifeTimeMaxBeforeMoving[pregMask]
        
        #### part about mutating life time ##
        
        random_probaMut2 = random.random()
        if random_probaMut2 < self.probaMutLifeTime:
        # ici changer la life time de une des cells 
        #newLife = random.random()*100 # la life max qu'on peut avoir en etant muté
        #c'est 100
            newLife = 200
            #newLife = random.randint(1, self.lifeMutMax)
            nonEmptyMask = self.world[:,:,0] != 0
            trueIndices = torch.nonzero(nonEmptyMask, as_tuple=False)
            trueIndicesList = trueIndices.tolist()
            randomIndices = random.choice(trueIndicesList)
            if self.world[randomIndices[0], randomIndices[1], 0] == 0 :
                raise ValueError("Something went wrong with the lifetime mutation")
            else:
                self.world[randomIndices[0], randomIndices[1], 4] = newLife
                self.world[randomIndices[0], randomIndices[1], 3] = newLife
                print("Mutation on life time, new life = ", newLife)
        
    def deathMechanism(self, dyingMask):
        
        #dyingMask is a (h,w) filter. We want to stack it to fill up all channels 
        dyingMaskFull = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        
        dyingMaskFull[:,:,:] = torch.stack([dyingMask * self.nbChannels], dim=2) 
        #le dim = 2 vient du fait qu'on veut les concatener sur la deuxieme dimension (celle ou j'ai mes channels)
        self.world[dyingMaskFull] = 0 #set values of all channels at zero for the dying cell


    def move(self, shiftedWorld, direction):
          #movement of rodents
          # for destinations the filter must have same values on all channel because I want to replace the whole individual
          # and not only change its state infected or not 
          destinations = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
          destinations[:, :, :] = torch.stack([(self.world[:, :, 0] == 0) & (shiftedWorld[:, :, 1] == direction)] * self.nbChannels, dim=2) 
          #le dim = 2 vient du fait qu'on veut les concatener sur la deuxieme dimension (celle ou j'ai mes channels)
          #the line above stack the (h,w) filters to fill up all channels in northDestinations
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
       
        #### make the move 
          self.world[destinations] = shiftedWorld[destinations]
          self.world[departures] = 0 #empty completely the cell (included direction)
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
            
    def computeWorldCharacteristics(self):
        healthyMask = (self.world[:,:,0] == 1)
        healthyCells = torch.sum(healthyMask).item()
        infectedMask = (self.world[:,:,0] == 2)
        infectedCells = torch.sum(infectedMask).item()
        emptyMask = (self.world[:,:,0] == 0)
        emptyCells = torch.sum(emptyMask).item()
        if ( (emptyCells + infectedCells + healthyCells) != self.w*self.h):
            raise ValueError("Some cells are neither infected healthy or empty")
        return healthyCells, infectedCells
       
                
    def print_world_characteristics(self):
         # Print some characteristics of self.world
         #print("Number of infected:", self.n)
         
         healthyCells, infectedCells = self.computeWorldCharacteristics()
         print("Number of healthy cells:", healthyCells)
         print("Number of infected cells:", infectedCells)
         emptyMask = (self.world[:,:,0] == 0)
         emptyCells = torch.sum(emptyMask).item()
         print("Time elapsed = ", self.timeCounter)
         if (emptyCells + infectedCells + healthyCells != self.w*self.h):
             raise ValueError("Some cells are neither infected healthy or empty")
         else:
             print("Total number of cells = ", emptyCells + infectedCells + healthyCells)
                 
         """
         print(f"Shape: {self.world.shape}")
         print(f"Max value: {torch.max(self.world)}")
         print(f"Min value: {torch.min(self.world)}")
         print(f"Mean value: {torch.mean(self.world.float())}")
         """
    
    def saveDatas(self):   
        with h5py.File(f'Results/AllTimesmut_{self.isMutation}_lifeTime_{self.lifeTimeMax}.h5', 'w') as hf:
            hf.create_dataset('infectedCellsNumber', data=self.infectedCellsNumber)
            hf.create_dataset('healthyCellsNumber', data=self.healthyCellsNumber)
            hf.create_dataset('totalCells', data = self.h*self.w)
            hf.create_dataset('gestationTime', data = self.gestationTime)
            hf.create_dataset('lifeTimeMax', data = self.lifeTimeMax)
    
    def draw_circle(self, center, radius, value):
        """
        Draw a circle on the given world tensor.
        
        Parameters:
        world (torch.Tensor): The tensor to draw on.
        center (tuple): The (x, y) coordinates of the circle's center.
        radius (int): The radius of the circle.
        value (int): The value to fill inside the circle.
        """
        world = self.world[:,:,0]
        for y in range(max(0, center[1] - radius), min(world.shape[0], center[1] + radius + 1)):
            for x in range(max(0, center[0] - radius), min(world.shape[1], center[0] + radius + 1)):
                if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2:
                    world[y, x] = random.randint(0,2)
                
                
                