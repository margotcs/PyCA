
"""
Created on Mon Apr 22 14:36:50 2024

@author: mcost
"""

from Automaton import Automaton
import torch, pygame
from matplotlib.colors import hsv_to_rgb

class CA2D25_04(Automaton):


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
        
        self.infectedWorld = False
        self.reproduction = True
        self.nbChannels = 3
        self.gestationTime = 5
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
        self.world[3:13, :10, 0] = 2
        self.world[200:200 + 10, 100:100 + 10, 0] = 1
        self.world[100:100 + 10, 200:200 + 10, 0] = 2

        # Set a 2x2 region in the center of the first channel to random integer values between 0 and 2
        self.world[self.w//2-1:self.w//2+1, self.h//2-1:self.h//2+1, 0] = torch.randint(0, 3, (2, 2))
        
        # give north directions to all non empty cells 
        setDirMask =  torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        setDirMask[:,:,1] = self.world[:,:,0] != 0
        self.world[setDirMask] = 1
        
        #initialize birth clock to gestationTime for all non empty cells :
        setBirthMask =  torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        setBirthMask[:,:,2] = self.world[:,:,0] != 0
        self.world[setBirthMask] = self.gestationTime

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
       # sw, se = w.roll(1, 1), e.roll(1, 1)
        #nw, ne = w.roll(-1, 1), e.roll(-1, 1)
        
        if self.infectedWorld:
            infection = torch.zeros_like(self.world[:,:,0]);
            infection[w[:,:,0] == 2] = 1
            infection[e[:,:,0] == 2] = 1
            infection[n[:,:,0] == 2] = 1
            infection[s[:,:,0] == 2] = 1 
        
            random_proba = torch.rand(self.h,self.w) #random between 0 and 1 same size as self.world
            infectedNeighboursMask = (infection == 1)
            combinedMask = (random_proba < 1/8) & infectedNeighboursMask
            new_infected = torch.zeros_like(self.world[:,:,0])
            new_infected[combinedMask] = 1
            #visunew_infected = new_infected.numpy()
            maskInfected = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
            maskInfected[:, :, 0] = (new_infected == 1)
            #visuWorld = self.world.numpy()
            self.world[maskInfected] = 2
            #visuWorld = self.world.numpy()
            
     
      #### then, part about directions ####
        # for destinations the filter must have same values on all channel because I want to replace the whole individual
        # and not only change its state infected or not 
        northDestinations = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
        northDestinations[:, :, :] = torch.stack([(self.world[:, :, 0] == 0) & (s[:, :, 1] == 1)] * self.nbChannels, dim=self.nbChannels -1) # le -1 vient du fait qu'il commence a compter les dimensions a zero
        #the line above stack the (h,w) filters to fill up all channels in northDestinations
        #visudesti = northDestinations.numpy()
        northDepartures = northDestinations.roll(1,0)  #+1 roll in the zeroth dimension (my y axis, +1 goes below in the y axis)     
        #visuDepart = northDepartures.numpy()
        
      #### before making the move, find out which cells are pregnant and are going to move, for the birthing part 
        if self.reproduction:
            pregMask0 = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
            pregMask1 = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)
            pregMask2 = torch.zeros(self.h, self.w, self.nbChannels, dtype=torch.bool)

            pregMask0[:,:,0] = (self.world[:,:,2] == 0) & northDepartures[:,:,0] # find all cells that have ended their birth clock
            pregMask1[:,:,1] = (self.world[:,:,2] == 0) & northDepartures[:,:,0]
            pregMask2[:,:,2] = (self.world[:,:,2] == 0) & northDepartures[:,:,0]
            #northDepartures is on all three channels the same, so we can select the first one
            #visuPrefMask = pregMask0.numpy()
        
        visuWorld = self.world.numpy()
      #### make the move 
        self.world[northDestinations] = s[northDestinations]
        self.world[northDepartures] = 0 #empty completely the cell (included direction)

        visuWorld = self.world.numpy()
        
      #### part about birth ##
        if self.reproduction:
            self.world[pregMask0] = 1 #birth of a non infected rodent
            self.world[pregMask1] = 1 #it will want to go north
            self.world[pregMask2] = self.gestationTime #starts far from being pregnant 
            #self.world[pregMask] = torch.tensor([[1, 1, 4]])

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
        
        
        
        
        
        
        
      #### below is thing I tried before 
      #les seuls qui vont avoir des bébés c'est ceux qui bougent, sinon pas de place
      #ils vont pondre a l'endroit d'ou ils viennent de partir donc je peux reutiliser northDepartures
      
      
      
      
        #random_probabilities = torch.rand_like(self.world)
        """
        new_world = torch.zeros_like(self.world)
        infectionMask = (infection == 1)
        random_probabilities = torch.rand_like(self.world)
        combinedMask = (random_probabilities < 1/8) & infectionMask
        new_world[combinedMask] = 1
        #infection[self.world == 0] = 0 #put to zero places where anyway there were no hamster
        """
        #self.world = torch.where(new_world==1,1,0).to(torch.int)

        #self.world = new_world.to(torch.int)
        #self.world = torch.where(infection==1,1,0).to(torch.int)

        # self.world = n
        #self.world = torch.where(self.world==1,self.get_nth_bit(self.s_num,count),self.get_nth_bit(self.b_num,count)).to(torch.int)
        

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