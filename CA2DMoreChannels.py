# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:03:49 2024

@author: mcost
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:36:50 2024

@author: mcost
"""

from Automaton import Automaton
import torch, pygame
from matplotlib.colors import hsv_to_rgb

class CA2DMoreChannels(Automaton):


    def __init__(self, size, random=False):
        super().__init__(size)

        #self.s_num = self.get_num_rule(s_num) # Translate string to number form
        #self.b_num = self.get_num_rule(b_num) # Translate string to number form
        self.random = random

        self.world = torch.zeros((self.h,self.w),dtype=torch.int)
        #ca ca initialise avec un 2D world ou chaque element c'est un entier...
        # c'est bien ce que je veux c'est que chaque element puisse etre 0,1 ou 2. 
        #ici il faudra que je rajoute vers ou ils vont etc -> je peux faire un self.directions, un self.birth
        #self.death etc. 
        self.directions = torch.zeros((self.h,self.w),dtype=torch.int)
        #ca sera la future direction que prend chaque cell associé. 1: north, 2: east
        #3 : south, 4: west
        
        # juste below c'est la ligne de baricelli :
        #self.world = torch.randint(-n_species,n_species+1,(self.h,self.w,2),dtype=torch.int,device=device)

        
        self.reset()


    def reset(self):
        """
            Resets the automaton to the initial state.
        """
        self._worldmap = torch.zeros((3,self.h,self.w))

        
        
        self.world = torch.zeros_like(self.world,dtype=torch.int)
        self.world[3:13, :10] = 1 #ca c'est moi qui l'ai rajoute pour voir qqchose

        self.world[200:200 + 10, 100:100 + 10] = 1
        self.world[100:100 + 10, 200:200 + 10] = 2

        self.world[self.w//2-1:self.w//2+1,self.h//2-1:self.h//2+1]=torch.randint(0,3,(2,2))
        
        self.directions = torch.ones_like(self.directions, dtype=torch.int)
        
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
        colorize[...,0]=torch.abs(self.world)/3
        colorize[...,1]=torch.abs(self.world)/3
        colorize[...,2]=torch.abs(self.world)/3
        
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

        infection = torch.zeros_like(self.world);
        infection[w == 2] = 1
        infection[e == 2] = 1
        infection[n == 2] = 1
        infection[s == 2] = 1 
        
        random_proba = torch.rand(self.h,self.w) #random between 0 and 1 same size as self.world
        infectionMask = (infection == 1)
        combinedMask = (random_proba < 1/8) & infectionMask
        new_infected = torch.zeros_like(self.world)
        new_infected[combinedMask] = 1
        maskAllInfected = (new_infected == 1) | (self.world == 2) #taking into account cells that are already infected
        self.world[maskAllInfected] = 2
        #self.world = new_world.to(torch.int)
        
        ## then, part about directions ##
        
        northMask = (self.directions == 1)
        eastMask = (self.directions == 2)
        southMask = (self.directions == 3)
        westMask = (self.directions == 4)
        
        #chercher les destinations : 
        #destinationNort = (self.world == 0) & s.
        
        
        
        #en fait ce que je fais below c'est que je trouve les cases départ mais je devrai chercher les destinations 
        #freeNorth = (n == 0) & northMask & (self.world != 0)
        moveToNorth = (n == 0) & northMask & self.world == 1
        #visuMoveNorth = moveToNorth.numpy()
        #visuWorld = self.world.numpy()
        self.world[moveToNorth] = s[moveToNorth] #on remplace le monde par celui du dessous 
        #visuWorld = self.world.numpy()
        #test = 1
        
        """
        # Get the indices where northMask is True
        indicesNorthMask = torch.nonzero(northMask)
        # Reshape the indices to match the original shape of self.directions
        indicesNorthMask = indicesNorthMask.view(-1, 2)
        # Use the indices to index the tensor w
        freeNorth = w[indicesNorthMask[:, 0], indicesNorthMask[:, 1]] == 0
        
        #tester si l'endroit est libre (pour l'instant que pour north)
        #freeNorth = w[northMask] == 0

        #freeNorth = (w[northMask] == 0) #verifier avec chat gpt que w s'est bien vers le haut 
        #now we have our mask of cell that will move to the north
        self.world[freeNorth] = w #now cells in the north that were free and where they wanted to move are replaced.
        # but need to empy the departure cell : il suffit de prendre le freenorth mask et de le shift 
        # de (-1, 0)
        """



        
        
        
        ## end directions, below is things I tried before : ##
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