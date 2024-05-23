# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:01:47 2024

@author: mcost
"""

from Automaton import Automaton
import torch, pygame
from matplotlib import pyplot as plt



class NewCATry(Automaton):

    def __init__(self, size):
        super().__init__(size) #j'utilise l'initialisation de la super class automaton
        self.world = torch.zeros((self.h,self.w),dtype=torch.int) #ca ca initialise avec un 2D world ou chaque element c'est un entier...
        # c'est bien ce que je veux c'est que chaque element puisse etre 0,1 ou 2. 
        #ici il faudra que je rajoute vers ou ils vont etc
        self.reset()
        
    def step(self):
        # Generate tensors for all 8 neighbors
        w, e = self.world.roll(-1, 0), self.world.roll(1, 0) 
        n, s = self.world.roll(-1, 1), self.world.roll(1, 1) 
        sw, se = w.roll(1, 1), e.roll(1, 1)
        nw, ne = w.roll(-1, 1), e.roll(-1, 1)

        #count = w + e + n + s + sw + se + nw + ne
        #mtn ca c'est un nouveau tensor qui prend en compte tous les voisins.
        #si je fais un tensor qui est soit un si un de ces neighbours tensors est infecté, 
        #zero otherwise 
        infection = torch.zeros_like(w);
        infection[w == 1] = 1
        infection[e == 1] = 1
        infection[n == 1] = 1
        infection[s == 1] = 1 
        infection[self.world == 0] = 0 #put to zero places where anyway there were no hamster
        
        self.world = 1
        #self.world[infection == 1] = 1
        #self.world = torch.where(infection==1,3,0).to(torch.int)
        # if infection is one at this point, the cell will become infected (value goes to 3), otherwise it stays the same.
        #self.world = torch.where(self.world==1,self.get_nth_bit(self.s_num,count),self.get_nth_bit(self.b_num,count)).to(torch.int)

    def reset(self): #elle je l'ai pas modifié
        """
            Resets the automaton to the initial state.
        """
        self._worldmap = torch.zeros((3,self.h,self.w))
        
        """ truc d'avant :
        if(self.random):
            self.world = self.get_init_mat(0.5)
        else:
            self.world = torch.zeros_like(self.world,dtype=torch.int)
            self.world[self.w//2-1:self.w//2+1,self.h//2-1:self.h//2+1]=torch.ranint(0,2,(2,2))
        """
        self.world = torch.zeros_like(self.world,dtype=torch.int)
        self.world[self.w//2-1:self.w//2+1,self.h//2-1:self.h//2+1]=torch.randint(0,2,(2,2))
        
        # de base c'etait la ligne du dessous : 
        #self.world[self.w//2-1:self.w//2+1,self.h//2-1:self.h//2+1]=torch.randint(0,2,(2,2))
        
    def draw(self): #elle j'ai rien changé, peut etre mettre celle de Baricellli vu que j'ai 3 states 
        """
            Updates the worldmap with the current state of the automaton.
        """
        self._worldmap = self.world[None,:,:].expand(3,-1,-1).to(dtype=torch.float)
    
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
    