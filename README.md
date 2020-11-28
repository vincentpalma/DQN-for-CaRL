# DQN-for-CaRL
## Requirements
requirements are pygame, pytorch (for the DQN) and tensorflow (for the CAE):

``pip install pygame``

- https://pytorch.org/get-started/locally/

``pip install tensorflow``

## Testing the agent
``python play.py``

Hit ``x`` to activate autopilot et ``c`` to deactivate it. 
Hit ``t`` to get a screenshot of the agent's view.
## Performance
when trained over 600 episodes:

![Training Performance](https://i.imgur.com/HUysRrE.png)

video:

https://www.youtube.com/watch?v=vL1klvbBcHU

## Credits
environment and autoencoder from: https://github.com/MatthiasSchinzel/SAC-for-CaRL

DQN paper: https://arxiv.org/pdf/1312.5602.pdf
