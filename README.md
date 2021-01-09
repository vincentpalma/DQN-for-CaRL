# DQN-for-CaRL
## Requirements
The requirements are pygame and pytorch:

``pip install pygame``

- https://pytorch.org/get-started/locally/

## Testing the agent
``python play.py``

Hit ``x`` to activate autopilot et ``c`` to deactivate it. 
Hit ``t`` to get a screenshot of the agent's view.

## Variational Autoencoder
The bottleneck is a vector of size 32.
Input image             |  Output image
:-------------------------:|:-------------------------:
![](https://i.imgur.com/0rNKaD1.png)  |  ![](https://i.imgur.com/QT2DJLh.png)

![Structure](https://i.imgur.com/MNvN6GA.png)

## Performance
When trained over 1000 episodes:

![Training Performance](https://i.imgur.com/8gXmBNz.png)

Video:

https://www.youtube.com/watch?v=vL1klvbBcHU

## Credits
Environment & inspiration from: https://github.com/MatthiasSchinzel/SAC-for-CaRL

DQN paper: https://arxiv.org/pdf/1312.5602.pdf
