# Welcome to Robot Object Search for CSE598 Perception in Robotics

This tutorial should help guide you through installation and setup of the project, as well as help you to locate relevant resources.
Additionally project requirements can be found at the bottom of this document.


## Binary Download
The binary for this project can be found via the following link: 


### Documentation

For Airsim documentation see reference to [detailed documentation](https://microsoft.github.io/AirSim/) on all aspects of AirSim.

The recommended reinforcement learning framework to use is RLlib. Other alternatives include Stable Baselines 3 and USC garage. Following are the links
to those projects:
RLlib: https://docs.ray.io/en/releases-1.5.1/rllib.html#:~:text=RLlib%20is%20an%20open%2Dsource,its%20internals%20are%20framework%20agnostic.
Stable Baselines 3: https://github.com/DLR-RM/stable-baselines3
USC garage: https://github.com/rlworkgroup/garage


### Installation

From the base project direction perform the following command:
pip install -r requirements.txt

Additionally you may need to also perform the following commands after the above pip install:

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge or equivalent pip install of pytorch with appropriate cuda version
for your system

## Participate

### Paper

If students use this environment in a publication the following citations must be present:
```
RO specific framework ASU APG reference here WIP
```

```
@inproceedings{airsim2017fsr,
  author = {Shital Shah and Debadeepta Dey and Chris Lovett and Ashish Kapoor},
  title = {AirSim: High-Fidelity Visual and Physical Simulation for Autonomous Vehicles},
  year = {2017},
  booktitle = {Field and Service Robotics},
  eprint = {arXiv:1705.05065},
  url = {https://arxiv.org/abs/1705.05065}
}
```

### Contribute

Please reach out to your project coordinator if you are interest in working with the Active Perception Group in research pertaining to robot object search
or visual navigation.


### External Project References?

TBD


## Contact

For issues pertaining to this project reach out to your project coordinator.


## What's New

TBD

## FAQ

If you run into problems, check the [FAQ](https://microsoft.github.io/AirSim/faq) and feel free to post issues in the  [AirSim](https://github.com/Microsoft/AirSim/issues) repository.


## License

This project is released under the MIT License. Please see reference to the [License file](LICENSE) file for more details.










