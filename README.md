# 任务说明
本阶段选手在模拟器中驾驶车辆采集数据，通过对数据的训练，实现在模拟器中的自动驾驶
# 规则说明
竞速比赛，选手在训练赛道上进行训练，提交模型后将给出在测试赛道的测试结果（测试赛道未开放）。要求选手在尽可能快的同时减少碰撞。每次碰撞总时间 + 5s，最后以完成总时间最短的选手得分最高
# 提交格式
代码文件result.py 和对应的加载模型，要求包括实现与模拟器的交互和在模拟器中的自动驾驶

# INTRODUCTION

USim is a simulator for autonomous driving developed by UISEE.
USimAPI is an interface library for user program communication with USim.
Typically, through USimAPI, an unmanned driving system could do virtual vehicle controlling and obtain virtual sensors result in the USim
simulation as if it is accessing a real vehicle.
Moreover, We also provide a lot of ground truth of the simulation scene so that users could do AI training
and test based on it.

The version number would be checked with USim at the beginning of the program running.
If not same, API would report error.
If the number is not changed, we promise the api is backward compatibility, i.e. the program with old api could
could work on new USim.
If the number is changed, no compatibility is promised.

# DEPENDENCIES

The Python version USimAPI code need to install the following dependencies:
  * libboost-python-dev
  * python-dev
  * python3-dev
The Python version lib decide to support python3.5 and python2.7 by default.

# TUTORIAL

USimAPI x86 shared Python libraries are provided in the folder 'lib_x86/lib_py27' and 'lib_x86/lib_py35', which support Python2 and Python3 respectively.

In particular, using x86 shared Python lib 'usimpy' need to search libraries 'libusim' and 'usimpy', so please add path of these libraries to environmental variable file:
  * vim ~/.bashrc
  * Append the following contents at the end of the bashrc file
      LD_LIBRARY_PATH=path/lib_dep:$LD_LIBRARY_PATH
      export LD_LIBRARY_PATH
      PYTHONPATH=path/lib_x86/lib_py35(or lib_py27):$PYTHONPATH
      export PYTHONPATH
  * source ~/.bashrc and reboot

Python api demo code is provided in the root directory. The demo code is about how to collect RGB images, actions, collisions information and how to control the vehicle. Demo usage is as follows:
  * Start uisee simulator(USim). Please refer to operation manual.
  * Modify parameters in demo code: replace the ip address in line20 with ip address of machine running uisee simulator(USim), if demo code and uisee simulator(USim) are running on the same machine, the ip adderss is 127.0.0.1
  * Then execute commands: python usim_py_demo.py, then folder 'dataset' is create containing images and actions files

# WINDOWS

if you want to run the code in Windows system, you can copy the file Windows_environment/usimpy.pyd to the python path ../python/Lib/site-packages, and then you can import the library.