from jointCtlComp import *
from taskCtlComp import *

# Controller in the joint space. The robot has to reach a fixed position.
# Choices are 'P', 'PD', 'PID', 'PD_Grav', 'ModelBased'
# jointCtlComp(['ModelBased'], True)
# Same controller, but this time the robot has to follow a fixed trajectory.
jointCtlComp(['ModelBased'], False)

# Controller in the task space.
# taskCtlComp(['JacNullSpace'],resting_pos=np.mat([0, pi]).T, pauseTime=0.0001)

input('Press Enter to close')
