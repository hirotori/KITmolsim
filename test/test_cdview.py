from kitmolsim.writer import write_cdview
import numpy as np


pos = np.array(
    [[0,0  ,0  ],
     [0,1.0,0  ],
     [0,  0,1.0],
     [0,1.0,1.0]]
)

radius = np.array([0.1,0.2,0.3,0.4])
colors = np.array(
    [[255,0,0],
     [255,255,0],
     [255,0,255],
     [0,255,255]], dtype="i4")
atype = np.array([0,1,2,3])
box_s = (0,0,0)
box_e = (1,1,1)

write_cdview.write_cdv("test.cdv", pos, atype, box_s, box_e, radius=radius, color=colors)