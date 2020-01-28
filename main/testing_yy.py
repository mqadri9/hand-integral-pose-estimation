import torch
import numpy as np 

a = np.array([[ 2.3843131920e-02, -3.6213150861e-03, -2.1338857335e+02],
        [ 1.3294972696e-02, -1.1649455648e-02, -2.1338073126e+02],
        [ 8.1459008216e-03, -1.5183025427e-02, -2.1337059646e+02],
        [ 2.1955090625e-03, -1.2757677170e-02, -2.1336205133e+02],
        [-4.7255186505e-03, -1.1727382299e-02, -2.1334943876e+02],
        [ 2.3734214598e-03, -2.6846884535e-03, -2.1336005276e+02],
        [-6.0654221520e-03,  1.1012706911e-03, -2.1335119854e+02],
        [-1.3904996045e-02,  3.3917189280e-03, -2.1334737212e+02],
        [-2.2490197397e-02,  6.3371786318e-03, -2.1334224551e+02],
        [-3.8902637847e-04,  6.2339883834e-03, -2.1336153614e+02],
        [-1.0827586350e-02,  9.8308275947e-03, -2.1335590036e+02],
        [-1.9555730071e-02,  1.1965284571e-02, -2.1335349143e+02],
        [-2.8604665654e-02,  1.5052100476e-02, -2.1334855767e+02],
        [ 3.7693319046e-04,  1.2660308350e-02, -2.1336931668e+02],
        [-9.8797671926e-03,  1.5428022923e-02, -2.1336490630e+02],
        [-1.9128399781e-02,  1.8321378556e-02, -2.1336293772e+02],
        [-2.8651371464e-02,  2.1174227228e-02, -2.1335916047e+02],
        [ 4.6432180656e-04,  1.5992449849e-02, -2.1337623252e+02],
        [-7.4806303741e-03,  1.8267313042e-02, -2.1337338020e+02],
        [-1.4721974600e-02,  2.0007982993e-02, -2.1337126086e+02],
        [-2.1885995476e-02,  2.2375871808e-02, -2.1336734927e+02]])
        

b = torch.from_numpy(a).double().cuda()

print(b - b.mean(1, keepdims=True))