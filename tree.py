###DECISION TREE AGENT "tree"
import numpy as np
from sklearn import tree as ml_tree

#SET THESE PARAMS
gather_steps = 25 #how many steps to gather info before tree kicks in | MUST BE >= 2
lookback_window = 8 #how many steps become training features | MUST BE <= gather_steps - 1

#gather input features for deicison tree model
opps = []
ours = []
X = None
Y = None
D = 2*lookback_window + 3 #features are each time step in lookback (2x for ours and opps) plus 3 location markers
#print(X)
pred_input = None

#not an agent, returns move that will defeat specified move
def defeat(choice):
    if choice == 0:
        return 1
    elif choice == 1:
        return 2
    elif choice == 2:
        return 0

def init_features():
    print("Initializing features!")
    global X
    global Y
    global D
    
    N = (len(opps) - lookback_window + 1) - 1 #number of training samples we can make given current observations
    #first part is convolutional arithmetic, last "-1" is because we need to know one more step in advance for defeat(opps[])
    print("Making "+str(N)+" samples.")
    X = np.zeros((N,D))
    Y = np.zeros((N))
    
    for n in range(N):
        X[n,:lookback_window] = opps[n:n+lookback_window]
        #print(X[n])
        X[n,lookback_window:2*lookback_window] = ours[n:n+lookback_window]
        #print("ayee")
        #print(X[n])
        top_index = n+lookback_window-1
        #print("top_index ="+str(top_index))
        X[n,2*lookback_window:] = np.array([top_index%2,top_index%3,top_index%5])
        #print(X[n])
        
        Y[n] = defeat(opps[n+lookback_window])
    
    update_features(False)
    print("Done intializing. Shape:")
    print(X.shape)

def update_features(concat=True):
    global X
    global Y
    global D
    global pred_input
    
    if concat:
        print("Appending old prediction")
        #print(pred_input)
        tempY = np.zeros((1))
        tempY[0] = defeat(opps[-1])
        
        X = np.vstack((X,pred_input))
        Y = np.hstack((Y,tempY))
    
    temp = np.zeros((1,D))
    
    temp[0,:lookback_window] = opps[-lookback_window:]
    temp[0,lookback_window:2*lookback_window] = ours[-lookback_window:]
    
    top_index = len(opps)-1
    #print("top_index ="+str(top_index))
    temp[0,2*lookback_window:] = np.array([top_index%2,top_index%3,top_index%5])
    
    pred_input = temp
    
    #print(X)
    #print("Added the sample:")
    #print(temp)
    
    
last_move = -1
initialized = False

def tree(observation, configuration):
    global last_move
    global gather_steps
    global initialized
    global pred_input
    global X
    global Y
    
    print("CURRENT STEP:"+str(observation.step))
    
    if observation.step > 0:
        opps.append(observation.lastOpponentAction)
        ours.append(last_move)
        #print(opps)
        #print(ours)
        #print(len(opps))
        #print(len(ours))
    
    if observation.step <= gather_steps: #beginning strategy
        move = np.random.randint(3) #gather strategy is random
    else: #regular algorithm
        if initialized == False:
            initialized = True
            init_features()
        else:
            update_features()
        
        model = ml_tree.DecisionTreeClassifier()
        model.fit(X,Y)
        move = int(model.predict(pred_input.reshape(1,-1))[0])
        #print(move)
        #print(X)
        #print(Y)
        #print(pred_input)
        #move = 0
        
        
        
    last_move = move
    return move