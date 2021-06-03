% % Problem Definition

model = SelectModel()
% Select Model

CostFunction = @(xhat) MyCost(xhat, model)
% Cost Function

VarSize = [model.N model.N]
% Decision Variables Matrix Size

nVar = prod(VarSize)
% Number of Decision Variables

VarMin = 0
% Lower Bound of Decision Variables
VarMax = 1
% Upper Bound of Decision Variables


% % PSO Parameters

MaxIt = 250
% Maximum Number of Iterations

nPop = 150
% Population Size(Swarm Size)

w = 1
% Inertia Weight
wdamp = 0.99
% Inertia Weight Damping Ratio
c1 = 1.5
% Personal Learning Coefficient
c2 = 2.0
% Global Learning Coefficient
