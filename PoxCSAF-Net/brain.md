# way 1: ***
Instead of standard conv layers, used Reparameterized convolution (RepConv) and Frequency Aware encoding (MSFCA)

No Bottleneck used here. 

Issue:

Stem layers also used RepConv
Simple addition I→D 
Low range/local receptive field of CNN layers in I branch 

# way 2 ***
Instead of standard conv layers, used Large Kernel Depth Wise Convolution and Cross attention in I->D branch


Fixes:

Stem layers use standard conv layers 
Cross attention on I->D branch

Issue:

Cross attention has no positional embedding
Depth Wise Convolution is best for inference speed up, but for increasing the accuracy we need robust feature extraction and large receptive field  

# way 3
Self-attention in I (global context aware) branch and Cross attention in I->D branch.




Fixes:

Used Self attention for global large receptive field and robust feature extraction

Issues: 

The attention mechanism not worked as expected, gives attention on wrong pixels 

# way 4
Modifications: 

Standard Conv layers in Stem 
P branch has aux loss and proper PAG from I branch 
I branch has 6-layers of 2 head self-attention and parallel MSFCA fused using weighted learnable gating mechanism (global large receptive field)
Proper cross attention with positional embedding on I→D fusion 
DAPPM used instead of PAPPM
Light BAG used instead pf BAG
Varifocal as main and aux loss

# way 5
Modifications: 

Standard Conv layers in Stem 
P branch has aux loss and proper PAG from I branch 
I branch has 3 stages of RepConv layers
Proper PAG-Style attention gating on I→D fusion with bi-directional gradient flow 
PAPPM
Light BAG used instead pf BAG
Varifocal as main and aux loss

# way 6 *****
Modifications: 

Standard Conv layers in Stem 
P branch has aux loss and proper PAG from I branch 
I branch has 3 stages of RepConv layers
Proper PAG-Style attention gating on I→D fusion with bi-directional gradient flow 
PAPPM
Light BAG used instead pf BAG
Varifocal as main and aux loss

# way 7 ***
Modifications: 

Standard Conv layers in Stem 
P branch has aux loss and proper PAG from I branch 
I branch has 3 stages of RepConv layers
PIDNet style addition to I→D branch 
PAPPM
Light BAG used instead pf BAG
Varifocal as main and aux loss

# way 8 *****
Modifications: 

Standard Conv layers in Stem 
P branch has aux loss and proper PAG from I branch 
I branch has 3 stages of RepConv layers
PIDNet style addition to I→D branch
PAPPM
Light BAG used instead pf BAG
Varifocal as main and aux loss
