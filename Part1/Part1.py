import gco
import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


SOURCE_COLOR = [0, 0, 255 ]   # blue = foreground
SINK_COLOR = [245, 210, 110]  # yellow = background

def dist (c1,c2 ):
    return ( abs( c1[0] - c2[0] )+ \
             abs( c1[1] - c2[1] )+ \
             abs( c1[2] - c2[2] )) / 3; 
'''
def construct_graph(H, W):
# This function is used to construct a corresponding directed graph of an
# image of H * W. It's faster than the original version in the example code
# of GCMex, since it avoids unnecessary operation on sparse matrices.
# The weights are initialized to 1.
# index matrix
    matrix = np.arange(0, H*W, 1)
    index = np.reshape(matrix, (W, H))    # row-wise indexing

# up-down connection
    up      = index(1:end-1, :)
    down    = index(2:end, :)

# left-right connection
    left    = index(:, 1:end-1)
    right   = index(:, 2:end)

# source and target pairs
    source  = [up(:); down(:); left(:); right(:)]
    target  = [down(:); up(:); right(:); left(:)]

# weights
    weights = ones(length(source), 1)
    pairwise = sparse(source, target, weights)

    return  pairwise
'''

m_lambda = 200     # change this value to change the weight of the smoothness or prior term

img = cv2.imread("bayes_in.jpg", cv2.COLOR_BGR2GRAY)
W, H = img.shape[0:2]

edges = np.array([])

unary_cost = np.array([])

for x in range (H-1):
    for y in range (W-1):

        c = img[x, y]
        node = x * W + y
        if x==0 and y ==0:
            unary_cost = np.array([dist(SOURCE_COLOR, c),dist(SINK_COLOR)])
        # data term: 
        else:
            unary_cost = np.vstack[unary_cost, np.array([dist(SOURCE_COLOR, c), dist(SINK_COLOR)])]

        # prior term: start

        nx = x + 1; # the right neighbor
        next_node_r = y*W + nx
        edges[:, node] =  node, next_node_r

        ny = y + 1; # the below neighbor
        next_node_b = ny*H + x

        if x==0 and y ==0:
            edges = np.array([node, next_node_r])
            edges = np.vstack[edges,np.array([node, next_node_b])]
        # data term: 
        else:
            edges = np.vstack[edges,np.array([node, next_node_r])]
            edges = np.vstack[edges,np.array([node, next_node_b])]

        #graph->add_edge(node, next_node, m_lambda, m_lambda )
        # prior term: end
print(tweights.shape)
print(edges.shape)
smooth = 1 - np.eye(2)

labels = gco.cut_general_graph(edges, tweights, unary, smooth, n_iter=1)





[H,W,~]=size(img);
N=H*W;

foreg=[0,0,255];
backg=[245,210,110];

dist_foreg=(abs(img(:,:,1)-foreg(1))+abs(img(:,:,2)-foreg(2))+abs(img(:,:,3)-foreg(3)))./3;
dist_backg=(abs(img(:,:,1)-backg(1))+abs(img(:,:,2)-backg(2))+abs(img(:,:,3)-backg(3)))./3;

unary=[reshape(dist_foreg,1,N);reshape(dist_backg,1,N)];

segclass=double(unary(1,:)>=unary(2,:));

k=1;
for col=1:W
    for row=1:H
        n=(col-1)*H+row;
        if row<H
            i(k)=n;
            j(k)=n+1;
            k=k+1;
        end
        if row>1
            i(k)=n;
            j(k)=n-1;
            k=k+1;
        end
        if col<W
            i(k)=n;
            j(k)=n+H;
            k=k+1;
        end
        if col>1
            i(k)=n;
            j(k)=n-H;
            k=k+1;
        end
    end
end

labelcost=[0,1;1,0];

for lambda=1:200
    pairwise=sparse(i,j,lambda);
    [label,~,~]=GCMex(segclass,single(unary),pairwise,single(labelcost),0);
    label=reshape(label,H,W);
    label1=label*backg(1)+(1-label)*foreg(1);
    label2=label*backg(2)+(1-label)*foreg(2);
    label3=label*backg(3)+(1-label)*foreg(3);
    result=cat(3,label1,label2,label3);
    if lambda==178
        figure(1)
        imshow(uint8(result))
        xlabel('Lambda=178')
        figure(2)
        imshow(img_gt)
        xlabel('Ground Truth')
    end
    error(lambda)=sum(sum(sum(abs(result-img_gt),3)./3))./N;
end
figure(3)
plot(error)
xlabel('Lambda')
ylabel('Error Rate (L1 Norm)')
