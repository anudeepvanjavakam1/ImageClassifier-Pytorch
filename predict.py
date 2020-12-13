import torch
from torchvision import models, datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from workspace_utils import active_session
import argparse
from load_and_preprocess_data import load_preprocess_data
from PIL import Image
import numpy as np
import json


# Create the parser
my_parser = argparse.ArgumentParser(description='predicts class from an image along with probability.')

# Add the arguments
my_parser.add_argument('image_path',
                       metavar='image_path',
                       type=str,
                       help='the path to a single image')

my_parser.add_argument('checkpoint_dir', default = '/home/workspace/ImageClassifier/checkpoint3.pth', type=str, help='directory path to get checkpoint.')
my_parser.add_argument('--top_k', default = 3, type=int, help='return top k most likely classes')
my_parser.add_argument('--category_names', default = '/home/workspace/ImageClassifier/cat_to_name.json', type=str, help='path to json file with all classes')
my_parser.add_argument('--gpu', action = 'store_true', help='will run the model on gpu if cuda is available')

# Execute parse_args()
args = my_parser.parse_args()



if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

checkpoint = torch.load(args.checkpoint_dir, map_location = map_location)

model = models.__dict__[checkpoint['arch']](pretrained=True)

    
for param in model.parameters():
    param.requires_grad = False
 
model.classifier = checkpoint['classifier']
    
model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx = checkpoint['class_to_idx']

device = torch.device("cuda" if torch.cuda.is_available()&args.gpu else "cpu")
#Move model to cpu/gpu based on user's preference.
model.to(device)

#function to process an image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im = im.resize((256, 256))
    im = im.crop((16, 16, 240, 240))
    np_image = (np.array(im))/255
    means = [0.485, 0.456, 0.406]
    sds = [0.229, 0.224, 0.225]
    np_image = (np_image - means)/sds
    np_image = np_image.transpose(2,0,1)
    
    return np_image
    


class_to_idx = model.class_to_idx

#function to predict a single image
def predict(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    top_class = []
    
    model.eval()
    # Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)
    image = image.to(device)
    
    with torch.no_grad():
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)
        top_p, top_idx = ps.topk(topk, dim = 1)         
        top_p = top_p.cpu().numpy()[0]        
        #idx_to_class = dict(map(reversed,class_to_idx.items())) #no need to convert as they are already converted in training part
        top_idx = top_idx.cpu().numpy()[0].tolist()
        for idx in top_idx:
            top_class.append(class_to_idx[idx])
            
    
    return top_p, top_class


#predict an image
probs, classes = predict(image_path = args.image_path, model = model, topk = args.top_k)

if args.category_names == 'cat_to_name.json':
    #a json file that has all output classes. Each class is associated with a name.
    # json file is read and is converted to a dictionary
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    class_names= []    
    for i in classes:
        class_names.append(cat_to_name[i])
        
    print(class_names, probs)

else:
    print(classes, probs)

