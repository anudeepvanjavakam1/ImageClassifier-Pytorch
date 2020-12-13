import torch
from torchvision import models, datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from workspace_utils import active_session
import argparse
from load_and_preprocess_data import load_preprocess_data
import json


# Create the parser
my_parser = argparse.ArgumentParser(description='Train a new network on a dataset and optinally save the model as a checkpoint.')

# Add the arguments
my_parser.add_argument('data_dir',
                       metavar='data_dir',
                       type=str,
                       help='the path to directory that contains train and valid folders each of which have train and valid image data sets respectively.')

my_parser.add_argument('--save_dir', default = '/home/workspace/ImageClassifier', type=str, help='directory path to save checkpoints.')
my_parser.add_argument('--arch', default = 'vgg16', type=str, help='architecture of the model. example: vgg16')
my_parser.add_argument('--learning_rate', default = 0.001, type=float, help='learning rate. example: 0.01')
my_parser.add_argument('--hidden_units', default = 1024, type=int, help='number of hidden units. example:512')
my_parser.add_argument('--epochs', default = 5, type=int, help='number of epochs. example: 20')
my_parser.add_argument('--gpu', action = 'store_true', help='will run the model on gpu if cuda is available')

# Execute parse_args()
args = my_parser.parse_args()


train_data, trainloader, validloader = load_preprocess_data(args.data_dir)

#a json file that has all output classes. Each class is associated with a name.
# json file is read and is converted to a dictionary
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
#Build and train your network
device = torch.device("cuda" if torch.cuda.is_available()&args.gpu else "cpu")
model = models.__dict__[args.arch](pretrained=True)

#turn off gradients for model (freeze the parameters)
for param in model.parameters():
    param.requires_grad = False
try:
    input_size = model.classifier[0].in_features
except TypeError:
    input_size = model.classifier.in_features
#create a new classifier and its architecture (but the features remain pretrained from vgg16)
classifier = nn.Sequential(nn.Linear(input_size, args.hidden_units),
                      nn.ReLU(),
                      nn.Dropout(p = 0.1),
                      nn.Linear(args.hidden_units, len(cat_to_name)),     
                      nn.LogSoftmax(dim = 1)
                      )
model.classifier = classifier

#define criterion
criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)

#move the model to cuda if available. Remember to move images, labels to the same later.
model.to(device)


epochs = args.epochs
steps = 0
running_loss = 0
print_every = 100

#begin training model
#active_session keeps session awake during long run times and prevents closing session
with active_session():

    #for each run
    for epoch in range(epochs):
        #for each batch of inputs and labels from the train data
        for inputs, labels in trainloader:
            steps +=1
            #move inputs and labels to cuda if available
            inputs, labels = inputs.to(device), labels.to(device)
            
            #clear all gradients before training
            optimizer.zero_grad()
            #feed forward
            log_ps = model.forward(inputs)
            #calc loss
            loss = criterion(log_ps, labels)
            #backpropagation
            loss.backward()
            #gradient descent step
            optimizer.step()
            #track cumulative training loss
            running_loss += loss.item()

            #for every 5 steps, do a validation pass and track test loss and test accuracy
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0

                #turn on model evaluation mode
                model.eval()
                #turn off the gradients
                with torch.no_grad():
                    for inputs, labels in validloader:

                        #move test data as well to cuda if available
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        log_ps = model.forward(inputs)
                        loss = criterion(log_ps, labels)
                        valid_loss += loss.item()

                        #test accuracy
                        ps = torch.exp(log_ps)
                        #topk returns the class with highest probability
                        top_p, top_class = ps.topk(1, dim = 1)
                        #comparing predicted class with actual class
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print("epoch:{}/{}..".format(epoch+1,epochs),
                     "running_loss:{0:.3f}..".format(running_loss/print_every),
                     "valid_loss:{0:.3f}..".format(valid_loss/len(validloader)),
                     "accuracy:{0:.3f}".format(accuracy/len(validloader)))
                running_loss = 0
                model.train()
                
        
#Save the checkpoint 
class_to_idx = train_data.class_to_idx
model.class_to_idx = { class_to_idx[k]: k for k in class_to_idx}

checkpoint = {'input_size' : input_size,
    'output_size' : len(cat_to_name),
    #'hidden_layers': args.hidden_units,
    'classifier': model.classifier,
    'optimizer': optimizer,
    'arch': args.arch,
    'state_dict': model.state_dict(),
    #'optimizer_state': optimizer.state_dict(),
    'class_to_idx': model.class_to_idx
}
if args.save_dir is None:
    torch.save(checkpoint,'checkpoint3.pth')
else:    
    torch.save(checkpoint, args.save_dir + '/checkpoint3.pth')        
        
        
        





