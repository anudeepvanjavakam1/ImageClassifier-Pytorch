import torch
from torchvision import datasets, transforms

def load_preprocess_data(data_dir):
    ''' given a data directory with train, validation image data sets, fn loads, 
    transforms and returns data loaders'''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    #test_dir = data_dir + '/test'
    
    #Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),                                       
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                         transforms.CenterCrop(224),                                      
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    
    
    #test_transforms = transforms.Compose([transforms.Resize(224),
    #                                     transforms.CenterCrop(224),                                      
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize([0.485, 0.456, 0.406],
    #                                                         [0.229, 0.224, 0.225])])
    
    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    #test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
    #testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)
    
    #returning train_data for class indexes
    return train_data, trainloader, validloader
    

