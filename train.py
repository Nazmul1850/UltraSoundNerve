import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from data_loader import CustomDataset
from rsaModel import RA_Net

class ContextEncoder(nn.Module):
    def __init__(self):
        super(ContextEncoder, self).__init__()

        #encoder blocks
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)

        #decoder blocks
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
    

    def forward(self, input):
        input = self.relu(self.conv1(input))
        input = self.relu(self.conv2(input))
        input = self.relu(self.conv3(input))
        input = self.relu(self.conv4(input))

        input = self.relu(self.deconv1(input))
        input = self.relu(self.deconv2(input))
        input = self.relu(self.deconv3(input))
        input = self.deconv4(input)

        return input


#hpyer paramters of the model
INIT_LR = 0.0001
BATCH_SIZE = 32
EPOCHS = 10

#75% train data
TRAIN_SPLIT = 0.80
TEST_SPLIT = 1 - TRAIN_SPLIT

# assign device and print to check gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# loading the dataset
dataset = CustomDataset(csv_file='names.csv', image_dir='sample',
                         transorm=transforms.ToTensor())
train_len = int(len(dataset)*TRAIN_SPLIT)
test_len = len(dataset) - train_len
train_set, test_set = random_split(dataset, [train_len,test_len], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False)

train_steps = len(train_loader.dataset) // BATCH_SIZE
test_steps = len(test_loader.dataset) // BATCH_SIZE

print('Dataset Loaded')
print('Train samples : {}'.format(len(train_loader.dataset)))
print('Test samples : {}'.format(len(test_loader.dataset)))

print(len(train_loader.dataset))

# initializing model parameters
# model = ContextEncoder().to(device)
model = RA_Net(n_channels=1).to(device)
optim = optim.Adam(model.parameters(), lr = INIT_LR)
sigmoid = nn.Sigmoid()
loss_func = nn.BCELoss()

# this 'H' will save the training history
H = {
    'train_loss' : [],
    'test_loss' : []
}

#training the network
print('Training Starts...')
start_time = time.time()
for e in tqdm(range(EPOCHS)):
    model.train()
    train_loss = 0
    test_loss = 0

    for i,(image, real_image) in enumerate(train_loader):
        # image send to device (cpu or gpu)
        (image,real_image) = (image.to(device), real_image.to(device))

        # forward propagation
        optim.zero_grad()
        outputs = model(image)

        # calculating loss
        loss = loss_func(sigmoid(outputs), real_image)

        # backward propagation
        loss.backward()
        optim.step()

        train_loss += loss

    # after each epoch calculating the test loss

    with torch.no_grad():
        model.eval()

        for i,(image,real_image) in enumerate(test_loader):
            (image,real_image) = (image.to(device), real_image.to(device))


            outputs = model(image)
            test_loss += loss_func(sigmoid(outputs), real_image)
    
    avg_train_loss = train_loss / train_steps
    avg_test_loss = test_loss / test_steps

    H['train_loss'].append(avg_train_loss.cpu().detach().numpy())
    H['test_loss'].append(avg_test_loss.cpu().detach().numpy())

    print('[INFO] EPOCH : {}/{}'.format(e+1, EPOCHS))
    print('Train loss : {:.6f}'.format(avg_train_loss))
    print('Test loss : {:.6f}'.format(avg_test_loss))

end_time = time.time()
print('Total time taken to train model : {:.2f}s'.format(end_time - start_time))

# finally plot the train loss and validation loss

plt.style.use('ggplot')
plt.figure()
plt.plot(H['train_loss'], label = 'train loss')
plt.plot(H['test_loss'], label = 'test loss')
plt.title('Training loss and Testing loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc = 'lower left')
plt.savefig('loss_plot.png')

torch.save(model.state_dict(), 'nerve_model_path.path')

print('hello world')