import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
import numpy as np
import json
from torch.utils.data.sampler import SubsetRandomSampler
import os
import torchvision
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0 # fix token for "start of sentence"
EOS_token = 1 # fix token for "end of sentence"
PAD_token = 2 # fix token for "padding"
UNK_token = 3 # fix token for "unknown"

class Vocabulary:
    def __init__(self):
        self.SOS_token = 0
        self.EOS_token = 1
        self.PAD_token = 2
        self.UNK_token = 3

        self.word2index = {"[SOS]": 0, "[EOS]": 1, "[PAD]": 2, "[UNK]": 3}
        self.index2word = {0: "[SOS]", 1: "[EOS]", 2: "[PAD]", 3: "[UNK]"}
        self.frequencies = {}
        self.freq_threshold=5
        self.num_words = len(self.word2index)     # The number of unique vocabs in your dataset. 
                                                  
        self.longest_sentence = 0       # Max sequence length of caption. (terminate condition while training)

    def add_word(self, word):
        """ add some word to vocab if we have not seen the word """

        if word not in self.frequencies:
          self.frequencies[word] = 1
        else:
          self.frequencies[word] += 1

        # word의 출현 횟수가 threshold를 넘으면 단어목록에 추가한다.
        if self.frequencies[word] == self.freq_threshold:
            self.word2index[word]=self.num_words  
            self.index2word[self.num_words] = word
            self.num_words += 1

    def add_sentence(self, sentence, tokenize_func=None):
        if tokenize_func is None:
            tokenize_func = lambda x: x.split(" ")

        # Tokenize the sentence, and add word to vocabulary class one by one
        tokens = tokenize_func(sentence)
        for token in tokens:
          self.add_word(token)
        sentence_len = len(tokens)

        # update the longest sentence
        if sentence_len > self.longest_sentence:
            self.longest_sentence = sentence_len # This is the longest sentence

class ImgCaptionDataset(Dataset):

    def __init__(self, img_path, caption_path, transform=None, tokenize_func=None):

        with open(caption_path) as json_file:
          itemData = json.loads(json_file.read())
          image_captions=itemData["images"]

        self.img2caption = {}
        self.imgs=[]
        self.img_path=img_path
        self.caption_path=caption_path
        self.max_captions_num = 0
        self.caption_vocab = Vocabulary()

        if tokenize_func is None:
            self.tokenize_func = lambda x: x.split(" ")
        else:
            self.tokenize_func = tokenize_func

        img_files = os.listdir(self.img_path)
        for image in image_captions:

          self.img2caption[image["file_name"]]=[x.lower() for x in image["captions"]]

          for cap in image["captions"]:
            self.caption_vocab.add_sentence(cap.lower())

          self.imgs.append(image["file_name"])

          if len(image["captions"]) > self.max_captions_num:
              self.max_captions_num=len(image["captions"])

        self.max_seq_length = self.caption_vocab.longest_sentence
        self.transform = transform
    def __getitem__(self, i):
        
        imgName=self.imgs[i]
        img = Image.open(os.path.join(self.img_path, imgName)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        captions=[]

        for i in range(self.max_captions_num):

          if i<len(self.img2caption[imgName]):
            caption = [self.caption_vocab.SOS_token] # add SOS token at the start of each caption
            caption += [self.caption_vocab.word2index[token]
                   if token in self.caption_vocab.word2index
                   else self.caption_vocab.word2index["[UNK]"]
                   for token in self.tokenize_func(self.img2caption[imgName][i]) ]
            caption.append(self.caption_vocab.EOS_token) # add EOS token at the end of each caption

          else:
            pass # use just prior caption

          # padding
          if len(caption) - 2 < self.max_seq_length:
            caption.extend([self.caption_vocab.PAD_token] * (self.max_seq_length - len(caption) + 2))

          captions.append(caption)


        captions = torch.Tensor(captions).long()
        return img, captions


    def __len__(self):

        return len(self.img2caption)

transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])


dataset = ImgCaptionDataset(img_path="./data", caption_path="./captions.json", transform=transform_train)

pickle.dump(dataset.caption_vocab, open('vocab_김태은_2017313008.pickle', 'wb'))

print("length of longest caption:", dataset.caption_vocab.longest_sentence)
print("number of vocabulary:", dataset.caption_vocab.num_words)

# split train set / valid set

validation_ratio=0.1
random_seed= 15

num_train = len(dataset)
val_split = int(np.floor(validation_ratio * num_train))
indices = list(range(num_train))

np.random.seed(random_seed)
np.random.shuffle(indices)

valid_idx, train_idx = indices[:val_split], indices[val_split:]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)


trainloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                          sampler=train_sampler)

validloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                          sampler=valid_sampler)

print("number of trainset:", len(train_sampler))
print("number of validset:", len(valid_sampler))


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embedding = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        with torch.no_grad():
        	features = self.resnet(images)
        features = features.view(features.shape[0], -1)
        features = self.embedding(features)
        features = self.bn(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)            
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        return (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                torch.zeros((1, batch_size, self.hidden_size), device=device))

    def forward(self, features, captions):
        captions = captions[:, :-1]
        batch_size = features.shape[0] 
        hidden = self.init_hidden(batch_size)

        embed = self.embedding(captions) 
        embed = self.dropout(embed)
        embed = torch.cat((features.unsqueeze(1), embed), dim=1) 
        out, hidden = self.lstm(embed, hidden) 
        outputs = self.linear(out)

        return outputs
        

class Img2Cap(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(Img2Cap, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, caption):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, caption)
        return outputs

    def make_caption(self, image, vocabulary):
        features = self.encoderCNN(image).unsqueeze(1)

        result = []
        batch_size = image.shape[0] 
        hidden = self.decoderRNN.init_hidden(batch_size) 
        max_length = vocabulary.longest_sentence

        for i in range(0, max_length):
            out, hidden = self.decoderRNN.lstm(features, hidden) 
            outs = self.decoderRNN.linear(out).squeeze(1)
            _, top_index = torch.max(outs, dim=1)
           
            # predict [UNK] token -> ignore
            if top_index !=3:
            	  result.append(top_index.cpu().numpy()[0].item()) # storing the word predicted

            if (top_index == 1):
                # predict [EOS] token -> stop 
                break

            features = self.decoderRNN.embedding(top_index).unsqueeze(1)
            
        # discard [SOS], [EOS] token
        return [vocabulary.index2word[idx] for idx in result[1:-1]]
   


embed_size = 256
hidden_size = 256
output_size = dataset.caption_vocab.num_words

model = Img2Cap(embed_size, hidden_size, output_size, 1).to(device)
optimizer = optim.AdamW(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss(ignore_index=2).to(device)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=2, factor=0.1, verbose=True)

N_EPOCHS = 10
best_valid_loss = 1000  # Any large number will suffice
patience = 0   # Bad epoch counter
CLIP = 1
print(len(trainloader))

for epoch in range(N_EPOCHS):
    model.train()
    
    epoch_loss = 0
    epoch_cnt = 0
    
    print("Epoch", epoch+1)
    for i, (image, captions) in enumerate(trainloader):
        #print(i)
        image, captions = image.to(device), captions.transpose(0, 1).to(device)

        for caption in captions:
          model.zero_grad()
          outputs = model(image, caption)
          loss = criterion(outputs.reshape(-1, outputs.shape[2]), caption.reshape(-1))
          
          torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)  # prevents exploding / vanishing
          loss.backward()
          optimizer.step()

          epoch_cnt +=1
          epoch_loss += loss.item()


    train_loss = epoch_loss / epoch_cnt

    model.eval()
  
    epoch_loss = 0
    epoch_cnt = 0

    with torch.no_grad():
      for i, (image, captions) in enumerate(validloader):
          image, captions = image.to(device), captions.transpose(0, 1).to(device)
          for caption in captions:
            outputs = model(image, caption)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), caption.reshape(-1))

            epoch_loss += loss.item()
            epoch_cnt +=1
    
    valid_loss = epoch_loss / epoch_cnt

    if best_valid_loss > valid_loss:
      best_valid_loss = valid_loss
      torch.save(model.state_dict(), "model_김태은_2017313008.pt")
      print("best model has changed")
      patience = 0
    
    print(f'\tEpoch: {epoch+1}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}')

    scheduler.step(metrics=valid_loss) # adjust learning rate (decay)

    if patience == 2:
      print('Early Stopping at Epoch %d' % (epoch+1))
      break	
    patience += 1



