import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from collections import OrderedDict
import pickle
from PIL import Image
import json
import os


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



def load_image(image_path, transform=None):
    image = Image.open(image_path).convert("RGB")

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

vocab = pickle.load(open('./vocab_김태은_2017313008.pickle', 'rb'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_size = 256
hidden_size = 256
output_size = vocab.num_words
print("number of vocabulary:", output_size)

state_dict = torch.load('./model_김태은_2017313008.pt', map_location=device)

transform_test = transforms.Compose([
    transforms.Resize(256),                         
    transforms.ToTensor(),                           
    transforms.Normalize((0.485, 0.456, 0.406),      
                         (0.229, 0.224, 0.225))])


model = Img2Cap(embed_size, hidden_size, output_size, 1).to(device)
model.load_state_dict(state_dict)

img_list = os.listdir('./test')
model.eval()

group_data=OrderedDict()
group_data['images']=[]

with torch.no_grad():
  for img in img_list:
        image = Image.open(os.path.join('./data', img)).convert("RGB")
        image = transform_test(image).unsqueeze(0).to(device)
        
        outputs = model.make_caption(image, vocab)
        sentence = ' '.join(outputs).capitalize() 
        
        file = OrderedDict()
        file['file_name']=img
        file['captions']=sentence

        group_data['images'].append(file.copy())

print(json.dumps(group_data, ensure_ascii=False, indent="\t") )

with open('김태은_result.json', 'w', encoding="utf-8") as make_file:
    json.dump(group_data, make_file, ensure_ascii=False, indent="\t")


