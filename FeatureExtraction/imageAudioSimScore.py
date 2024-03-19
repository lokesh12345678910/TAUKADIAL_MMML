#conda activate MSpeechCLIP_3.6
#from metrics import compute_metrics
import torch as th
#from torch.utils.data import DataLoader
import clip
from parallel_model import Parallel
from tqdm import tqdm
#import data_paths

clip_model, img_preprocess = clip.load('ViT-L/14',device='cuda')
clip_model.eval()
from PIL import Image
imagePath = "BILP006_BISE004_PreTx_PicnicDescription_Catalan.png"
image = img_preprocess (Image.open(imagePath)).unsqueeze(0).to('cuda')
imgFeatVector = clip_model.encode_image(image)

# Comment out next line if not using pre-computed image features
#image_path = data_paths.image_encodings



modelPath = '/work/09424/smgrasso1/ls6/M-SpeechCLIP/all8GPUs_fullytrainable_crosslingual.pth'
#model = th.nn.DataParallel()
model = th.nn.DataParallel(Parallel(heads=8, layers=1, batch=100, gpus=1, feat_trainable=True, weighted_sum=True, hubert_size='base', clip_size='large', use_langID=False)).cuda()
state_dict = th.load(modelPath)
try:
    model.load_state_dict(state_dict)
except:
    # Older state dicts need cleaning
    to_pop = []
    to_modify = []
    for k,v in state_dict.items():
        if '.clip' in k:
            to_pop.append(k)
        elif 'transform' in k and 'layers' not in k:
            to_modify.append(k)
    for k in to_pop:
        state_dict.pop(k)
    for k in to_modify:
        state_dict[k.replace('transform', 'transform.layers.0')] = state_dict.pop(k)
    model.load_state_dict(state_dict)

audioPath = "/work/09424/smgrasso1/ls6/M-SpeechCLIP/BILP006_Pre_Tx_PicnicDescription_Catalan.wav"
import librosa
audio = th.FloatTensor(librosa.load(audioPath, sr=16_000)[0]).unsqueeze(0).to('cuda')

with th.no_grad():
        print(image.shape, audio.shape)
        image_out, speech_out = model(imgFeatVector, audio)
        print(speech_out.shape,speech_out.dtype,image_out.dtype,imgFeatVector.dtype)
        imageSpeechSimScore = th.matmul(image_out.float(),speech_out.t())
        #imageSpeechSimScore = th.matmul(imgFeatVector,speech_out.t().float())
        #imageSpeechSimScore = th.matmul(image_out.half(), speech_out.t())
        print("M-SpeechCLIP Audio-Image Similarity score:", imageSpeechSimScore.item())

