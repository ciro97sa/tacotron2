import IPython.display as ipd
import numpy as np
import torch
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
import os
import sys
sys.path.append('/home/Ciro/Desktop/sansone-english-finetuning/tacotron2/waveglow')
from denoiser import Denoiser

def english_synthesis(c, OUT_PATH, text):
    MODEL_PATH =  c['audio']['model_path']
    WAVEGLOW_MODEL_PATH =  c['audio']['vocoder_path']
    
    thisdict = {}
    for line in reversed((open('merged.dict.txt', "r").read()).splitlines()):
        thisdict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()
    def ARPA(text):
        out = ''
        for word_ in text.split(" "):
            word=word_; end_chars = ''
            while any(elem in word for elem in r"!?,.;") and len(word) > 1:
                if word[-1] == '!': end_chars = '!' + end_chars; word = word[:-1]
                if word[-1] == '?': end_chars = '?' + end_chars; word = word[:-1]
                if word[-1] == ',': end_chars = ',' + end_chars; word = word[:-1]
                if word[-1] == '.': end_chars = '.' + end_chars; word = word[:-1]
                if word[-1] == ';': end_chars = ';' + end_chars; word = word[:-1]
                else: break
            try: word_arpa = thisdict[word.upper()]
            except: word_arpa = ''
            if len(word_arpa)!=0: word = "{" + str(word_arpa) + "}"
            out = (out + " " + word + end_chars).strip()
            if out[-1] != ";": out = out + ";"
            return out
    torch.set_grad_enabled(False)

    # initialize Tacotron2 with the pretrained model
    hparams = create_hparams()
    hparams.sampling_rate = 22050 # Don't change this
    hparams.max_decoder_steps = 1000 # How long the audio will be before it cuts off (1000 is about 11 seconds)
    hparams.gate_threshold = 0.1 # Model must be 90% sure the clip is over before ending generation (the higher this number is, the more likely that the AI will keep generating until it reaches the Max Decoder Steps)
    model = Tacotron2(hparams)
    model.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
    _ = model.cuda().eval()

    waveglow = torch.load(WAVEGLOW_MODEL_PATH)['model']
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    sigma = 0.66
    denoise_strength = 0.01
    raw_input = True # disables automatic ARPAbet conversion, useful for inputting your own ARPAbet pronounciations or just for testing

    for i in text.split("\n"):
        if len(i) < 1: continue;
        if raw_input:
            if i[-1] != ";": i=i+";" 
        else: i = ARPA(i)
        with torch.no_grad(): # save VRAM by not including gradients
            sequence = np.array(text_to_sequence(i, ['english_cleaners']))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
            audio = waveglow.infer(mel_outputs_postnet, sigma=sigma); print(""); ipd.display(ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate))
            audio_denoised = denoiser(audio, strength=denoise_strength)[:, 0]
            MAX_WAV_VALUE = 32768.0
            audio_denoised = audio_denoised * MAX_WAV_VALUE
            audio_denoised = audio_denoised.squeeze()
            audio_numpy = audio_denoised.cpu().numpy().astype('int16')
            audio_numpy = audio[0].data.cpu().numpy()

            from scipy.io.wavfile import write
            write(OUT_PATH, hparams.sampling_rate, audio_numpy)
