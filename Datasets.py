import torch
import librosa, asyncio
import numpy as np
import pickle, os
from typing import Dict, List, Optional
import functools
from random import choice

from Phonemize import Phonemize, Language, English_Phoneme_Split
from Modules.Nvidia_Alignment_Learning_Framework import Attention_Prior_Generator
from meldataset import mel_spectrogram

def Text_to_Token(text: str, token_dict: Dict[str, int]):
    return np.array([
        token_dict[letter]
        for letter in ['<S>'] + list(text) + ['<E>']
        ], dtype= np.int32)

def Token_Stack(tokens: List[np.ndarray], token_dict: Dict[str, int], max_length: Optional[int]= None):
    max_token_length = max_length or max([token.shape[0] for token in tokens])
    tokens = np.stack(
        [np.pad(token, [0, max_token_length - token.shape[0]], constant_values= token_dict['<E>']) for token in tokens],
        axis= 0
        )
    return tokens

def F0_Stack(f0s: List[np.ndarray], max_length: int= None):
    max_f0_length = max_length or max([f0.shape[0] for f0 in f0s])
    f0s = np.stack(
        [np.pad(f0, [0, max_f0_length - f0.shape[0]], constant_values= 0.0) for f0 in f0s],
        axis= 0
        )
    return f0s

def Mel_Stack(mels: List[np.ndarray], max_length: Optional[int]= None):
    max_mel_length = max_length or max([mel.shape[1] for mel in mels])
    mels = np.stack(
        [np.pad(mel, [[0, 0], [0, max_mel_length - mel.shape[1]]], constant_values= mel.min()) for mel in mels],
        axis= 0
        )
    return mels

def Attention_Prior_Stack(attention_priors: List[np.ndarray], max_token_length: int, max_mel_length: int):
    attention_priors_padded = np.zeros(
        shape= (len(attention_priors), max_mel_length, max_token_length),
        dtype= np.float32
        )    
    for index, attention_prior in enumerate(attention_priors):
        attention_priors_padded[index, :attention_prior.shape[0], :attention_prior.shape[1]] = attention_prior

    return attention_priors_padded

def Audio_Stack(audios: List[np.ndarray], max_length: Optional[int]= None) -> np.ndarray:
    max_audio_length = max_length or max([audio.shape[0] for audio in audios])
    audios = np.stack(
        [np.pad(audio, [0, max_audio_length - audio.shape[0]], constant_values= 0.0) for audio in audios],
        axis= 0
        )

    return audios

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        language_dict: Dict[str, Dict[str, float]],
        speaker_dict: Dict[str, Dict[str, float]],
        use_between_padding: bool,
        pattern_path: str,
        metadata_file: str,
        mel_length_min: int,
        mel_length_max: int,
        text_length_min: int,
        text_length_max: int,
        accumulated_dataset_epoch: int= 1,
        augmentation_ratio: float= 0.0,
        use_pattern_cache: bool= False
        ):
        super().__init__()
        self.token_dict = token_dict
        self.speaker_dict = speaker_dict
        self.language_dict = language_dict
        self.use_between_padding = use_between_padding
        self.pattern_path = pattern_path

        self.attention_prior_generator = Attention_Prior_Generator()

        metadata_dict = pickle.load(open(
            os.path.join(pattern_path, metadata_file).replace('\\', '/'), 'rb'
            ))

        self.patterns = []
        max_pattern_by_speaker = max([
            len(patterns)
            for patterns in metadata_dict['File_List_by_Speaker_Dict'].values()
            ])
        for patterns in metadata_dict['File_List_by_Speaker_Dict'].values():
            ratio = float(len(patterns)) / float(max_pattern_by_speaker)
            if ratio < augmentation_ratio:
                patterns *= int(np.ceil(augmentation_ratio / ratio))
            self.patterns.extend(patterns)

        self.patterns = [
            x for x in self.patterns
            if all([
                metadata_dict['Mel_Length_Dict'][x] >= mel_length_min,
                metadata_dict['Mel_Length_Dict'][x] <= mel_length_max,
                metadata_dict['Text_Length_Dict'][x] >= text_length_min,
                metadata_dict['Text_Length_Dict'][x] <= text_length_max
                ])
            ] * accumulated_dataset_epoch
        
        self.file_list_by_speaker = metadata_dict['File_List_by_Speaker_Dict']

        if use_pattern_cache:
            self.Pattern_LRU_Cache = functools.lru_cache(maxsize= None)(self.Pattern_LRU_Cache)

    def __getitem__(self, idx):
        path = os.path.join(self.pattern_path, self.patterns[idx]).replace('\\', '/')
        token, languages, mel, f0, speaker = self.Pattern_LRU_Cache(path)

        attention_prior = self.attention_prior_generator.get_prior(mel.shape[1], token.shape[0])
        reference_mel = pickle.load(open(os.path.join(self.pattern_path, choice(self.file_list_by_speaker[speaker])), 'rb'))['Mel']

        return token, reference_mel, languages, mel, f0, attention_prior
    
    def Pattern_LRU_Cache(self, path: str):
        pattern_dict = pickle.load(open(path, 'rb'))

        if self.use_between_padding:
            # padding between tokens
            token = ['<P>'] * (len(pattern_dict['Pronunciation']) * 2 - 1)
            token[0::2] = pattern_dict['Pronunciation']
        else:
            token = pattern_dict['Pronunciation']
        token = Text_to_Token(token, self.token_dict)

        language = self.language_dict[pattern_dict['Language']]
        
        return token, language, pattern_dict['Mel'], pattern_dict['F0'], pattern_dict['Speaker']

    def __len__(self):
        return len(self.patterns)

class Inference_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        token_dict: Dict[str, int],
        language_dict: Dict[str, Dict[str, float]],
        sample_rate: int,
        hop_size: int,
        n_mels: int,
        use_between_padding: bool,
        texts: List[str],
        reference_paths: List[str],
        languages: List[str],
        ):
        super().__init__()
        self.token_dict = token_dict
        self.language_dict = language_dict
        self.use_between_padding = use_between_padding

        pronunciations = [
            English_Phoneme_Split(pronunciation)
            for pronunciation in Phonemize([x.upper().strip() for x in texts], language= Language.English)
            ]

        reference_audios = [
            librosa.util.normalize(librosa.load(path, sr=sample_rate)[0]) * 0.95
            for path in reference_paths
            ]
        audio_lengths = [audio.shape[0] for audio in reference_audios]
        mel_lengths = [length // hop_size for length in audio_lengths]
        reference_audios_tensor = torch.from_numpy(Audio_Stack(reference_audios)).float()
        reference_mels = mel_spectrogram(
            y= reference_audios_tensor,
            n_fft= hop_size * 4,
            num_mels= n_mels,
            sampling_rate= sample_rate,
            hop_size= hop_size,
            win_size= hop_size * 4,
            fmin= 0,
            fmax= None,
            center= False
            ).cpu().numpy()
        reference_mels = [mel[:, :length] for mel, length in zip(reference_mels, mel_lengths)]
        
        self.patterns = list(zip(pronunciations, reference_mels, languages, texts))

    def __getitem__(self, idx):
        pronunciation, reference_mel, language, text = self.patterns[idx]

        if self.use_between_padding:
            token = ['<P>'] * (len(pronunciation) * 2 - 1)
            token[0::2] = pronunciation
        else:
            token = pronunciation
        token = Text_to_Token(token, self.token_dict)

        return token, reference_mel, self.language_dict[language], text, pronunciation

    def __len__(self):
        return len(self.patterns)


class Collater:
    def __init__(
        self,
        token_dict: Dict[str, int]
        ):
        self.token_dict = token_dict

    def __call__(self, batch):
        tokens, reference_mels, languages, mels, f0s, attention_priors = zip(*batch)
        token_lengths = np.array([token.shape[0] for token in tokens])
        reference_mel_lengths = np.array([mel.shape[1] for mel in reference_mels])
        mel_lengths = np.array([mel.shape[1] for mel in mels])
        
        tokens = Token_Stack(
            tokens= tokens,
            token_dict= self.token_dict
            )
        reference_mels = Mel_Stack(mels= reference_mels)
        languages = np.array(languages)
        mels = Mel_Stack(mels= mels)
        f0s = F0_Stack(f0s= f0s)
        attention_priors = Attention_Prior_Stack(
            attention_priors= attention_priors,
            max_token_length= token_lengths.max(),
            max_mel_length= mel_lengths.max()
            )

        tokens = torch.LongTensor(tokens)   # [Batch, Token_t]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        reference_mels = torch.FloatTensor(reference_mels)  # [Batch, Latent_Code_n, Latent_t]
        reference_mel_lengths = torch.LongTensor(reference_mel_lengths)   # [Batch]
        languages = torch.LongTensor(languages)   # [Batch]
        mels = torch.FloatTensor(mels)  # [Batch, Mel_d, Latent_t]
        mel_lengths = torch.LongTensor(mel_lengths)   # [Batch]
        f0s = torch.FloatTensor(f0s)    # [Batch, Latent_t]
        attention_priors = torch.FloatTensor(attention_priors) # [Batch, Token_t, Latent_t]

        return \
            tokens, token_lengths, reference_mels, reference_mel_lengths, languages, \
            mels, mel_lengths, f0s, attention_priors

class Inference_Collater:
    def __init__(self,
        token_dict: Dict[str, int]
        ):
        self.token_dict = token_dict
         
    def __call__(self, batch):
        tokens, reference_mels, languages, texts, pronunciations = zip(*batch)
        token_lengths = np.array([len(token) for token in tokens])
        reference_mel_lengths = np.array([mel.shape[1] for mel in reference_mels])

        tokens = Token_Stack(tokens, self.token_dict)
        reference_mels = Mel_Stack(reference_mels)
        languages = np.array(languages)

        tokens = torch.LongTensor(tokens)   # [Batch, Time]
        token_lengths = torch.LongTensor(token_lengths)   # [Batch]
        reference_mels = torch.FloatTensor(reference_mels)
        reference_mel_lengths = torch.LongTensor(reference_mel_lengths)   # [Batch]
        languages = torch.LongTensor(languages)   # [Batch]
        
        return tokens, token_lengths, reference_mels, reference_mel_lengths, languages, texts, pronunciations