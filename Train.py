import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'    # This is ot prevent to be called Fortran Ctrl+C crash in Windows.
import torch
import numpy as np
import logging, yaml, os, sys, argparse, math, wandb, warnings, traceback
from functools import partial
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from accelerate import Accelerator
from Logger import Logger
from typing import List

from Modules.Modules import RectifiedFlowTTS, Mask_Generate
from Modules.Nvidia_Alignment_Learning_Framework import AttentionBinarizationLoss, AttentionCTCLoss
from Datasets import Dataset, Inference_Dataset, Collater, Inference_Collater

from meldataset import mel_spectrogram
from Arg_Parser import Recursive_Parse, To_Non_Recursive_Dict

import matplotlib as mpl
# 유니코드 깨짐현상 해결
mpl.rcParams['axes.unicode_minus'] = False
# 나눔고딕 폰트 적용
plt.rcParams["font.family"] = 'NanumGothic'

logging.basicConfig(
    level=logging.INFO, stream=sys.stdout,
    format= '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    )
warnings.filterwarnings('ignore')

def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    with open("error_log.txt", "a") as log_file:
        log_file.write("Uncaught exception:\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=log_file)

# sys.excepthook = log_uncaught_exceptions

# torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, hp_path, steps= 0):
        self.hp_path = hp_path
        self.num_gpus = int(os.getenv("WORLD_SIZE", '1'))
        
        self.hp = Recursive_Parse(yaml.load(
            open(self.hp_path, encoding='utf-8'),
            Loader=yaml.Loader
            ))

        self.accelerator = Accelerator(
            split_batches= True,
            mixed_precision= 'fp16' if self.hp.Use_Mixed_Precision else 'no',   # no, fp16, bf16, fp8
            gradient_accumulation_steps= self.hp.Train.Accumulated_Gradient_Step
            )

        if not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = self.accelerator.device
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
        
        self.steps = steps

        self.Dataset_Generate()
        self.Model_Generate()
        
        self.dataloader_dict['Train'], self.dataloader_dict['Eval'], self.dataloader_dict['Inference'], \
        self.model_dict['RectifiedFlowTTS'], self.model_dict['RectifiedFlowTTS_EMA'], \
        self.optimizer_dict['RectifiedFlowTTS'], self.scheduler_dict['RectifiedFlowTTS'] = self.accelerator.prepare(
            self.dataloader_dict['Train'],
            self.dataloader_dict['Eval'],
            self.dataloader_dict['Inference'],
            self.model_dict['RectifiedFlowTTS'],
            self.model_dict['RectifiedFlowTTS_EMA'],
            self.optimizer_dict['RectifiedFlowTTS'],
            self.scheduler_dict['RectifiedFlowTTS'],
            )
        
        self.Load_Checkpoint()

        self.scalar_dict = {
            'Train': defaultdict(float),
            'Evaluation': defaultdict(float),
            }

        if self.accelerator.is_main_process:
            self.writer_dict = {
                'Train': Logger(os.path.join(self.hp.Log_Path, 'Train')),
                'Evaluation': Logger(os.path.join(self.hp.Log_Path, 'Evaluation')),
                }
            
            if self.hp.Weights_and_Biases.Use:
                wandb.init(
                    project= self.hp.Weights_and_Biases.Project,
                    entity= self.hp.Weights_and_Biases.Entity,
                    name= self.hp.Weights_and_Biases.Name,
                    config= To_Non_Recursive_Dict(self.hp)
                    )
                wandb.watch(self.model_dict['RectifiedFlowTTS'])

    def Dataset_Generate(self):
        token_dict = yaml.load(open(self.hp.Token_Path, encoding= 'utf-8-sig'), Loader=yaml.Loader)
        speaker_dict = yaml.load(open(self.hp.Speaker_Info_Path, encoding= 'utf-8-sig'), Loader=yaml.Loader)
        language_dict = yaml.load(open(self.hp.Language_Info_Path, encoding= 'utf-8-sig'), Loader=yaml.Loader)
        
        train_dataset = Dataset(
            token_dict= token_dict,
            language_dict= language_dict,            
            speaker_dict= speaker_dict,
            use_between_padding= self.hp.Use_Between_Padding,
            pattern_path= self.hp.Train.Train_Pattern.Path,
            metadata_file= self.hp.Train.Train_Pattern.Metadata_File,
            latent_length_min= self.hp.Train.Train_Pattern.Feature_Length.Min,
            latent_length_max= self.hp.Train.Train_Pattern.Feature_Length.Max,
            text_length_min= self.hp.Train.Train_Pattern.Text_Length.Min,
            text_length_max= self.hp.Train.Train_Pattern.Text_Length.Max,
            accumulated_dataset_epoch= self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch,
            augmentation_ratio= self.hp.Train.Train_Pattern.Augmentation_Ratio,
            use_pattern_cache= self.hp.Train.Pattern_Cache
            )
        eval_dataset = Dataset(
            token_dict= token_dict,
            speaker_dict= speaker_dict,
            language_dict= language_dict,
            use_between_padding= self.hp.Use_Between_Padding,
            pattern_path= self.hp.Train.Eval_Pattern.Path,
            metadata_file= self.hp.Train.Eval_Pattern.Metadata_File,
            latent_length_min= self.hp.Train.Train_Pattern.Feature_Length.Min,
            latent_length_max= self.hp.Train.Eval_Pattern.Feature_Length.Max,
            text_length_min= self.hp.Train.Eval_Pattern.Text_Length.Min,
            text_length_max= self.hp.Train.Eval_Pattern.Text_Length.Max,
            use_pattern_cache= self.hp.Train.Pattern_Cache
            )
        inference_dataset = Inference_Dataset(
            token_dict= token_dict,
            language_dict= language_dict,
            texts= self.hp.Train.Inference_in_Train.Text,
            languages= self.hp.Train.Inference_in_Train.Language,
            )

        if self.accelerator.is_main_process:
            logging.info('The number of train patterns = {}.'.format(len(train_dataset) // self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch))
            logging.info('The number of development patterns = {}.'.format(len(eval_dataset)))
            logging.info('The number of inference patterns = {}.'.format(len(inference_dataset)))

        collater = Collater(
            token_dict= token_dict
            )
        inference_collater = Inference_Collater(
            token_dict= token_dict
            )

        self.dataloader_dict = {}
        self.dataloader_dict['Train'] = torch.utils.data.DataLoader(
            dataset= train_dataset,
            sampler= torch.utils.data.RandomSampler(train_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Eval'] = torch.utils.data.DataLoader(
            dataset= eval_dataset,
            sampler= torch.utils.data.RandomSampler(eval_dataset),
            collate_fn= collater,
            batch_size= self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )
        self.dataloader_dict['Inference'] = torch.utils.data.DataLoader(
            dataset= inference_dataset,
            sampler= torch.utils.data.SequentialSampler(inference_dataset),
            collate_fn= inference_collater,
            batch_size= self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size,
            num_workers= self.hp.Train.Num_Workers,
            pin_memory= True
            )

    def Model_Generate(self):
        self.model_dict = {
            'RectifiedFlowTTS': RectifiedFlowTTS(self.hp).to(self.device)
            }
        self.model_dict['RectifiedFlowTTS_EMA'] = torch.optim.swa_utils.AveragedModel(
            self.model_dict['RectifiedFlowTTS'],
            multi_avg_fn= torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
            )
        self.model_dict['RectifiedFlowTTS_EMA'].Inference = self.model_dict['RectifiedFlowTTS_EMA'].module.Inference

        self.criterion_dict = {
            'MSE': torch.nn.MSELoss(reduction= 'none').to(self.device),
            'CE': torch.nn.CrossEntropyLoss().to(self.device),
            'Attention_Binarization': AttentionBinarizationLoss(),
            'Attention_CTC': AttentionCTCLoss(),
            }

        self.optimizer_dict = {
            'RectifiedFlowTTS': torch.optim.NAdam(
                params= self.model_dict['RectifiedFlowTTS'].parameters(),
                lr= self.hp.Train.Learning_Rate.Initial,
                betas=(self.hp.Train.ADAM.Beta1, self.hp.Train.ADAM.Beta2),
                eps= self.hp.Train.ADAM.Epsilon,
                weight_decay= self.hp.Train.Weight_Decay
                )
            }
        self.scheduler_dict = {
            'RectifiedFlowTTS': torch.optim.lr_scheduler.ExponentialLR(
                optimizer= self.optimizer_dict['RectifiedFlowTTS'],
                gamma= self.hp.Train.Learning_Rate.Decay,
                last_epoch= -1
                )
            }
        
        self.mel_func = partial(
            mel_spectrogram,
            n_fft= self.hp.Sound.N_FFT,
            num_mels= self.hp.Sound.Num_Mel,
            sampling_rate= self.hp.Sound.Sample_Rate,
            hop_size= self.hp.Sound.Hop_Size,
            win_size= self.hp.Sound.Hop_Size * 4,
            fmin= 0,
            fmax= None
            )

        # if self.accelerator.is_main_process:
        #     logging.info(self.model_dict['RectifiedFlowTTS'])

    def Train_Step(
        self,
        tokens: torch.IntTensor,
        token_lengths: torch.IntTensor,
        languages: torch.IntTensor,
        latent_codes: torch.IntTensor,
        latent_code_lengths: torch.IntTensor,
        f0s: torch.FloatTensor,
        mels: torch.FloatTensor,
        speakers: torch.IntTensor,
        attention_priors: torch.FloatTensor
        ):
        loss_dict = {}

        with self.accelerator.accumulate(self.model_dict['RectifiedFlowTTS']):
            flows, prediction_flows, \
            durations, prediction_durations, prediction_f0s, \
            attention_softs, attention_hards, attention_logprobs, \
            prediction_speakers, _ = self.model_dict['RectifiedFlowTTS'](
                tokens= tokens,
                token_lengths= token_lengths,
                languages= languages,
                latent_codes= latent_codes,
                latent_code_lengths= latent_code_lengths,
                f0s= f0s,
                mels= mels,
                attention_priors= attention_priors
                )
            
            token_masks = ~Mask_Generate(
                lengths= token_lengths,
                max_length= tokens.size(1)
                ).to(tokens.device)
            latent_masks = ~Mask_Generate(
                lengths= latent_code_lengths,
                max_length= latent_codes.size(2)
                ).to(latent_codes.device)[:, None, :]
            
            loss_dict['Diffusion'] = (self.criterion_dict['MSE'](
                prediction_flows,
                flows,
                ) * latent_masks).sum() / latent_masks.sum() / prediction_flows.size(1)            
            loss_dict['Attention_Binarization'] = self.criterion_dict['Attention_Binarization'](attention_hards, attention_softs)
            loss_dict['Attention_CTC'] = self.criterion_dict['Attention_CTC'](attention_logprobs, token_lengths, latent_code_lengths)            
            loss_dict['Duration'] = (self.criterion_dict['MSE'](
                (prediction_durations + 1).log(),
                (durations + 1).log(),
                ) * token_masks).sum() / token_masks.sum()
            loss_dict['F0'] = (self.criterion_dict['MSE'](
                (prediction_f0s + 1).log(),
                (f0s + 1).log(),
                ) * latent_masks).sum() / latent_masks.sum()
            loss_dict['Speaker'] = self.criterion_dict['CE'](
                input= prediction_speakers,
                target= speakers.long(),
                )

            self.optimizer_dict['RectifiedFlowTTS'].zero_grad()
            self.accelerator.backward(
                loss_dict['Diffusion'] +
                loss_dict['Attention_Binarization'] +
                loss_dict['Attention_CTC'] +
                loss_dict['Duration'] +
                loss_dict['F0']
                # loss_dict['Speaker']
                )

            if self.hp.Train.Gradient_Norm > 0.0:
                self.accelerator.clip_grad_norm_(
                    parameters= self.model_dict['RectifiedFlowTTS'].parameters(),
                    max_norm= self.hp.Train.Gradient_Norm
                    )
                
            self.optimizer_dict['RectifiedFlowTTS'].step()            
            self.model_dict['RectifiedFlowTTS_EMA'].update_parameters(self.model_dict['RectifiedFlowTTS'])

            self.steps += 1
            self.tqdm.update(1)

        for tag, loss in loss_dict.items():
            self.scalar_dict['Train']['Loss/{}'.format(tag)] += loss.item()

    def Train_Epoch(self):
        for tokens, token_lengths, languages, latent_codes, latent_code_lengths, f0s, mels, speakers, attention_priors in self.dataloader_dict['Train']:
            self.Train_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                languages= languages,
                latent_codes= latent_codes,
                latent_code_lengths= latent_code_lengths,
                f0s= f0s,
                mels= mels,
                speakers= speakers,
                attention_priors= attention_priors
                )

            if self.steps % (math.ceil(len(self.dataloader_dict['Train'].dataset) / self.hp.Train.Train_Pattern.Accumulated_Dataset_Epoch / self.hp.Train.Batch_Size) * self.hp.Train.Learning_Rate.Decay_Epoch) == 0:
                self.scheduler_dict['RectifiedFlowTTS'].step()

            if self.steps % self.hp.Train.Checkpoint_Save_Interval == 0:
                self.Save_Checkpoint()

            if self.steps % self.hp.Train.Logging_Interval == 0 and self.accelerator.is_main_process:
                self.scalar_dict['Train'] = {
                    tag: loss / self.hp.Train.Logging_Interval
                    for tag, loss in self.scalar_dict['Train'].items()
                    }
                self.scalar_dict['Train']['Learning_Rate'] = self.scheduler_dict['RectifiedFlowTTS'].get_last_lr()[0]
                self.writer_dict['Train'].add_scalar_dict(self.scalar_dict['Train'], self.steps)
                if self.hp.Weights_and_Biases.Use:
                    wandb.log(
                        data= {
                            f'Train.{key}': value
                            for key, value in self.scalar_dict['Train'].items()
                            },
                        step= self.steps,
                        commit= self.steps % self.hp.Train.Evaluation_Interval != 0
                        )
                self.scalar_dict['Train'] = defaultdict(float)

            if self.steps % self.hp.Train.Evaluation_Interval == 0:
                self.Evaluation_Epoch()

            if self.steps % self.hp.Train.Inference_Interval == 0:
                self.Inference_Epoch()
            
            if self.steps >= self.hp.Train.Max_Step:
                return

    @torch.no_grad()
    def Evaluation_Step(
        self,
        tokens: torch.IntTensor,
        token_lengths: torch.IntTensor,
        languages: torch.IntTensor,
        latent_codes: torch.IntTensor,
        latent_code_lengths: torch.IntTensor,
        f0s: torch.FloatTensor,
        mels: torch.FloatTensor,
        speakers: torch.IntTensor,
        attention_priors: torch.FloatTensor
        ):
        loss_dict = {}

        flows, prediction_flows, \
        durations, prediction_durations, prediction_f0s, \
        attention_softs, attention_hards, attention_logprobs, \
        prediction_speakers, alignments = self.model_dict['RectifiedFlowTTS'](
            tokens= tokens,
            token_lengths= token_lengths,
            languages= languages,
            latent_codes= latent_codes,
            latent_code_lengths= latent_code_lengths,
            f0s= f0s,
            mels= mels,
            attention_priors= attention_priors
            )

        token_masks = ~Mask_Generate(
            lengths= token_lengths,
            max_length= tokens.size(1)
            ).to(tokens.device)
        latent_masks = ~Mask_Generate(
            lengths= latent_code_lengths,
            max_length= latent_codes.size(2)
            ).to(latent_codes.device)[:, None, :]
        
        loss_dict['Diffusion'] = (self.criterion_dict['MSE'](
            prediction_flows,
            flows,
            ) * latent_masks).sum() / latent_masks.sum() / prediction_flows.size(1)
        loss_dict['Attention_Binarization'] = self.criterion_dict['Attention_Binarization'](attention_hards, attention_softs)
        loss_dict['Attention_CTC'] = self.criterion_dict['Attention_CTC'](attention_logprobs, token_lengths, latent_code_lengths)            
        loss_dict['Duration'] = (self.criterion_dict['MSE'](
            (prediction_durations + 1).log(),
            (durations + 1).log(),
            ) * token_masks).sum() / token_masks.sum()
        loss_dict['F0'] = (self.criterion_dict['MSE'](
            (prediction_f0s + 1).log(),
            (f0s + 1).log(),
            ) * latent_masks).sum() / latent_masks.sum()     
        loss_dict['Speaker'] = self.criterion_dict['CE'](
            input= prediction_speakers,
            target= speakers.long(),
            )

        for tag, loss in loss_dict.items():
            self.scalar_dict['Evaluation']['Loss/{}'.format(tag)] += loss.item()

        return alignments

    def Evaluation_Epoch(self):
        logging.info('(Steps: {}) Start evaluation.'.format(self.steps))

        for model in self.model_dict.values():
            model.eval()

        for step, (tokens, token_lengths, languages, latent_codes, latent_code_lengths, f0s, mels, speakers, attention_priors) in tqdm(
            enumerate(self.dataloader_dict['Eval'], 1),
            desc='[Evaluation]',
            total= math.ceil(len(self.dataloader_dict['Eval'].dataset) / self.hp.Train.Batch_Size / self.num_gpus)
            ):
            alignments = self.Evaluation_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                languages= languages,
                latent_codes= latent_codes,
                latent_code_lengths= latent_code_lengths,
                f0s= f0s,
                mels= mels,
                speakers= speakers,
                attention_priors= attention_priors
                )

        if self.accelerator.is_main_process:
            self.scalar_dict['Evaluation'] = {
                tag: loss / step
                for tag, loss in self.scalar_dict['Evaluation'].items()
                }
            self.writer_dict['Evaluation'].add_scalar_dict(self.scalar_dict['Evaluation'], self.steps)
            # self.writer_dict['Evaluation'].add_histogram_model(self.model_dict['RectifiedFlowTTS'], 'RectifiedFlowTTS', self.steps, delete_keywords=[])
        
            index = np.random.randint(0, tokens.size(0))

            with torch.inference_mode():
                if self.num_gpus > 1:
                    target_audios = self.model_dict['RectifiedFlowTTS_EMA'].module.module.hificodec(latent_codes[index, None].mT.to(self.device))[:, 0]
                    inference_func = self.model_dict['RectifiedFlowTTS_EMA'].module.module.Inference
                else:
                    target_audios = self.model_dict['RectifiedFlowTTS_EMA'].module.hificodec(latent_codes[index, None].mT.to(self.device))[:, 0]
                    inference_func = self.model_dict['RectifiedFlowTTS_EMA'].Inference

                prediction_audios, prediction_f0s, prediction_alignments = inference_func(
                    tokens= tokens[index, None].to(self.device),
                    token_lengths= token_lengths[index, None].to(self.device),
                    languages= languages[index].to(self.device),
                    )

            token_length = token_lengths[index].item()
            target_latent_length = latent_code_lengths[index].item()
            prediction_latent_length = prediction_alignments[0, :token_lengths[index], :].sum().long().item()
            target_audio_length = target_latent_length * self.hp.Sound.Hop_Size
            prediction_audio_length = prediction_latent_length * self.hp.Sound.Hop_Size

            target_audio = target_audios[0, :target_audio_length].float().clamp(-1.0, 1.0)
            prediction_audio = prediction_audios[0, :prediction_audio_length].float().clamp(-1.0, 1.0)

            target_mel = self.mel_func(target_audio[None])[0].cpu().numpy()
            prediction_mel = self.mel_func(prediction_audio[None])[0].cpu().numpy()

            target_audio = target_audio.cpu().numpy()
            prediction_audio = prediction_audio.cpu().numpy()

            target_f0 = f0s[index, :target_latent_length].cpu().numpy() 
            prediction_f0 = prediction_f0s[0, :prediction_latent_length].cpu().numpy()

            target_alignment = alignments[index, :token_length, :target_latent_length].cpu().numpy()
            prediction_alignment = prediction_alignments[0, :token_length, :prediction_latent_length].cpu().numpy()

            image_dict = {
                'Mel/Target': (target_mel, None, 'auto', None, None, None),
                'Mel/Prediction': (prediction_mel, None, 'auto', None, None, None),
                'F0/Target': (target_f0, None, 'auto', None, None, None),
                'F0/Prediction': (prediction_f0, None, 'auto', None, None, None),
                'Alignment/Target': (target_alignment, None, 'auto', None, None, None),
                'Alignment/Prediction': (prediction_alignment, None, 'auto', None, None, None),
                }
            audio_dict = {
                'Audio/Target': (target_audio, self.hp.Sound.Sample_Rate),
                'Audio/Linear': (prediction_audio, self.hp.Sound.Sample_Rate),
                }

            self.writer_dict['Evaluation'].add_image_dict(image_dict, self.steps)
            self.writer_dict['Evaluation'].add_audio_dict(audio_dict, self.steps)

            if self.hp.Weights_and_Biases.Use:
                wandb.log(
                    data= {
                        f'Evaluation.{key}': value
                        for key, value in self.scalar_dict['Evaluation'].items()
                        },
                    step= self.steps,
                    commit= False
                    )
                wandb.log(
                    data= {
                        'Evaluation.Mel.Target': wandb.Image(target_mel),
                        'Evaluation.Mel.Prediction': wandb.Image(prediction_mel),
                        'Evaluation.Audio.Target': wandb.Audio(
                            target_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Target_Audio'
                            ),
                        'Evaluation.Audio.Prediction': wandb.Audio(
                            prediction_audio,
                            sample_rate= self.hp.Sound.Sample_Rate,
                            caption= 'Prediction_Audio'
                            ),
                        },
                    step= self.steps,
                    commit= True
                    )

        self.scalar_dict['Evaluation'] = defaultdict(float)

        for model in self.model_dict.values():
            model.train()


    @torch.inference_mode()
    def Inference_Step(
        self,
        tokens: torch.IntTensor,
        token_lengths: torch.IntTensor,
        languages: torch.IntTensor,
        texts: List[str],
        pronunciations: List[str],
        start_index= 0,
        tag_step= False
        ):
        if self.num_gpus > 1:
            inference_func = self.model_dict['RectifiedFlowTTS_EMA'].module.module.Inference
        else:
            inference_func = self.model_dict['RectifiedFlowTTS_EMA'].Inference

        prediction_audios, prediction_f0s, prediction_alignments = inference_func(
            tokens= tokens.to(self.device),
            token_lengths= token_lengths.to(self.device),
            languages= languages.to(self.device),
            )
        latent_code_lengths = [
            alignment[:token_length, :].sum().long()
            for token_length, alignment in zip(token_lengths, prediction_alignments)
            ]
        audio_lengths = [
            length * self.hp.Sound.Hop_Size
            for length in latent_code_lengths
            ]

        prediction_alignments = [
            alignment[:token_length, :mel_length]
            for alignment, token_length, mel_length in zip(prediction_alignments.cpu().numpy(), token_lengths, latent_code_lengths)
            ]

        prediction_mels = [
            mel[:, :length]
            for mel, length in zip(self.mel_func(prediction_audios).cpu().numpy(), latent_code_lengths)
            ]
        prediction_audios = [
            audio[:length]
            for audio, length in zip(prediction_audios.cpu().numpy(), audio_lengths)
            ]
        
        prediction_f0s = [
            f0[:length]
            for f0, length in zip(prediction_f0s.cpu().numpy(), latent_code_lengths)
            ]

        files = []
        for index in range(tokens.size(0)):
            tags = []
            if tag_step: tags.append('Step-{}'.format(self.steps))
            tags.append('IDX_{}'.format(index + start_index))
            files.append('.'.join(tags))

        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG').replace('\\', '/'), exist_ok= True)
        os.makedirs(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV').replace('\\', '/'), exist_ok= True)
        for index, (
            mel,
            audio,
            alignment,
            text,
            pronunciation,
            file
            ) in enumerate(zip(
            prediction_mels,
            prediction_audios,
            prediction_alignments,
            texts,
            pronunciations,
            files
            )):
            title = 'Text: {}'.format(text if len(text) < 90 else text[:90] + '…')
            new_figure = plt.figure(figsize=(20, 5 * 3), dpi=100)

            ax = plt.subplot2grid((3, 1), (0, 0))
            plt.imshow(mel, aspect='auto', origin='lower')
            plt.title(f'Prediction mel  {title}')
            plt.colorbar(ax= ax)
            ax = plt.subplot2grid((3, 1), (1, 0))
            plt.imshow(alignment, aspect= 'auto', origin= 'lower')
            plt.margins(x= 0)
            plt.yticks(
                range(len(pronunciation) + 2),
                ['<S>'] + list(pronunciation) + ['<E>'],
                fontsize = 10
                )
            plt.title(f'Alignment  {title}')
            ax = plt.subplot2grid((3, 1), (1, 0))
            plt.plot(audio)
            plt.title('Prediction Audio    {}'.format(title))
            plt.margins(x= 0)
            plt.colorbar(ax= ax)            
            plt.tight_layout()
            plt.savefig(os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'PNG', '{}.png'.format(file)).replace('\\', '/'))
            plt.close(new_figure)
            
            wavfile.write(
                os.path.join(self.hp.Inference_Path, 'Step-{}'.format(self.steps), 'WAV', '{}.wav'.format(file)).replace('\\', '/'),
                self.hp.Sound.Sample_Rate,
                audio
                )
            
    def Inference_Epoch(self):
        if not self.accelerator.is_main_process:
            return
            
        logging.info('(Steps: {}) Start inference.'.format(self.steps))

        for model in self.model_dict.values():
            model.eval()

        batch_size = self.hp.Inference_Batch_Size or self.hp.Train.Batch_Size
        for step, (tokens, token_lengths, languages, texts, pronunciations) in tqdm(
            enumerate(self.dataloader_dict['Inference']),
            desc='[Inference]',
            total= math.ceil(len(self.dataloader_dict['Inference'].dataset) / batch_size)
            ):
            self.Inference_Step(
                tokens= tokens,
                token_lengths= token_lengths,
                languages= languages,
                texts= texts,
                pronunciations= pronunciations,
                start_index= step * batch_size
                )

        for model in self.model_dict.values():
            model.train()

    def Load_Checkpoint(self):
        if not os.path.exists(self.hp.Checkpoint_Path):
            return
        elif self.steps == 0:
            paths = [
                os.path.join(self.hp.Checkpoint_Path, path).replace('\\', '/')
                for path in os.listdir(self.hp.Checkpoint_Path)
                if path.startswith('S_')
                ]
            if len(paths) > 0:
                path = max(paths, key = os.path.getctime)
                self.steps = int(path.split('_')[1])
            else:
                return  # Initial training
        else:
            path = os.path.join(self.hp.Checkpoint_Path, 'S_{}'.format(self.steps).replace('\\', '/'))

        self.accelerator.load_state(path)

        if self.accelerator.is_main_process:
            logging.info('Checkpoint loaded at {} steps.'.format(self.steps))

    def Save_Checkpoint(self):        
        if not self.accelerator.is_main_process:
            return
        
        os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
        checkpoint_path = os.path.join(self.hp.Checkpoint_Path, 'S_{}'.format(self.steps).replace('\\', '/'))

        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(checkpoint_path, safe_serialization= False) # sefetensor cannot use because of shared tensor problem
        # self.accelerator.save_state(checkpoint_path)

        logging.info('Checkpoint saved at {} steps.'.format(self.steps))

        if all([
            self.hp.Weights_and_Biases.Use,
            self.hp.Weights_and_Biases.Save_Checkpoint.Use,
            self.steps % self.hp.Weights_and_Biases.Save_Checkpoint.Interval == 0
            ]):
            wandb.save(checkpoint_path)

    def Train(self):
        hp_path = os.path.join(self.hp.Checkpoint_Path, 'Hyper_Parameters.yaml').replace('\\', '/')
        if not os.path.exists(hp_path):
            from shutil import copyfile
            os.makedirs(self.hp.Checkpoint_Path, exist_ok= True)
            copyfile(self.hp_path, hp_path)
        
        if self.steps == 0:
            self.Evaluation_Epoch()

        if self.hp.Train.Initial_Inference:
            self.Inference_Epoch()

        self.tqdm = tqdm(
            initial= self.steps,
            total= self.hp.Train.Max_Step,
            desc='[Training]'
            )

        while self.steps < self.hp.Train.Max_Step:
            try:
                self.Train_Epoch()
            except KeyboardInterrupt:
                self.Save_Checkpoint()
                exit(1)
            
        self.tqdm.close()
        logging.info('Finished training.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    parser.add_argument('-s', '--steps', default= 0, type= int)    
    parser.add_argument('-r', '--local_rank', default= 0, type= int)
    args = parser.parse_args()
    
    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    os.environ['CUDA_VISIBLE_DEVICES'] = hp.Device
    
    trainer = Trainer(hp_path= args.hyper_parameters, steps= args.steps)
    trainer.Train()

# accelerate launch Train.py -hp Hyper_Parameters.yaml