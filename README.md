# Rethinking Perturbations in Encoder-Decoders for Fast Training

This repository contains transformers with perturbations used in our paper except for adversarial perturbations.

>[Rethinking Perturbations in Encoder-Decoders for Fast Training](https://arxiv.org/abs/2104.01853)

>Sho Takase, Shun Kiyono

>Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies



In addition, this document provides the way to train WMT En-De and IWSLT De-En with Rep(sim)+WDrop as examples.

For adversarial perturbations, please ask @aonotas.


## Requirements

- PyTorch version == 1.4.0
- Python version >= 3.6


## WMT En-De

### Training

##### 1. Download and pre-process datasets following the description in [this page](https://github.com/pytorch/fairseq/tree/master/examples/scaling_nmt)

##### 2. Train model

Run the following command on 4 GPUs.

```bash
python -u train.py \
    pre-processed-data-dir \
    --arch transformer_vaswani_wmt_en_de_big --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --warmup-init-lr 1e-07 --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --max-tokens 3584 --min-lr 1e-09 --update-freq 32  --log-interval 100  --max-update 50000 \
    --sampling-method worddrop_with_sim --enc-replace-rate 0.1 --dec-replace-rate 0.1 --decay-val 1000 \
    --share-all-embeddings --keep-last-epochs 20 --seed 1 --save-dir model-save-dir
```

If training diverges, please set `--clip-norm` to 1.0.

`--sampling-method` specifies the type of the perturbations.
To use other perturbations, check the following list:

* Rep(Uni): `uniform`
* Rep(Sim): `similarity`
* WDrop: `worddrop`
* Rep(Uni)+WDrop: `worddrop_with_uni`
* Rep(Sim)+WDrop: `worddrop_with_sim`
* Rep(SS): `condprob`

### Test (decoding)

Averaging latest 10 checkpoints.

```bash
python scripts/average_checkpoints.py --inputs model-save-dir --num-epoch-checkpoints 10 --output model-save-dir/averaged.pt
```

Decoding with the averaged checkpoint.

```bash
python generate.py pre-processed-data-dir --path model-save-dir/averaged.pt  --beam 4 --lenpen 0.6 --remove-bpe | grep '^H' | sed 's/^H\-//g' | sort -t ' ' -k1,1 -n | cut -f 3- > generated.result
```

* We used ```--lenpen 0.6``` for newstest2014, and ```--lenpen 1.0``` for otherwise.


### Compute SacreBLEU score

Detokenize the generated result.

```bash
cat generated.result | $mosesscripts/tokenizer/detokenizer.perl -l de > generated.result.detok
```

* mosesscripts is the PATH to mosesdecoder/scripts

Compute SacreBLEU.

```bash
cat generated.result.detok | sacrebleu -t wmt14/full -l en-de
```

## IWSLT De-En

### Training

##### 1. Download and pre-process datasets

* Download dataset with [this script](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh) to ```IWSLT_DATA```

* Pre-processing with the following command.

```bash
python preprocess.py --source-lang de --target-lang en \
    --trainpref IWSLT_DATA/train --validpref IWSLT_DATA/valid --testpref IWSLT_DATA/test \
    --joined-dictionary --workers 20 \
    --destdir IWSLT_DATA_BIN
```

##### 2. Training

Run the following command on 1 GPU.

```bash
python -u train.py \
    IWSLT_DATA_BIN \
    --arch transformer_iwslt_de_en --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --max-tokens 4096 --log-interval 100  --max-update 100000 \
    --sampling-method worddrop_with_sim_enc_drop --enc-replace-rate 0.1 --dec-replace-rate 0.1 --decay-val 1000 \
    --share-all-embeddings --keep-last-epochs 20 --seed 1 --save-dir model-save-dir
```

### Test (decoding)

Averaging latest 10 checkpoints.

```bash
python scripts/average_checkpoints.py --inputs model-save-dir --num-epoch-checkpoints 10 --output model-save-dir/averaged.pt
```

Decoding with the averaged checkpoint.

```bash
python generate.py IWSLT_DATA_BIN --path model-save-dir/averaged.pt  --beam 5 --remove-bpe | grep '^H' | sed 's/^H\-//g' | sort -t ' ' -k1,1 -n | cut -f 3- > generated.result
```

### Compute BLEU

```bash
cat generated.result | $mosesscripts/generic/multi-bleu.perl IWSLT_DATA/test.en.tokenized
```

## Acknowledgements

A large portion of this repo is borrowed from [fairseq](https://github.com/pytorch/fairseq).
