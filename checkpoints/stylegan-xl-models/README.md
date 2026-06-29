# The folder to store the pre-trained StyleGAN-XL

`cd` into this folder and run the following command to download the pre-trained StyleGAN-XL model:

```bash
wget https://s3.eu-central-1.amazonaws.com/avg-projects/stylegan_xl/models/imagenet64.pkl
```

## Note

To successfully load this model for FedKTL, it requires an old version of timm package [[Source]](https://github.com/autonomousvision/stylegan-xl/issues/111).

```base
pip install timm==0.4.12
```
