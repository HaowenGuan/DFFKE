# The folder to store the pre-trained StyleGAN-3

`cd` into this folder and run the following command to download the pre-trained StyleGAN-3 model:

## Pretrained on Bench
```bash
wget https://g-75671f.f5dc97.75bc.dn.glob.us/benches/network-snapshot-011000.pkl
mv network-snapshot-011000.pkl Benches-512.pkl
```

## Pretrained on AFHQv2
```bash
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-afhqv2-512x512.pkl
mv stylegan3-t-afhqv2-512x512.pkl AFHQv2-512.pkl
```

## Pretrained on FFHQ-U
```bash
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl
mv stylegan3-r-ffhqu-256x256.pkl FFHQ-U-256.pkl
```
