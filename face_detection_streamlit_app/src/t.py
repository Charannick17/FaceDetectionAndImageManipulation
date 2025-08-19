from torchvision.datasets.utils import download_url

urls = {
    "Hayao": "http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hayao_net_G_float.pth",
    "Hosoda": "http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hosoda_net_G_float.pth",
    "Paprika": "http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Paprika_net_G_float.pth",
    "Shinkai": "http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Shinkai_net_G_float.pth",
}

for style, url in urls.items():
    download_url(url, root="models", filename=f"{style}_net_G_float.pth")
