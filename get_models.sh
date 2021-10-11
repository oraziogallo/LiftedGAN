MODELS_DIR="pretrained"
mkdir -p $MODELS_DIR

cd $MODELS_DIR

# StyleGAN
gdown https://drive.google.com/u/0/uc?id=1qVoWu_fps17iTzYptwuN3ptgYeCpIl2e -O pretrained.zip
unzip pretrained.zip

# LiftedGAN
gdown https://drive.google.com/u/0/uc?id=1-44Eivt7GHINkX6zox89HHttujYWThz2 -O pretrained_liftedgan.zip
unzip pretrained_liftedgan.zip
