sudo apt update && sudo apt upgrade
sudo apt install -y python-pip
sudo apt install -y libsm6 libxext6
sudo apt install -y libfontconfig1 libxrender1
sudo apt install -y tesseract-ocr
pip install virtualenvwrapper

workon ...
pip install pillow pytesseract python-dotenv opencv-python scipy