
# funciona en python 3.11

git clone https://github.com/JoseGBarroso/Accuro_tech_faceRecognition.git
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
pip install -r requirements.txt
cd trained_model
unzip face_recognition_model
cd ..
python .\main.py
