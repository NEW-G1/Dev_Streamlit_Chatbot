import os

def install_ffmpeg():
    os.system('apt-get update')
    os.system('apt-get install -y ffmpeg')

# 앱 시작 부분에 호출
install_ffmpeg()