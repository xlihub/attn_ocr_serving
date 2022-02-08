ssh://root@localhost:10022/usr/bin/pkill -f web_service
sleep(10)
ssh://root@localhost:10022/usr/bin/python -u /home/PaddleOCR/deploy/pdserving/web_service.py &>log.txt&

