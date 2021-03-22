from paramiko import SSHClient, AutoAddPolicy
import sqlite3 as sq
import pandas as pd

ssh_private_key = "/home/icaro/.ssh/id_rsa.pub"
host = "192.168.0.60"
user="pi"

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.set_missing_host_key_policy(AutoAddPolicy())
ssh.connect(host, username=user)
sftp_client = ssh.open_sftp()

#sftp_client.put("predict.html", "/home/pi/bmpbot/predict.html")
sftp_client.put("predict3.html", "/home/pi/bmpbot/predict.html")