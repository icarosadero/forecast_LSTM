notify-send "BMP Forecast" "Started training"
python3 database.py
python3 train.py
python3 train2.py
python3 insert.py
notify-send "BMP Forecast" "Completed"