
from datetime import datetime, date
from contextlib import closing
from tqsdk import TqApi, TqAuth, TqSim
from tqsdk.tools import DataDownloader

import os
api = TqApi(auth=TqAuth(os.environ['tqname'], os.environ['tqpassword']))

exchange = "INE"
future = "nr2412"
symbol = f"{exchange}.{future}"
start_date = datetime(2024, 9, 26)
end_date = datetime(2024, 10, 24)



download_tasks = {}
download_tasks[future] = DataDownloader(api, symbol_list=[symbol], 
                                        dur_sec=0,
                                        start_dt=start_date, 
                                        end_dt=end_date, 
                                        csv_file_name=f'{future}.csv')

with closing(api):
    while not all([v.is_finished() for v in download_tasks.values()]):
        api.wait_update()
        print("progress: ", { k:("%.2f%%" % v.get_progress()) for k,v in download_tasks.items() })