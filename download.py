
from datetime import datetime, date
from contextlib import closing
from tqsdk import TqApi, TqAuth, TqSim
from tqsdk.tools import DataDownloader

# 输入你的天勤tqsdk账号密码
api = TqApi(auth=TqAuth("", ""))
download_tasks = {}

download_tasks["T_tick"] = DataDownloader(api, symbol_list=["SHFE.rb2501"], dur_sec=0,
                    start_dt=datetime(2024, 7, 1), end_dt=datetime(2024, 9, 10), csv_file_name="rb2501_tick.csv")
# 使用with closing机制确保下载完成后释放对应的资源
with closing(api):
    while not all([v.is_finished() for v in download_tasks.values()]):
        api.wait_update()
        print("progress: ", { k:("%.2f%%" % v.get_progress()) for k,v in download_tasks.items() })