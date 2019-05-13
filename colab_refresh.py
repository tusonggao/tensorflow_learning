import time
import os
import win32api

urls_to_open = ['https://colab.research.google.com/drive/1F2_7Zf4OBBF1iRsYQO7g0CuPG6CMV122#scrollTo=NEQ4JE_2SS_N']
browser_path = 'C:/Users/tusonggao/AppData/Local/Google/Chrome/Application/chrome.exe'

num = 0
while num < 1000:
    num += 1
    for url in urls_to_open:
        win32api.WinExec(browser_path + ' ' + url)
    print('open a url, num: ', num)
    time.sleep(30*60)
    # time.sleep(10)
    # os.system('taskkill / IM iexplore.exe')

print('prog finished!')