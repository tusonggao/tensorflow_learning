import time
import os
import win32api

urls_to_open = ['https://colab.research.google.com/drive/1VVhv7byQBH8K7HJW_t2srOKoQOOYctY4#scrollTo=EAn4Hpz2pEM9']
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