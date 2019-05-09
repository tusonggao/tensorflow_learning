import smtplib
from email import encoders
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr


def send_email(from_addr, to_addr, password, subject, body_text):
    msg = MIMEText(body_text,'html','utf-8')
    msg['From'] = u'<%s>' % from_addr
    msg['To'] = u'<%s>' % to_addr
    msg['Subject'] = subject

    # smtp = smtplib.SMTP_SSL('smtp.163.com', 465)   #for 163.com邮箱
    smtp = smtplib.SMTP_SSL('smtp.qq.com', 465)  #for qq.com邮箱
    smtp.set_debuglevel(1)
    # smtp.ehlo("smtp.163.com")   #for 163.com邮箱
    smtp.ehlo('smtp.qq.com')  #for qq.com邮箱
    smtp.login(from_addr, password)
    smtp.sendmail(from_addr, [to_addr], msg.as_string())


body_text = '自动发送的邮件正文'

# send_email("tusonggao@163.com", "360426145@qq.com", "tusg070185", "55555 from qq", body_text)
send_email("360426145@qq.com", "tusonggao@163.com", "zzmaqtctbweobigd", "55555 from qq", body_text)


