import smtplib
import sys
from email.message import EmailMessage

def send_mail(subject, body, to_email):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = "o.fox3@newcastle.ac.uk"
    msg['To'] = to_email

    try:
        with smtplib.SMTP('smtp.ncl.ac.uk', 25) as s:
            s.send_message(msg)
    except Exception as e:
        print(f"Failed to send: {e}")

if __name__ == "__main__":
    # Usage: python3 send_status.py "Subject" "Body"
    send_mail(sys.argv[1], sys.argv[2], "o.fox3@newcastle.ac.uk")