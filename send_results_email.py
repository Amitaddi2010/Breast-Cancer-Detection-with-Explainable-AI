"""
send_results_email.py — Prepares the final results email and attempts to send it.
If SMTP is not locally configured, it automatically compiles a clean draft at outputs/results_email_draft.txt.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_results_email():
    sender = "repsegnetv2@ultrasound.ai"
    receivers = ["amit.addi2010@gmail.com", "amitraajsaraswat@gmail.com"]
    subject = "RepSegNet V2 Pipeline: Training and Evaluation Complete"
    
    # Read the updated paper section report
    report_path = os.path.join("outputs", "RepSegNet_V2_Paper_Section.md")
    body = "Hello Amit,\n\nRepSegNet V2 training and leak-proof comparative evaluations have completed successfully!\n\nHere is the publication-ready draft section containing all results and methodology details:\n\n"
    
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            body += f.read()
    else:
        body += "The paper draft section could not be found."
        
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ", ".join(receivers)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        # Try local SMTP relay
        with smtplib.SMTP('localhost', 25) as server:
            server.sendmail(sender, receivers, msg.as_string())
        print("[Email] Email sent successfully via local SMTP relay!")
    except Exception as e:
        # Save a copy as a local text draft that the user can copy immediately
        draft_path = os.path.join("outputs", "results_email_draft.txt")
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(f"To: {', '.join(receivers)}\nSubject: {subject}\n\n{body}")
        print(f"[Email OK] Local SMTP relay not configured. Saved compiled email draft to {draft_path}")

if __name__ == "__main__":
    send_results_email()
