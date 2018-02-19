import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from string import Template
from datetime import datetime
from ..config.utils import get_config_from_file

ERROR_MESSAGE = Template(
    """New update on experiment running at ${HOSTNAME}.
    Status: ERROR
    Dataset: ${DATA_TAG}
    Learner: ${LEARNER_TAG}
    Error Message:
    ${TRACEBACK}""")

END_MESSAGE = Template(
    """New update on experiment running at ${HOSTNAME}.
    Status: FINISHED ${SPECIFIER}
    Experiment summary: ${SUMMARY}""")


class EmailSender:
    def __init__(self):
        config = get_config_from_file('email.yml')
        self.sender_email = config['sender_email']
        self.recipient_email = config['receipient_email']
        self.active = config['active']

        # create SMTP server
        if self.active:
            self.server = smtplib.SMTP(config['service'], config['port'])
            self.server.ehlo()
            self.server.starttls()
            self.server.login(config['username'], config['sender_password'])

    def get_hostname(self):
        import socket
        return socket.gethostname()

    def send_error_email(self, data_tag, learner_tag, traceback):
        body = ERROR_MESSAGE.substitute(HOSTNAME=self.get_hostname(), DATA_TAG=data_tag, LEARNER_TAG=learner_tag,
                                        TRACEBACK=traceback)
        self.send_message(body)

    def send_end_email(self, specifier, summary):
        body = END_MESSAGE.substitute(HOSTNAME=self.get_hostname(), SUMMARY=summary, SPECIFIER=specifier)
        self.send_message(body)

    def send_message(self, body):
        if not self.active:  # if not active, do not send email
            return

        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = self.recipient_email
        msg['Subject'] = "Experiment status update at {0}".format(datetime.now())
        msg.attach(MIMEText(body, 'plain'))
        self.server.sendmail(self.sender_email, self.recipient_email, msg.as_string())

    def quit(self):
        if self.active:
            self.server.quit()