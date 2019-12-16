#  Copyright (c) 2019 École Polytechnique
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this file, you can obtain one at http://mozilla.org/MPL/2.0
#
#  Authors:
#        Luciano Di Palma <luciano.di-palma@polytechnique.edu>
#        Enhui Huang <enhui.huang@polytechnique.edu>
#
#  Description:
#  AIDEme is a large-scale interactive data exploration system that is cast in a principled active learning (AL) framework: in this context,
#  we consider the data content as a large set of records in a data source, and the user is interested in some of them but not all.
#  In the data exploration process, the system allows the user to label a record as “interesting” or “not interesting” in each iteration,
#  so that it can construct an increasingly-more-accurate model of the user interest. Active learning techniques are employed to select
#  a new record from the unlabeled data source in each iteration for the user to label next in order to improve the model accuracy.
#  Upon convergence, the model is run through the entire data source to retrieve all relevant records.

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from string import Template
from datetime import datetime
from .utils import get_config_from_resources

ERROR_MESSAGE = Template(
    """New experiment update on ${HOSTNAME}.
    Status: ERROR
    Dataset: ${DATA_TAG}
    Learner: ${LEARNER_TAG}
    Error Message:
    ${TRACEBACK}""")

END_MESSAGE = Template(
    """New experiment update on ${HOSTNAME}.
    Status: FINISHED
    Experiment summary: ${SUMMARY}""")


class EmailSender:
    def __init__(self):
        config = get_config_from_resources('email', '')
        self.sender_email = config['sender_email']
        self.recipient_email = config['recipient_email']
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

    def send_end_email(self, summary):
        body = END_MESSAGE.substitute(HOSTNAME=self.get_hostname(), SUMMARY=summary)
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
