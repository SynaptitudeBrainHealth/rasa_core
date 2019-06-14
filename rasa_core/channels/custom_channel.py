"""Custom Channel script.

This module consists of the implementation of custom channel used
to accept Fitbit notifications from the backend.
"""

# Standard imports
import json
import inspect
from typing import Text, Optional, List
import logging
import asyncio
from asyncio import Queue, CancelledError
# Third party imports
from threading import Thread
from sanic import Sanic, Blueprint, response
from rasa_core import utils
from rasa_core.channels.channel import InputChannel, UserMessage, QueueOutputChannel, RestInput
from raven.handlers.logging import SentryHandler
from raven import Client
from raven.contrib.flask import Sentry
from raven.conf import setup_logging
from flask.json import jsonify
# Local imports

sentry = Sentry(dsn="https://d0d37fc0964e41b699ac5fe456218ae0:f8bd8517dfb54e5bb124f1a99fc682f4@sentry.io/1318460",logging=True)
client = Client(dsn="https://d0d37fc0964e41b699ac5fe456218ae0:f8bd8517dfb54e5bb124f1a99fc682f4@sentry.io/1318460")
handler = SentryHandler(client)
handler.setLevel(logging.INFO)
setup_logging(handler)

logger = logging.getLogger(__name__)

SLACK = False
TWILIO = True
class BackendInput(InputChannel):
    """A custom http input channel for accepting requests from the 
    backend and sending the response back to our primary channel 
    (Slack or Twilio).
    """

    @classmethod
    def name(cls):
        return "backend"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        if SLACK:
            # For using Slack as a primary channel
            return cls(slack_token=credentials.get("slack_token"))

        elif TWILIO:
            # For using Twilio as a primary channel
            return cls(account_sid=credentials.get("account_sid"),
                       auth_token=credentials.get("auth_token"),
                       twilio_number=credentials.get("twilio_number"))


    def __init__(self, slack_token=None, account_sid=None, auth_token=None, twilio_number=None,
                 debug_mode=True):
        if SLACK:
            # For using Slack as a primary channel
            self.slack_token = slack_token

        elif TWILIO:
            # For using Twilio as a primary channel
            self.account_sid = account_sid
            self.auth_token = auth_token
            self.twilio_number = twilio_number
            self.debug_mode = debug_mode

    @staticmethod
    async def on_message_wrapper(on_new_message, text, queue, sender_id):
        collector = QueueOutputChannel(queue)

        message = UserMessage(text, collector, sender_id,
                              input_channel=BackendInput.name())
        await on_new_message(message)

        await queue.put("DONE")

    async def _extract_sender(self, req):
        return req.json.get("sender", None)

    # noinspection PyMethodMayBeStatic
    def _extract_message(self, req):
        return req.json.get("message", None)

    def stream_response(self, on_new_message, text, sender_id):
        async def stream(resp):
            q = Queue()
            task = asyncio.ensure_future(
                self.on_message_wrapper(on_new_message, text, q, sender_id))
            while True:
                result = await q.get()
                if result == "DONE":
                    break
                else:
                    await resp.write(json.dumps(result) + "\n")
            await task

        return stream

    def blueprint(self, on_new_message):
        custom_webhook = Blueprint(
            'custom_webhook_{}'.format(type(self).__name__),
            inspect.getmodule(self).__name__)

        @custom_webhook.route("/", methods=['GET'])
        async def health(request):
            return jsonify({"status": "ok"})

        @custom_webhook.route("/webhook", methods=['POST'])
        async def receive(request):
            from rasa_core.channels.slack import SlackBot
            from rasa_core.channels.twilio import TwilioOutput
            app = request.app
            app.add_task
            sender_id = await self._extract_sender(request)
            text = self._extract_message(request)
            should_use_stream = utils.bool_arg(request, "stream", default=False)

            if should_use_stream:
                return response.stream(
                    self.stream_response(on_new_message, text, sender_id),
                    content_type='text/event-stream')
            else:
                if SLACK:
                    ## For using Slack as a primary channel
                    out_channel = SlackBot(self.slack_token)
                elif TWILIO:
                    ## For using Twilio as a primary channel
                    out_channel = TwilioOutput(self.account_sid, self.auth_token,
                        self.twilio_number)

                try:
                    app = request.app
                    app.add_task(on_new_message(UserMessage(text, out_channel, sender_id,
                                               input_channel=self.name())))
                    # loop = asyncio.get_event_loop()
                    # loop.create_task(on_new_message(UserMessage(text, out_channel, sender_id,
                    #                            input_channel=self.name())))
                except Exception as e:
                    logger.error("Exception when trying to handle "
                                 "message.{0}".format(e))
                    logger.debug(e, exc_info=True)

                return response.text("success")

        return custom_webhook