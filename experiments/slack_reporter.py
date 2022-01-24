"""
A reporter object to send messages to a Slack channel.

Copyright brobeson
"""

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import yaml


class SlackReporter:
    """
    Send messages to a Slack channel.

    Arguments:
        source (str): An arbitrary string that communicates the source of the notification.
        channel (str): The report sends notifications to this Slack channel.
        token (str): The security API token to use for sending notifications to the ``channel``.
    """

    def __init__(self, source: str, channel: str, token: str) -> None:
        self.__source = source
        self.__channel = channel
        self.__client = WebClient(token=token)
        self.__ts = None

    def send_message(self, message: str) -> None:
        """
        Send a message to the Slack channel.

        Arguments:
            message (str): The message to send.
        """
        if self.__ts is None:
            self.__start_new_thread(message)
        else:
            self.__reply_to_thread(message)

    def __start_new_thread(self, message: str) -> None:
        try:
            response = self.__client.chat_postMessage(
                channel=self.__channel, text=self.__format_message(message),
            )
            self.__ts = response["ts"]
        except SlackApiError as e:
            print(e)

    def __reply_to_thread(self, message: str) -> None:
        try:
            self.__client.chat_postMessage(
                channel=self.__channel, thread_ts=self.__ts, text=self.__format_message(message),
            )
        except SlackApiError as e:
            print(e)

    def __format_message(self, message: str) -> str:
        return f"[ {self.__source} ]  {message}"


def read_slack_configuration(filepath: str) -> dict:
    """
    Read a Slack report configuration from a YAML file on disk.

    Args:
        filepath (str): The path to the configuration file.

    Returns:
        dict: The configuration as a dictionary. The keys are:

        =========== ===============================================
        ``token``   The API token to use for sending notifications.
        ``channel`` The Slack channel to send notifications to.
        =========== ===============================================
    """
    with open(filepath, "r") as configuration_file:
        configuration = yaml.safe_load(configuration_file)
    return configuration
