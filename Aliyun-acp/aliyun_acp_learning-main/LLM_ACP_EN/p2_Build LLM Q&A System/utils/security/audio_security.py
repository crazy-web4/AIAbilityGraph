# coding=utf-8
# python version >= 3.6
import time

from alibabacloud_green20220302.client import Client
from alibabacloud_green20220302 import models
from alibabacloud_tea_openapi.models import Config
import json
import os

access_key_id = os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
access_key_secret = os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']


config = Config(
    # The AccessKey of your Alibaba Cloud account has access to all APIs. It is recommended to use a RAM user for API access or daily operations.
    # Strongly recommend not saving the AccessKey ID and AccessKey Secret in the project code, otherwise it may lead to AccessKey leakage, threatening the security of all resources under your account.
    # Common ways to get environment variables:
    # Get RAM user AccessKey ID: os.environ['ALIBABA_CLOUD_ACCESS_KEY_ID']
    # Get RAM user AccessKey Secret: os.environ['ALIBABA_CLOUD_ACCESS_KEY_SECRET']
    access_key_id=access_key_id,
    access_key_secret=access_key_secret,
    # Connection timeout, in milliseconds (ms).
    connect_timeout=10000,
    # Read timeout, in milliseconds (ms).
    read_timeout=3000,
    region_id='cn-shanghai',
    endpoint='green-cip.cn-shanghai.aliyuncs.com'
)

# Note: The instantiated client should be reused as much as possible to improve detection performance. Avoid repeatedly establishing connections.
client = Client(config)


def submit_task(audio_url):
    serviceParameters = {
        'url': audio_url,
    }
    voiceModerationRequest = models.VoiceModerationRequest(
        # Detection type: audio_media_detection indicates audio file detection, live_stream_detection indicates live audio stream detection.
        service='audio_media_detection',
        service_parameters=json.dumps(serviceParameters)
    )

    try:
        response = client.voice_moderation(voiceModerationRequest)
        if response.status_code == 200:
            # Call succeeded
            result = response.body
            print('audio submit task:{}'.format(result.data))
            # Return task_id
            return result.data.task_id
        else:
            print('audio submit task fail. status:{} ,result:{}'.format(response.status_code, response))
    except Exception as err:
        print(err)


def get_result(task_id):
    # taskId returned when submitting the task.
    service_parameters = {
        "taskId": task_id
    }
    voice_moderation_result_request = models.VoiceModerationResultRequest(
        # Detection type.
        service='audio_media_detection',
        service_parameters=json.dumps(service_parameters)
    )

    try:
        response = client.voice_moderation_result(voice_moderation_result_request)
        if response.status_code == 200:
            # Get the review result
            result = response.body
            print('audio detect result:{}'.format(result.data))
        else:
            print('audio detect fail. status:{} ,result:{}'.format(response.status_code, response))
    except Exception as err:
        print(err)


def detect(audio_url):
    task_id = submit_task(audio_url)
    # Wait for 3 seconds before querying
    time.sleep(3)
    # Query results based on task id
    result = get_result(task_id)
    # Return a common structure


if __name__ == "__main__":
    # audio_url = ''
    # submit_task(audio_url)
    task_id = 'au_f_vrex9uxM7MXc8flPCiOK5V-1AzKWX'
    get_result(task_id)